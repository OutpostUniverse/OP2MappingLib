// Learned map generator, part 3: large shape, calmer biomes and terrain, the
// gradient, lava rock and volcano sites (see learned_extra.hpp).
// Ported from fefi-byte's op2-codegen-editor (MIT, see THIRD-PARTY-NOTICES.md).
#include "learned_extra.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>

namespace om::gen {

// ---------------------------------------------------------------------------
// Gradient (Python: mapgen_basis.verlauf_feld, verlauf_gelaende)

bool gradientField(const Params& p, int w, int h, int margin, Floats& u) {
    if (p.gradient < 1 || p.gradient > 10) return false;
    const std::string name = kGradients[p.gradient - 1];
    const std::string a = name.substr(0, name.find('-')), b = name.substr(name.find('-') + 1);
    u = Floats(w, h);
    const double H = std::max(1, h - 2 * margin), W = std::max(1, w - 2 * margin);
    const bool radial = name == "mitte-rand" || name == "rand-mitte";
    int dx = 0, dy = 0;
    auto add = [&](const std::string& s, int sign) {
        for (char c : s) {
            if (c == 'n') dy -= sign;
            if (c == 's') dy += sign;
            if (c == 'w') dx -= sign;
            if (c == 'o') dx += sign;
        }
    };
    add(b, 1);
    add(a, -1);
    const double n = std::max(1, std::abs(dx) + std::abs(dy));
    for (int yy = 0; yy < h; ++yy)
        for (int xx = 0; xx < w; ++xx) {
            const double y = (yy - margin + .5) / H, x = (xx - margin + .5) / W;
            double v;
            if (radial) {
                v = std::sqrt((x - .5) * (x - .5) + (y - .5) * (y - .5)) / .6;
                if (a != "mitte") v = 1 - v;
            } else {
                v = (x - .5) * dx / n + (y - .5) * dy / n + .5;
            }
            u.at(xx, yy) = std::clamp(v, 0.0, 1.0);
        }
    return true;
}

bool gradientTerrain(const Params& p, int w, int h, int margin, Floats& rough) {
    Floats u;
    if (!gradientField(p, w, h, margin, u)) return false;
    const double a = std::clamp(p.gradientTerrain[0], 0.0, 1.0), b = std::clamp(p.gradientTerrain[1], 0.0, 1.0);
    rough = Floats(w, h);
    for (size_t i = 0; i < u.v.size(); ++i) rough.v[i] = (1 - u.v[i]) * a + u.v[i] * b;
    return true;
}

namespace extra {

namespace {

/// Directions a wall falls towards: E, SE, S, SW, W, NW, N, NE, normalised.
const double kRx[8] = {1, 0.70710678118654752440, 0, -0.70710678118654752440, -1, -0.70710678118654752440, 0,
                       0.70710678118654752440};
const double kRy[8] = {0, 0.70710678118654752440, 1, 0.70710678118654752440, 0, -0.70710678118654752440, -1,
                       -0.70710678118654752440};

inline int roundEven(double v) { return static_cast<int>(std::nearbyint(v)); }

Mask maskOf(const Grid<int32_t>& g, int lo, int hi) {   // lo <= v < hi
    Mask m(g.w, g.h);
    for (size_t i = 0; i < m.v.size(); ++i) m.v[i] = g.v[i] >= lo && g.v[i] < hi;
    return m;
}

Mask invert(const Mask& m) {
    Mask o(m.w, m.h);
    for (size_t i = 0; i < m.v.size(); ++i) o.v[i] = !m.v[i];
    return o;
}

/// _glatt: majority of the 3x3 neighbours (edge padded), `times` times.
Mask majority3(Mask m, int times) {
    for (int it = 0; it < times; ++it) {
        Mask o(m.w, m.h);
        for (int y = 0; y < m.h; ++y)
            for (int x = 0; x < m.w; ++x) {
                int s = 0;
                for (int a = -1; a <= 1; ++a)
                    for (int b = -1; b <= 1; ++b) s += m.at(std::clamp(x + b, 0, m.w - 1), std::clamp(y + a, 0, m.h - 1));
                o.at(x, y) = s >= 5;
            }
        m = std::move(o);
    }
    return m;
}

/// _treppe: coarse raster, the cliffs run in straight steps as in the originals.
Mask stairs(const Mask& m, int n) {
    Mask o = m;
    const int a = m.h / n * n, b = m.w / n * n;
    for (int y = 0; y < a; y += n)
        for (int x = 0; x < b; x += n) {
            int s = 0;
            for (int dy = 0; dy < n; ++dy)
                for (int dx = 0; dx < n; ++dx) s += m.at(x + dx, y + dy);
            const uint8_t v = 2 * s >= n * n;
            for (int dy = 0; dy < n; ++dy)
                for (int dx = 0; dx < n; ++dx) o.at(x + dx, y + dy) = v;
        }
    return o;
}

/// _rand: enters the rim of the high side as a cliff falling away from it.
void rim(Grid<int32_t>& wall, const Mask& high, const Mask& where) {
    const int w = high.w, h = high.h;
    Floats f(w, h);
    for (size_t i = 0; i < f.v.size(); ++i) f.v[i] = high.v[i];
    const Floats s = blur3(f, 1);
    auto at = [&](int x, int y) { return high.at(std::clamp(x, 0, w - 1), std::clamp(y, 0, h - 1)); };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (!high.at(x, y) || !where.at(x, y)) continue;
            if (at(x, y - 1) && at(x, y + 1) && at(x - 1, y) && at(x + 1, y)) continue;
            // numpy.gradient: central differences, one-sided at the edges.
            const double gx = w < 2 ? 0 : x == 0 ? s.at(1, y) - s.at(0, y)
                                    : x == w - 1 ? s.at(x, y) - s.at(x - 1, y)
                                                 : (s.at(x + 1, y) - s.at(x - 1, y)) / 2;
            const double gy = h < 2 ? 0 : y == 0 ? s.at(x, 1) - s.at(x, 0)
                                    : y == h - 1 ? s.at(x, y) - s.at(x, y - 1)
                                                 : (s.at(x, y + 1) - s.at(x, y - 1)) / 2;
            const double vx = -gx, vy = -gy;
            if (std::sqrt(vx * vx + vy * vy) <= 1e-6) continue;
            int best = 0;
            double bv = -1e300;
            for (int r = 0; r < 8; ++r) {
                const double d = kRx[r] * vx + kRy[r] * vy;
                if (d > bv) { bv = d; best = r; }
            }
            wall.at(x, y) = best + 1;
        }
}

/// _tiefe: distance of every cell from the mask's rim; the map edge does not count.
Floats depth(const Mask& m, int most) {
    Floats d(m.w, m.h, 0.0);
    Mask core = m;
    for (int i = 0; i < most; ++i) {
        if (!core.any()) break;
        for (size_t k = 0; k < d.v.size(); ++k) d.v[k] += core.v[k];
        const Mask grown = dilate4(invert(core), 1);
        for (size_t k = 0; k < core.v.size(); ++k) core.v[k] = core.v[k] && !grown.v[k];
    }
    return d;
}

/// _bergterrassen: nested contours inside a mountain instead of single pieces.
void mountainTerraces(Grid<int32_t>& wall, const Mask& tight, double density, Rng& rng, double calm) {
    if (!tight.any()) return;
    const int w = tight.w, h = tight.h;
    Floats t(w, h);
    for (size_t i = 0; i < t.v.size(); ++i) t.v[i] = tight.v[i];
    const Floats sm = blur3(t, 3);
    const Floats nz = softNoise(w, h, rng, {{3, 1.0}, {2, .4}});
    Floats height(w, h);
    for (size_t i = 0; i < height.v.size(); ++i) height.v[i] = 2.2 * sm.v[i] + .5 * nz.v[i];
    const Floats gap = softNoise(w, h, rng, {{3, 1.0}});
    const double gapLimit = quantileOf(gap.v, .12);
    std::vector<double> inside;
    for (size_t i = 0; i < height.v.size(); ++i) if (tight.v[i]) inside.push_back(height.v[i]);
    const int steps = 1 + static_cast<int>(3 * density * (1 - calm) + .5);
    for (int s = 0; s < steps; ++s) {
        const double q = steps == 1 ? .15 : .15 + (.80 - .15) * s / (steps - 1);
        const double limit = quantileOf(inside, q);
        Mask m(w, h);
        for (size_t i = 0; i < m.v.size(); ++i) m.v[i] = height.v[i] > limit;
        m = bigAreas(stairs(majority3(m, 2 + static_cast<int>(4 * calm)), 2 + (calm > .4 ? 1 : 0)),
                     12 + static_cast<int>(30 * calm));
        Mask where(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                bool free = wall.at(x, y) == 9;
                for (int a = -1; a <= 1 && free; ++a)
                    for (int b = -1; b <= 1 && free; ++b)
                        if (wall.inside(x + b, y + a) && wall.at(x + b, y + a) < 9) free = false;
                where.at(x, y) = free && tight.at(x, y) && gap.at(x, y) > gapLimit;
            }
        rim(wall, m, where);
    }
}

/// _kuppen: small hills closed all around, common in the originals.
void hills(Grid<int32_t>& wall, const Grid<int32_t>& raster, int count, Rng& rng) {
    const int w = wall.w, h = wall.h;
    if (w < 13 || h < 13) return;
    for (int it = 0; it < 20 * count && count > 0; ++it) {
        const int y = static_cast<int>(rng.integers(6, h - 6)), x = static_cast<int>(rng.integers(6, w - 6));
        const double a = rng.uniform(1.6, 3.6), b = rng.uniform(1.6, 3.6);
        Mask m(w, h);
        for (int yy = 0; yy < h; ++yy)
            for (int xx = 0; xx < w; ++xx) {
                const double u = (yy - y) / a, v = (xx - x) / b;
                m.at(xx, yy) = u * u + v * v <= 1;
            }
        if (rng.uniform() < .5) {   // L shape
            const double c = rng.uniform(1.4, 2.4), d = rng.uniform(1.4, 2.4);
            const int dy = rng.integers(0, 2) ? 1 : -1, dx = rng.integers(0, 2) ? 1 : -1;
            for (int yy = 0; yy < h; ++yy)
                for (int xx = 0; xx < w; ++xx)
                    if (std::abs(yy - y - dy * a * .8) <= c && std::abs(xx - x - dx * b * .8) <= d) m.at(xx, yy) = 1;
        }
        const Mask near = dilate4(m, 2);
        bool taken = false;
        for (size_t i = 0; i < near.v.size() && !taken; ++i) taken = near.v[i] && wall.v[i] < 9;
        if (taken || (raster.at(x, y) == 2 && rng.uniform() < .6)) continue;
        const Mask s = stairs(m, 2);
        rim(wall, s, s);
        --count;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Noise and masks

Floats blur3(const Floats& in, int r) {
    Floats f = in;
    const int w = f.w, h = f.h, n = 2 * r + 1;
    std::vector<double> line;
    for (int pass = 0; pass < 3; ++pass) {
        // along y (axis 0), then along x (axis 1)
        Floats g(w, h);
        for (int x = 0; x < w; ++x)
            for (int y = 0; y < h; ++y) {
                double s = 0;
                for (int k = -r; k <= r; ++k) s += f.at(x, std::clamp(y + k, 0, h - 1));
                g.at(x, y) = s / n;
            }
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                double s = 0;
                for (int k = -r; k <= r; ++k) s += g.at(std::clamp(x + k, 0, w - 1), y);
                f.at(x, y) = s / n;
            }
    }
    return f;
}

namespace {
double spread(const Floats& f) {
    double mean = 0;
    for (double v : f.v) mean += v;
    mean /= double(f.v.size());
    double s = 0;
    for (double v : f.v) s += (v - mean) * (v - mean);
    return std::sqrt(s / double(f.v.size()));
}
}  // namespace

Floats softNoise(int w, int h, Rng& rng, std::initializer_list<std::pair<int, double>> radii) {
    Floats sum(w, h, 0.0);
    for (const auto& [r, weight] : radii) {
        Floats n(w, h);
        for (double& v : n.v) v = rng.normal();
        const Floats f = blur3(n, r);
        const double sd = spread(f) + 1e-9;
        for (size_t i = 0; i < sum.v.size(); ++i) sum.v[i] += weight * f.v[i] / sd;
    }
    const double sd = spread(sum) + 1e-9;
    for (double& v : sum.v) v /= sd;
    return sum;
}

double quantileOf(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const double pos = q * double(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos)), hi = std::min(v.size() - 1, lo + 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
}

Mask bigAreas(const Mask& m, int minCells) {
    Grid<int32_t> labels;
    const int n = label8(m, labels);
    std::vector<int> size(size_t(n) + 1, 0);
    for (int32_t l : labels.v) size[size_t(l)] += 1;
    Mask o(m.w, m.h);
    for (size_t i = 0; i < o.v.size(); ++i) o.v[i] = labels.v[i] > 0 && size[size_t(labels.v[i])] >= minCells;
    return o;
}

Grid<int32_t> dropShortWalls(const Grid<int32_t>& wall, int shortest, int empty) {
    const Mask k = maskOf(wall, 1, 9);
    const Mask keep = bigAreas(k, shortest);
    Grid<int32_t> o = wall;
    for (size_t i = 0; i < o.v.size(); ++i)
        if (k.v[i] && !keep.v[i]) o.v[i] = empty;
    return o;
}

// ---------------------------------------------------------------------------
// Large shape

Grid<int32_t> largeShape(const Grid<int32_t>& raster, double strength, Rng& rng, double density, double calm,
                         const Floats* rough) {
    const int w = raster.w, h = raster.h;
    Grid<int32_t> wall(w, h, 9);
    if (strength <= 0) return wall;
    const Floats height = softNoise(w, h, rng, {{static_cast<int>(11 * (1 + .5 * calm)), 1.0},
                                                {static_cast<int>(5 * (1 + .5 * calm)), .45}});
    const Mask taboo = dilate4(maskOf(raster, 1, 2), 2);
    const Floats gap = softNoise(w, h, rng, {{5, 1.0}, {2, .5}});
    const double open = quantileOf(gap.v, .25 * (1 - .5 * strength));
    const double share = .08 + .135 * strength;
    // Gradient: uplands and basins towards the rugged side.
    Floats up(w, h), down(w, h);
    for (size_t i = 0; i < up.v.size(); ++i) {
        const double tilt = rough ? 1.6 * (rough->v[i] - .5) : 0.0;
        up.v[i] = height.v[i] + tilt;
        down.v[i] = height.v[i] - tilt;
    }
    for (int nr = 0; nr < 2; ++nr) {
        Mask m(w, h);
        const double limit = nr == 0 ? quantileOf(up.v, 1 - share) : quantileOf(down.v, .8 * share);
        for (size_t i = 0; i < m.v.size(); ++i) m.v[i] = nr == 0 ? up.v[i] > limit : down.v[i] < limit;
        m = bigAreas(majority3(m, 2 + static_cast<int>(5 * calm)), static_cast<int>(60 * (1 + 1.5 * calm)));
        // Small uplands stay closed, large ones get ramps.
        const Mask large = dilate4(bigAreas(m, 180), 2);
        m = stairs(m, 3);
        const Mask high = nr == 0 ? m : invert(m);
        Mask where(w, h);
        for (size_t i = 0; i < where.v.size(); ++i) where.v[i] = !taboo.v[i] && (gap.v[i] > open || !large.v[i]);
        rim(wall, high, where);
    }
    wall = dropShortWalls(wall, static_cast<int>(9 * (1 + calm)), 9);
    mountainTerraces(wall, maskOf(raster, 1, 2), density, rng, calm);
    hills(wall, raster, roundEven(3 * (1 - .6 * calm) * strength * h * w / 4096.0), rng);
    return wall;
}

// ---------------------------------------------------------------------------
// Grounds

namespace {
/// _ausbreiten: values spread into their surroundings, but not across walls.
void spreadPlanes(std::vector<Floats>& planes, const Mask& barrier, int steps) {
    const int w = barrier.w, h = barrier.h;
    Floats through(w, h);
    for (size_t i = 0; i < through.v.size(); ++i) through.v[i] = barrier.v[i] ? .02 : 1.0;
    Floats right(w, h, 0.0), down(w, h, 0.0), weight(w, h, 1.0);   // right(x, y): edge (x, y)-(x+1, y)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (x + 1 < w) {
                right.at(x, y) = std::min(through.at(x, y), through.at(x + 1, y));
                weight.at(x, y) += right.at(x, y);
                weight.at(x + 1, y) += right.at(x, y);
            }
            if (y + 1 < h) {
                down.at(x, y) = std::min(through.at(x, y), through.at(x, y + 1));
                weight.at(x, y) += down.at(x, y);
                weight.at(x, y + 1) += down.at(x, y);
            }
        }
    for (Floats& z : planes)
        for (int s = 0; s < steps; ++s) {
            Floats n = z;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    double v = n.at(x, y);
                    if (x > 0) v += right.at(x - 1, y) * z.at(x - 1, y);
                    if (x + 1 < w) v += right.at(x, y) * z.at(x + 1, y);
                    if (y > 0) v += down.at(x, y - 1) * z.at(x, y - 1);
                    if (y + 1 < h) v += down.at(x, y) * z.at(x, y + 1);
                    n.at(x, y) = v / weight.at(x, y);
                }
            z = std::move(n);
        }
}
}  // namespace

Grid<int32_t> calmBiomes(const Grid<int32_t>& wall, const Grid<int32_t>& ground, double calm, Rng& rng,
                         const std::vector<Floats>* pref) {
    const int w = ground.w, h = ground.h;
    const size_t n = ground.v.size();
    std::vector<Floats> z(4, Floats(w, h, 0.0));
    double want[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < n; ++i) {
        const int b = ground.v[i] > 3 ? 0 : ground.v[i];
        z[size_t(b)].v[i] = 1;
        want[b] += 1.0 / double(n);
    }
    spreadPlanes(z, maskOf(wall, 1, 1 << 30), roundEven(2.5 * (1 + 7 * calm) * (1 + 7 * calm)));
    if (pref) {
        // Gradient: every place prefers its ground.
        for (int k = 0; k < 4; ++k) {
            double mean = 0;
            for (double v : (*pref)[size_t(k)].v) mean += v;
            mean /= double(n);
            for (size_t i = 0; i < n; ++i) {
                const double f = (*pref)[size_t(k)].v[i] / (mean + 1e-6) + .05;
                z[size_t(k)].v[i] *= f * std::sqrt(f);
            }
        }
    }
    double g[4] = {0, 0, 0, 0};
    auto choose = [&](size_t i) {
        int best = 0;
        double bv = -1;
        for (int k = 0; k < 4; ++k) {
            const double v = z[size_t(k)].v[i] * nn::detExp(g[k]);
            if (v > bv) { bv = v; best = k; }
        }
        return best;
    };
    for (int it = 0; it < 40; ++it) {
        double is[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < n; ++i) is[choose(i)] += 1.0 / double(n);
        for (int k = 0; k < 4; ++k)
            g[k] = std::clamp(g[k] + .35 * (nn::detLog(want[k] + .01) - nn::detLog(is[k] + .01)), -1.5, 1.5);
    }
    Grid<int32_t> calmGrid(w, h);
    for (size_t i = 0; i < n; ++i) calmGrid.v[i] = choose(i);
    // Originals have tongues and bays: bend with soft noise, not near walls.
    const Mask nearWall = dilate4(maskOf(wall, 1, 1 << 30), 2);
    Floats open(w, h);
    for (size_t i = 0; i < n; ++i) open.v[i] = nearWall.v[i] ? 0.0 : 1.0;
    const Floats reach = blur3(open, 2);
    const Floats a1 = softNoise(w, h, rng, {{7, 1.0}}), a2 = softNoise(w, h, rng, {{3, 1.0}});
    const Floats b1 = softNoise(w, h, rng, {{7, 1.0}}), b2 = softNoise(w, h, rng, {{3, 1.0}});
    Grid<int32_t> out(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = size_t(y) * w + x;
            const double dy = reach.v[i] * (8.0 * a1.v[i] + 4.5 * a2.v[i]);
            const double dx = reach.v[i] * (8.0 * b1.v[i] + 4.5 * b2.v[i]);
            out.at(x, y) = calmGrid.at(std::clamp(roundEven(x + dx), 0, w - 1), std::clamp(roundEven(y + dy), 0, h - 1));
        }
    return out;
}

Grid<int32_t> dropSmallPatches(const Grid<int32_t>& ground, const Grid<int32_t>& wall, double size) {
    const int w = ground.w, h = ground.h;
    Grid<int32_t> b = ground;
    for (int pass = 0; pass < 8; ++pass) {
        struct Patch { int size; std::vector<int> cells; };
        std::vector<Patch> patches;
        for (int k = 0; k < 4; ++k) {
            Grid<int32_t> labels;
            const int n = label8(maskOf(b, k, k + 1), labels);
            std::vector<std::vector<int>> cells(size_t(n) + 1);
            for (size_t i = 0; i < labels.v.size(); ++i)
                if (labels.v[i]) cells[size_t(labels.v[i])].push_back(static_cast<int>(i));
            for (int i = 1; i <= n; ++i)
                if (double(cells[size_t(i)].size()) < size)
                    patches.push_back({static_cast<int>(cells[size_t(i)].size()), std::move(cells[size_t(i)])});
        }
        std::stable_sort(patches.begin(), patches.end(), [](const Patch& u, const Patch& v) { return u.size < v.size; });
        Mask changed(w, h, 0);
        bool any = false;
        for (const Patch& pt : patches) {
            Mask m(w, h, 0);
            for (int c : pt.cells) m.v[size_t(c)] = 1;
            const Mask grown = dilate4(m, 1);
            int count[5] = {0, 0, 0, 0, 0};
            bool ring = false, clash = false, walls = false;
            for (size_t i = 0; i < m.v.size(); ++i) {
                if (grown.v[i] && !m.v[i]) {
                    ring = true;
                    count[std::clamp(b.v[i], 0, 4)] += 1;
                }
                if (grown.v[i] && changed.v[i]) clash = true;
                if (m.v[i] && wall.v[i] > 0) walls = true;
            }
            if (!ring || clash) continue;
            if (walls) count[1] = count[3] = count[4] = 0;   // only dirt (0) or sand (2)
            int best = -1;
            for (int k = 0; k < 4; ++k)
                if (count[k] > 0 && (best < 0 || count[k] > count[best])) best = k;
            if (best < 0) continue;
            for (int c : pt.cells) {
                b.v[size_t(c)] = best;
                changed.v[size_t(c)] = 1;
            }
            any = true;
        }
        if (!any) break;
    }
    return b;
}

Grid<int32_t> narrowRim(const Grid<int32_t>& wall, const Grid<int32_t>& ground) {
    const int w = ground.w, h = ground.h;
    const Mask k = maskOf(wall, 1, 1 << 30);
    Mask wide(w, h, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (!k.at(x, y)) continue;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if (wide.inside(x + dx, y + dy)) wide.at(x + dx, y + dy) = 1;
        }
    Grid<int32_t> labels;
    const int n = label8(wide, labels);
    std::vector<std::array<int, 5>> count(size_t(n) + 1, std::array<int, 5>{0, 0, 0, 0, 0});
    for (size_t i = 0; i < labels.v.size(); ++i)
        if (labels.v[i]) count[size_t(labels.v[i])][size_t(std::clamp(ground.v[i], 0, 4))] += 1;
    Grid<int32_t> b = ground;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int l = labels.at(x, y);
            if (!l || !k.at(x, y)) continue;
            const auto& c = count[size_t(l)];
            const int g = c[2] > c[0] ? 2 : 0;
            const int bx = x / 2 * 2, by = y / 2 * 2;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    if (b.inside(bx + dx, by + dy)) b.at(bx + dx, by + dy) = g;
        }
    return b;
}

// ---------------------------------------------------------------------------
// Rock and lava

const Pattern kPatterns[5] = {{439, 454, 4, false, 0}, {455, 470, 4, false, .11}, {471, 486, 4, true, 0},
                              {487, 495, 3, true, .29}, {496, 504, 3, true, .24}};

int8_t tileKind(int t) {
    if (t >= 439 && t <= 470) return kRock;
    if (t >= 471 && t <= 504) return kLava;
    if (t >= 559 && t <= 606) return kSeam;
    return kOther;
}

Mask lavaPlan(const Grid<int32_t>& ground, const Grid<int32_t>& wall, double amount, Rng& rng) {
    const int w = ground.w, h = ground.h;
    Mask lava(w, h, 0);
    if (amount <= 0) return lava;
    const Mask nearWall = dilate4(maskOf(wall, 1, 1 << 30), 2);
    Mask rock(w, h);
    for (size_t i = 0; i < rock.v.size(); ++i) rock.v[i] = ground.v[i] == 1 && !nearWall.v[i];
    const Floats deep = depth(rock, 9), belt = depth(rock, 2);
    const Floats nz = softNoise(w, h, rng, {{5, 1.0}, {2, .6}});
    Floats note(w, h);
    for (size_t i = 0; i < note.v.size(); ++i) note.v[i] = std::min(deep.v[i], 9.0) / 9 + .5 * nz.v[i];
    Grid<int32_t> labels;
    const int n = label8(maskOf(ground, 1, 2), labels);
    std::vector<std::vector<int>> areas(size_t(n) + 1);
    for (size_t i = 0; i < labels.v.size(); ++i)
        if (labels.v[i]) areas[size_t(labels.v[i])].push_back(static_cast<int>(i));
    // Originals: how often a rock area has a lava core (by size in tiles) and its share.
    const int limits[4] = {300, 1000, 3000, 1 << 30};
    const double chances[4] = {0.0, .45, .62, .78};
    const double xs[5] = {0, .1, .5, .9, 1}, ys[5] = {.04, .08, .31, .60, .7};
    for (int i = 1; i <= n; ++i) {
        const auto& area = areas[size_t(i)];
        const int tiles = 4 * static_cast<int>(area.size());
        double chance = chances[3];
        for (int k = 0; k < 4; ++k)
            if (tiles < limits[k]) { chance = chances[k]; break; }
        const double u0 = rng.uniform(), u1 = rng.uniform();
        if (u0 >= std::min(1.0, 2 * amount * chance)) continue;
        double share = ys[4];
        for (int k = 0; k < 4; ++k)
            if (u1 <= xs[k + 1]) { share = ys[k] + (ys[k + 1] - ys[k]) * (u1 - xs[k]) / (xs[k + 1] - xs[k]); break; }
        share = std::min(.85, share * std::min(1.6, .5 + amount));
        std::vector<int> possible;
        for (int c : area) if (belt.v[size_t(c)] >= 2) possible.push_back(c);
        if (possible.size() < 30) continue;
        const size_t want = std::min(static_cast<size_t>(share * double(area.size())),
                                     static_cast<size_t>(.9 * double(possible.size())));
        std::stable_sort(possible.begin(), possible.end(),
                         [&](int a, int b) { return note.v[size_t(a)] > note.v[size_t(b)]; });
        for (size_t k = 0; k < want; ++k) lava.v[size_t(possible[k])] = 1;
    }
    // Fill holes, drop crumbs, smooth rim.
    Mask f = lava;
    for (int it = 0; it < 3; ++it) f = majority3(f, 1);
    Grid<int32_t> fi(w, h);
    for (size_t i = 0; i < fi.v.size(); ++i) fi.v[i] = f.v[i];
    fi = dropSmallPatches(fi, Grid<int32_t>(w, h, 0), 14);
    for (size_t i = 0; i < lava.v.size(); ++i) lava.v[i] = fi.v[i] == 1 && rock.v[i];
    return lava;
}

Grid<int8_t> cellPlan(const Mask& lava, const Grid<int32_t>& ground) {
    const int w = lava.w * 2, h = lava.h * 2;
    Mask kl(w, h), kb(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            kl.at(x, y) = lava.at(x / 2, y / 2);
            kb.at(x, y) = ground.at(x / 2, y / 2) == 1;
        }
    const Mask a = dilate4(kl, 1), b = dilate4(invert(kl), 1);
    Grid<int8_t> plan(w, h);
    for (size_t i = 0; i < plan.v.size(); ++i) {
        plan.v[i] = kb.v[i] ? (kl.v[i] ? kLava : kRock) : kOther;
        if (a.v[i] && b.v[i] && (kb.v[i] || kl.v[i])) plan.v[i] = kSeam;
    }
    return plan;
}

std::vector<Mask> patternAreas(int w, int h, Rng& rng) {
    const Floats r = softNoise(w, h, rng, {{9, 1.0}, {4, .35}});
    const Floats q = softNoise(w, h, rng, {{6, 1.0}, {3, .35}});
    const double t2 = quantileOf(r.v, 1 - kPatterns[1].share);
    const double t4 = quantileOf(q.v, kPatterns[3].share), t5 = quantileOf(q.v, 1 - kPatterns[4].share);
    std::vector<Mask> out(5, Mask(w, h));
    for (size_t i = 0; i < r.v.size(); ++i) {
        const bool f2 = r.v[i] > t2, l4 = q.v[i] < t4, l5 = q.v[i] > t5;
        out[0].v[i] = !f2;
        out[1].v[i] = f2;
        out[2].v[i] = !l4 && !l5;
        out[3].v[i] = l4;
        out[4].v[i] = l5;
    }
    return out;
}

std::vector<VolcanoSite> volcanoSites(Grid<int32_t>& ground, const Grid<int32_t>& wall, const int counts[3], int x0,
                                      int y0, int x1, int y1, Rng& rng, const Mask* allowed) {
    const int w = ground.w, h = ground.h;
    std::vector<VolcanoSite> sites;
    const Floats deep = depth(maskOf(ground, 1, 2), 8);
    const Mask walls = maskOf(wall, 1, 1 << 30);
    // Large first, then medium, then small: (size, lava, radius).
    const int sizes[3] = {2, 1, 0}, radius[3] = {6, 4, 3};
    for (int s = 0; s < 3; ++s) {
        const int size = sizes[s], r = radius[s];
        const bool lava = size > 0;
        const Mask near = dilate4(walls, r + 3);
        for (int k = 0; k < counts[size]; ++k) {
            Floats note(w, h, -1.0);
            bool any = false;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    bool free = !near.at(x, y) && (!allowed || allowed->at(x, y));
                    free = free && y >= y0 + r + 2 && y < y1 - r - 2 && x >= x0 + r + 2 && x < x1 - r - 2;
                    for (const VolcanoSite& o : sites)
                        free = free && (y - o.y) * (y - o.y) + (x - o.x) * (x - o.x) > (r + o.r + 10) * (r + o.r + 10);
                    if (free) {
                        note.at(x, y) = deep.at(x, y) + rng.uniform() * 3;
                        any = true;
                    }
                }
            if (!any) break;
            size_t best = 0;
            for (size_t i = 1; i < note.v.size(); ++i)
                if (note.v[i] > note.v[best]) best = i;
            const int y = static_cast<int>(best) / w, x = static_cast<int>(best) % w;
            for (int yy = 0; yy < h; ++yy)
                for (int xx = 0; xx < w; ++xx)
                    if ((yy - y) * (yy - y) + (xx - x) * (xx - x) <= (r + 2) * (r + 2)) ground.at(xx, yy) = 1;
            sites.push_back({y, x, size, lava, r});
        }
    }
    return sites;
}

}  // namespace extra
}  // namespace om::gen
