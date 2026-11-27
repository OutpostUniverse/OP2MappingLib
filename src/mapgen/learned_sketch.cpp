// Learned map generator, part 1: the network file and the sketch (Python:
// mapgen_netz.skizze and mapgen_netz_waende.py).
// Ported from fefi-byte's op2-codegen-editor (MIT, see THIRD-PARTY-NOTICES.md).
#include "learned.hpp"
#include "learned_extra.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace om::gen {

// ---------------------------------------------------------------------------
// Network file

std::unique_ptr<LearnedModel> LearnedModel::load(const uint8_t* data, size_t len) {
    auto m = std::make_unique<LearnedModel>();
    m->weights = nn::Weights::parse(data, len);
    m->sketch.load(m->weights);
    m->tiles.load(m->weights);
    const int V = m->tiles.vocab;
    m->vocab = V;
    const auto& vok = m->weights.u16("vocab");
    const auto& vct = m->weights.u8("vocab_cell_types");
    if (static_cast<int>(vok.size()) != V || static_cast<int>(vct.size()) != V)
        throw std::runtime_error("network file: vocabulary size does not match the tile network");
    m->vocabMapping.assign(vok.begin(), vok.end());
    m->vocabCell = vct;
    m->vocabWall.resize(size_t(V));
    m->vocabFree.resize(size_t(V));
    for (int i = 0; i < V; ++i) {
        const int ct = vct[size_t(i)];
        m->vocabWall[size_t(i)] = ct == 8 || ct == 9 || ct == 10;
        Grid<uint8_t> one(1, 1, static_cast<uint8_t>(ct));
        m->vocabFree[size_t(i)] = passableMask(one).v[0];
        if (m->vocabWall[size_t(i)]) m->wallIdx.push_back(i);
    }
    // numpy.packbits rows (big-endian bit order) -> 64-bit words, plus the transposes.
    m->words = (V + 63) / 64;
    const size_t rowBytes = size_t(V + 7) / 8;
    auto unpack = [&](const std::vector<uint8_t>& packed, std::vector<uint64_t>& rows, std::vector<uint64_t>& cols) {
        if (packed.size() != rowBytes * size_t(V)) throw std::runtime_error("network file: pair table has the wrong size");
        rows.assign(size_t(V) * m->words, 0);
        cols.assign(size_t(V) * m->words, 0);
        for (int a = 0; a < V; ++a)
            for (int b = 0; b < V; ++b)
                if ((packed[size_t(a) * rowBytes + size_t(b) / 8] >> (7 - b % 8)) & 1) {
                    rows[size_t(a) * m->words + size_t(b) / 64] |= uint64_t(1) << (b % 64);
                    cols[size_t(b) * m->words + size_t(a) / 64] |= uint64_t(1) << (a % 64);
                }
    };
    unpack(m->weights.u8("pairs_right"), m->right, m->left);
    unpack(m->weights.u8("pairs_down"), m->down, m->up);
    m->mapCell = m->weights.u8("mapping_cell_types");
    const auto& mg = m->weights.u8("mapping_grounds");
    m->mapGround.resize(mg.size());
    for (size_t i = 0; i < mg.size(); ++i) m->mapGround[i] = mg[i] == 255 ? int8_t(-1) : static_cast<int8_t>(mg[i]);
    m->mapFlags = m->weights.u8("mapping_flags");
    if (m->mapGround.size() != m->mapCell.size() || m->mapFlags.size() != m->mapCell.size())
        throw std::runtime_error("network file: mapping tables differ in size");
    for (int32_t t : m->vocabMapping)
        if (t < 0 || t >= static_cast<int32_t>(m->mapCell.size())) throw std::runtime_error("network file: vocabulary outside the mapping table");
    return m;
}

// ---------------------------------------------------------------------------
// Helpers

int label8(const Mask& m, Grid<int32_t>& labels) {
    labels = Grid<int32_t>(m.w, m.h, 0);
    int n = 0;
    std::vector<int> stack;
    for (int sy = 0; sy < m.h; ++sy)
        for (int sx = 0; sx < m.w; ++sx) {
            if (!m.at(sx, sy) || labels.at(sx, sy)) continue;
            ++n;
            labels.at(sx, sy) = n;
            stack.assign(1, sy * m.w + sx);
            while (!stack.empty()) {
                const int c = stack.back();
                stack.pop_back();
                const int x = c % m.w, y = c / m.w;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx, ny = y + dy;
                        if ((dx || dy) && m.inside(nx, ny) && m.at(nx, ny) && !labels.at(nx, ny)) {
                            labels.at(nx, ny) = n;
                            stack.push_back(ny * m.w + nx);
                        }
                    }
            }
        }
    return n;
}

void softmax(const float* logits, int n, double temperature, double* out) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) mx = std::max(mx, double(logits[i]) / temperature);
    double s = 0;
    for (int i = 0; i < n; ++i) s += out[i] = nn::detExp(double(logits[i]) / temperature - mx);
    for (int i = 0; i < n; ++i) out[i] /= s;
}

namespace {

constexpr int kMud = 0, kSand = 2;
const double kGroundShare[5] = {.59, .19, .20, .02, .004};   // BODEN_ANTEIL
constexpr double kMountainOnSand = .06;                       // BERG_AUF_SAND

/// One random stream per step (Python: default_rng([seed, k])).
Rng stream(uint64_t seed, uint64_t k) { return Rng(seed * 0x9E3779B97F4A7C15ull ^ (0x6e65747a00000000ull + k)); }

inline int roundEven(double v) { return static_cast<int>(std::nearbyint(v)); }

/// Directions a wall falls towards (dx, dy): E, SE, S, SW, W, NW, N, NE, normalised.
struct Dirs {
    double x[8], y[8];
    Dirs() {
        const int d[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
        for (int i = 0; i < 8; ++i) {
            const double l = std::sqrt(double(d[i][0] * d[i][0] + d[i][1] * d[i][1]));
            x[i] = d[i][0] / l;
            y[i] = d[i][1] / l;
        }
    }
};
const Dirs kDir;
inline bool isNorth(int v) { return v == 6 || v == 7 || v == 8; }   // NW, N, NE: one cell thick

struct Vec { double x = 0, y = 0; };
inline double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y; }
/// tangente: walking direction of a wall, high side left.
inline Vec tangent(int r) { return {kDir.y[r], -kDir.x[r]}; }
/// fallrichtung
inline int fallDirection(Vec t) {
    int best = 0;
    double bv = -1e300;
    for (int r = 0; r < 8; ++r) {
        const double v = kDir.x[r] * -t.y + kDir.y[r] * t.x;
        if (v > bv) { bv = v; best = r; }
    }
    return best;
}

const int kMirrorX[9] = {0, 5, 4, 3, 2, 1, 8, 7, 6};
const int kMirrorY[9] = {0, 1, 8, 7, 6, 5, 4, 3, 2};
const int kTurn[9] = {0, 5, 6, 7, 8, 1, 2, 3, 4};

/// _spiegel: copies the base half; with `walls` the fall directions turn along.
template <class T>
void mirrorSketch(Grid<T>& a, Sym s, bool walls) {
    const int w = a.w, h = a.h;
    auto tab = [&](T v, const int* t) { return walls ? static_cast<T>(t[int(v)]) : v; };
    if (s == Sym::MirrorX || s == Sym::Quad)
        for (int y = 0; y < h; ++y)
            for (int x = w / 2; x < w; ++x) a.at(x, y) = tab(a.at(w - 1 - x, y), kMirrorX);
    if (s == Sym::MirrorY || s == Sym::Quad)
        for (int y = h / 2; y < h; ++y)
            for (int x = 0; x < w; ++x) a.at(x, y) = tab(a.at(x, h - 1 - y), kMirrorY);
    if (s == Sym::Rot180)
        for (int y = h / 2; y < h; ++y)
            for (int x = 0; x < w; ++x) a.at(x, y) = tab(a.at(w - 1 - x, h - 1 - y), kTurn);
}

double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double pos = q * double(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos)), hi = std::min(v.size() - 1, lo + 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
}

template <class T>
Grid<T> upscale2(const Grid<T>& g) {
    Grid<T> o(g.w * 2, g.h * 2);
    for (int y = 0; y < o.h; ++y)
        for (int x = 0; x < o.w; ++x) o.at(x, y) = g.at(x / 2, y / 2);
    return o;
}

template <class T>
void shuffle(std::vector<T>& v, Rng& rng) {
    for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[size_t(rng.integers(0, int64_t(i)))]);
}

// ---------------------------------------------------------------------------
// Zones

/// _zonen: zones on the larger world (1 mountain, 2 plateau, 3 valley, 4 gorge), from the mountain plan.
Grid<int8_t> worldZones(const Params& p, const MountainPlan& plan) {
    const int R = kWorldMargin, W = p.width + 2 * R, H = p.height + 2 * R;
    Grid<int8_t> z(W, H, 0);
    auto set = [&](const Mask& m, int v) {
        for (size_t i = 0; i < m.v.size(); ++i) if (m.v[i]) z.v[i] = static_cast<int8_t>(v);
    };
    for (const Cut& e : plan.valleys) set(e.sole, 3);
    for (const Cut& e : plan.gorges) set(e.sole, 4);
    for (const Mask& m : plan.plateaus) set(m, 2);
    if (!plan.terraces.empty()) set(plan.terraces[0], 1);
    // Features are not used: the editor's learned template hides them (op2-codegen-editor 0.12).
    if (p.symmetry != Sym::None) mirrorSketch(z, p.symmetry, false);
    return z;
}

/// _zonenraster: zones -> guide per sketch cell: 1 tight, 2 building ground, 0 any.
Grid<int32_t> zoneRaster(const Grid<int8_t>& z, const Params& p, Rng& rng) {
    const int H4 = z.h / 4, W4 = z.w / 4;
    Grid<int32_t> o(W4, H4, 0);
    for (int y = 0; y < H4; ++y)
        for (int x = 0; x < W4; ++x) {
            int tight = 0, build = 0;
            for (int dy = 0; dy < 4; ++dy)
                for (int dx = 0; dx < 4; ++dx) {
                    const int v = z.at(4 * x + dx, 4 * y + dy);
                    tight += v == 1 || v == 4;
                    build += v == 2 || v == 3;
                }
            o.at(x, y) = tight >= 8 ? 1 : build >= 8 ? 2 : 0;
        }
    const int fh = z.h / 8 + 1, fw = z.w / 8 + 1;
    Floats f(fw, fh);
    for (double& v : f.v) v = rng.normal();
    for (int it = 0; it < 6; ++it) {
        Floats n(fw, fh);
        for (int y = 0; y < fh; ++y)
            for (int x = 0; x < fw; ++x) {
                auto at = [&](int xx, int yy) { return f.at(std::clamp(xx, 0, fw - 1), std::clamp(yy, 0, fh - 1)); };
                n.at(x, y) = (at(x, y - 1) + at(x, y + 1) + at(x - 1, y) + at(x + 1, y) + at(x, y)) / 5;
            }
        f = std::move(n);
    }
    Mask tight(W4, H4);
    for (size_t i = 0; i < tight.v.size(); ++i) tight.v[i] = o.v[i] == 1;
    const Mask nearTight = dilate4(tight, 2);
    const double share = std::clamp(1 - p.cliffs / .40, 0.0, 1.0);
    std::vector<double> vals;
    for (int y = 0; y < H4; ++y)
        for (int x = 0; x < W4; ++x)
            if (o.at(x, y) == 0 && !nearTight.at(x, y)) vals.push_back(f.at(x / 2, y / 2));
    if (!vals.empty() && share > 0) {
        const double q = quantile(vals, share);
        for (int y = 0; y < H4; ++y)
            for (int x = 0; x < W4; ++x)
                if (o.at(x, y) == 0 && !nearTight.at(x, y) && f.at(x / 2, y / 2) <= q) o.at(x, y) = 2;
    }
    return upscale2(o);
}

/// _bergboden: one ground per mountain plus rim: mud, rarely sand. 5 = not given.
Grid<int32_t> mountainGround(const Grid<int32_t>& zone, Rng& rng) {
    const int h = zone.h / 2, w = zone.w / 2;
    Mask tight(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) tight.at(x, y) = zone.at(2 * x, 2 * y) == 1;
    Grid<int32_t> labels;
    const int n = label8(tight, labels);
    Grid<int32_t> o(w, h, 5);
    for (int i = 1; i <= n; ++i) {
        const int ground = rng.uniform() < kMountainOnSand ? kSand : kMud;
        Mask m(w, h);
        for (size_t k = 0; k < m.v.size(); ++k) m.v[k] = labels.v[k] == i;
        const Mask rim = dilate4(m, 2);
        for (size_t k = 0; k < m.v.size(); ++k) {
            if (rim.v[k] && o.v[k] == 5) o.v[k] = ground;
            if (m.v[k]) o.v[k] = ground;
        }
    }
    return upscale2(o);
}

// ---------------------------------------------------------------------------
// Drawing the sketch

/// _skizze_ziehen: draws a sketch from the network in random order.
std::pair<Grid<int32_t>, Grid<int32_t>> drawFromNet(const LearnedModel& m, const Grid<int32_t>& zone,
                                                    const Grid<int32_t>& ground0, const double* target, Rng& rng,
                                                    int rounds, nn::Pool& pool, const ProgressFn& progress,
                                                    const Grid<int32_t>* wall0 = nullptr,
                                                    const std::vector<Floats>* bias0 = nullptr,
                                                    double temperature = .94, double zoneForce = .7) {
    const int h = zone.h, w = zone.w;
    const size_t P = size_t(h) * w;
    // wall0: fixed walls (1..8) and open cells (9) from the large shape;
    // bias0: per ground a bias per cell (the gradient).
    Grid<int32_t> wall = wall0 ? *wall0 : Grid<int32_t>(w, h, 9), ground = ground0, none(w, h, 0);
    std::vector<uint8_t> fixed(P);
    for (size_t i = 0; i < P; ++i) fixed[i] = wall.v[i] < 9;
    std::vector<double> pw(P * 9), pb(P * 5);
    nn::Act lw, lb, lw0, lb0;

    auto mixBias = [&](double* bias) {
        for (int k = 0; k < 5; ++k) bias[k] = 0;
        if (!target) return;
        double laid[5] = {0, 0, 0, 0, 0}, sum = 0;
        for (int32_t g : ground.v)
            if (g < 5) { laid[g] += 1; sum += 1; }
        for (int k = 0; k < 5; ++k) {
            const double is = sum > .04 * double(P) ? laid[k] / std::max(1.0, sum) : kGroundShare[k];
            bias[k] = std::clamp(1.2 * nn::detLog((target[k] + .004) / (is + .004)), -4.0, 4.0);
        }
    };
    auto probabilities = [&]() {
        m.sketch.run(h, w, wall.v.data(), ground.v.data(), zone.v.data(), lw, lb, pool);
        m.sketch.run(h, w, wall.v.data(), ground.v.data(), none.v.data(), lw0, lb0, pool);
        double bias[5];
        mixBias(bias);
        float a[9], b[5];
        for (size_t i = 0; i < P; ++i) {
            for (int k = 0; k < 9; ++k) {
                const float v = lw.ch(k)[i];
                a[k] = static_cast<float>(v + zoneForce * (double(v) - double(lw0.ch(k)[i])));
            }
            for (int k = 0; k < 5; ++k)
                b[k] = static_cast<float>(double(lb.ch(k)[i]) + bias[k] + (bias0 ? (*bias0)[size_t(k)].v[i] : 0.0));
            softmax(a, 9, temperature, &pw[i * 9]);
            softmax(b, 5, 1.0, &pb[i * 5]);
        }
    };
    auto pick = [&](const double* p, int n, double u) {
        double c = 0;
        for (int k = 0; k < n; ++k) {
            c += p[k];
            if (c > u) return k;
        }
        return 0;
    };
    auto place = [&](const std::vector<int>& cells, const std::vector<int>& blocks) {
        probabilities();
        std::vector<double> u(cells.size());
        for (double& x : u) x = rng.uniform();
        for (size_t i = 0; i < cells.size(); ++i) wall.v[size_t(cells[i])] = pick(&pw[size_t(cells[i]) * 9], 9, u[i]);
        std::vector<double> ub(blocks.size());
        for (double& x : ub) x = rng.uniform();
        for (size_t i = 0; i < blocks.size(); ++i) {
            const int by = blocks[i] / (w / 2), bx = blocks[i] % (w / 2);
            double mean[5] = {0, 0, 0, 0, 0};
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    for (int k = 0; k < 5; ++k) mean[k] += pb[(size_t(2 * by + dy) * w + 2 * bx + dx) * 5 + k];
            for (double& x : mean) x /= 4;
            const int c = pick(mean, 5, ub[i]);
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) ground.at(2 * bx + dx, 2 * by + dy) = c;
        }
    };

    std::vector<int> openW, openB;
    for (int i = 0; i < static_cast<int>(P); ++i)
        if (!fixed[size_t(i)]) openW.push_back(i);
    for (int y = 0; y < h / 2; ++y)
        for (int x = 0; x < w / 2; ++x)
            if (ground.at(2 * x, 2 * y) == 5) openB.push_back(y * (w / 2) + x);
    shuffle(openW, rng);
    shuffle(openB, rng);
    const double kPi = 3.14159265358979323846;
    size_t a = 0, b = 0;
    for (int r = 0; r < rounds; ++r) {
        if (progress && !progress(0.0)) throw Cancelled();
        const double plan = 1 - std::cos(kPi / 2 * double(r + 1) / rounds);
        const size_t a2 = static_cast<size_t>(roundEven(double(openW.size()) * plan));
        const size_t b2 = static_cast<size_t>(roundEven(double(openB.size()) * std::min(1.0, plan * 2)));
        place(std::vector<int>(openW.begin() + long(a), openW.begin() + long(a2)),
              std::vector<int>(openB.begin() + long(b), openB.begin() + long(b2)));
        a = a2;
        b = b2;
    }
    // One more pass over an eighth of the cells, now with all neighbours known.
    std::vector<int> again;
    for (int i = 0; i < static_cast<int>(P); ++i)
        if (rng.uniform() < .12 && !fixed[size_t(i)]) again.push_back(i);
    for (int i : again) wall.v[size_t(i)] = 9;
    place(again, {});
    return {wall, ground};
}

// ---------------------------------------------------------------------------
// Walls (mapgen_netz_waende.py)

struct ChainList {
    std::vector<std::vector<int>> chains;   ///< cell indices y * w + x
    std::vector<bool> closed;
};

/// ketten: joins wall cells into directed chains.
ChainList chainsOf(const Grid<int32_t>& wv) {
    const int h = wv.h, q = wv.w;
    struct Cand { double score; int from, to; };
    std::vector<Cand> cand;
    std::vector<int> all;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < q; ++x) {
            if (wv.at(x, y) <= 0) continue;
            all.push_back(y * q + x);
            const int r = wv.at(x, y) - 1;
            const Vec d{kDir.x[r], kDir.y[r]}, t = tangent(r);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int py = y + dy, px = x + dx;
                    if (!(dy || dx) || py < 0 || py >= h || px < 0 || px >= q || !wv.at(px, py)) continue;
                    const double len = std::sqrt(double(dx * dx + dy * dy));
                    const Vec s{dx / len, dy / len};
                    const int r2 = wv.at(px, py) - 1;
                    const double a = dot(s, t), b = dot(s, tangent(r2)), c = dot(d, Vec{kDir.x[r2], kDir.y[r2]});
                    if (a + b > .25 && a > -.2 && b > -.2 && c > -.4)
                        cand.push_back({a + b + .6 * c - .5 * len, y * q + x, py * q + px});
                }
        }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& u, const Cand& v) { return u.score > v.score; });
    std::vector<int> next(size_t(h) * q, -1), prev(size_t(h) * q, -1);
    for (const Cand& c : cand) {
        if (next[size_t(c.from)] >= 0 || prev[size_t(c.to)] >= 0) continue;
        int e = c.to, l = 0;
        while (next[size_t(e)] >= 0 && l < 5) {
            e = next[size_t(e)];
            ++l;
        }
        if (e == c.from) continue;
        next[size_t(c.from)] = c.to;
        prev[size_t(c.to)] = c.from;
    }
    ChainList out;
    std::vector<uint8_t> seen(size_t(h) * q, 0);
    std::vector<int> order;
    for (int p : all) if (prev[size_t(p)] < 0) order.push_back(p);
    order.insert(order.end(), all.begin(), all.end());
    for (int p : order) {
        if (seen[size_t(p)]) continue;
        std::vector<int> chain{p};
        seen[size_t(p)] = 1;
        while (next[size_t(chain.back())] >= 0 && !seen[size_t(next[size_t(chain.back())])]) {
            chain.push_back(next[size_t(chain.back())]);
            seen[size_t(chain.back())] = 1;
        }
        out.closed.push_back(next[size_t(chain.back())] == p && chain.size() > 3);
        out.chains.push_back(std::move(chain));
    }
    return out;
}

/// putzen: joins wall pieces, extends them in tight zones, drops snippets.
Grid<int32_t> tidyWalls(const Grid<int32_t>& wv, const Grid<int32_t>& zone, Rng& rng, int minLen = 7, int reach = 7) {
    const int h = wv.h, q = wv.w;
    const ChainList all = chainsOf(wv);
    std::vector<int> dir(size_t(h) * q, -1);
    for (const auto& c : all.chains)
        for (int p : c) dir[size_t(p)] = wv.v[size_t(p)] - 1;
    std::map<int, std::vector<int>> pieces;
    std::vector<std::vector<int>> rings;
    for (size_t i = 0; i < all.chains.size(); ++i) {
        if (all.closed[i]) rings.push_back(all.chains[i]);
        else pieces[static_cast<int>(i)] = all.chains[i];
    }
    auto Y = [&](int p) { return p / q; };
    auto X = [&](int p) { return p % q; };
    auto endDir = [&](const std::vector<int>& k, bool back) {
        const size_t n = std::min<size_t>(4, k.size());
        const int s0 = back ? k[k.size() - n] : k[0], s1 = back ? k.back() : k[n - 1];
        const Vec t = tangent(dir[size_t(back ? s1 : s0)]);
        if (n > 1) {
            const Vec v{double(X(s1) - X(s0)), double(Y(s1) - Y(s0))};
            const double l = std::sqrt(v.x * v.x + v.y * v.y);
            if (l > 0) return Vec{.6 * v.x / l + .4 * t.x, .6 * v.y / l + .4 * t.y};
        }
        return t;
    };
    struct Ends { int first, last; Vec tFirst, tLast; };
    std::map<int, Ends> ends;
    auto remember = [&](int i) {
        const auto& k = pieces[i];
        ends[i] = {k.front(), k.back(), endDir(k, false), endDir(k, true)};
    };
    using Item = std::tuple<double, int, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    auto near = [&](int i, int j) {
        const int a = ends[i].last, b = ends[j].first;
        return std::max(std::abs(Y(a) - Y(b)), std::abs(X(a) - X(b))) <= reach;
    };
    auto pair = [&](int i, int j) {
        const int a1 = ends[i].last, b0 = ends[j].first;
        const Vec ta = ends[i].tLast, tb = ends[j].tFirst;
        const Vec v{double(X(b0) - X(a1)), double(Y(b0) - Y(a1))};
        const double len = std::sqrt(v.x * v.x + v.y * v.y);
        if (len < 1 || len > reach) return;
        const Vec s{v.x / len, v.y / len};
        const int ra = dir[size_t(a1)], rb = dir[size_t(b0)];
        const double f0 = dot(s, ta), f1 = dot(s, tb), f2 = kDir.x[ra] * kDir.x[rb] + kDir.y[ra] * kDir.y[rb];
        const bool good = len <= 2.9 ? (f0 > -.1 && f1 > -.1 && f2 > -.3) : (f0 > .45 && f1 > .45 && f2 > .1);
        if (good) {
            const double note = (f0 + f1 + f2) - .35 * len +
                                .1 * std::min(4.0, double(pieces[i].size() + pieces[j].size()) / 8);
            heap.emplace(-note, i, j);
        }
    };
    for (auto& [i, k] : pieces) remember(i);
    for (auto& [i, ki] : pieces)
        for (auto& [j, kj] : pieces)
            if (i != j && near(i, j)) pair(i, j);
    int nr = static_cast<int>(all.chains.size());
    while (!heap.empty()) {
        const auto [note, i, j] = heap.top();
        heap.pop();
        if (!pieces.count(i) || !pieces.count(j)) continue;
        const std::vector<int> a = pieces[i], b = pieces[j];
        // _zwischen: cells between the two ends
        const int ay = Y(a.back()), ax = X(a.back()), by = Y(b.front()), bx = X(b.front());
        const int n = std::max(std::abs(by - ay), std::abs(bx - ax));
        std::vector<int> mid;
        for (int s = 1; s < n; ++s)
            mid.push_back(roundEven(ay + double(by - ay) * s / n) * q + roundEven(ax + double(bx - ax) * s / n));
        bool blocked = false;
        for (int p : mid) blocked = blocked || dir[size_t(p)] >= 0;
        if (blocked) continue;
        std::vector<int> pts{a.back()};
        pts.insert(pts.end(), mid.begin(), mid.end());
        pts.push_back(b.front());
        for (size_t k = 0; k < mid.size(); ++k) {
            const Vec t{double(X(pts[k + 2]) - X(pts[k])), double(Y(pts[k + 2]) - Y(pts[k]))};
            const double l = std::max(1e-6, std::sqrt(t.x * t.x + t.y * t.y));
            dir[size_t(mid[k])] = fallDirection({t.x / l, t.y / l});
        }
        pieces.erase(i);
        pieces.erase(j);
        std::vector<int> joined = a;
        joined.insert(joined.end(), mid.begin(), mid.end());
        joined.insert(joined.end(), b.begin(), b.end());
        pieces[nr] = joined;
        remember(nr);
        for (auto& [k, kk] : pieces)
            if (k != nr) {
                if (near(nr, k)) pair(nr, k);
                if (near(k, nr)) pair(k, nr);
            }
        ++nr;
    }

    // 2. Drop snippets.
    std::vector<std::vector<int>> open;
    for (auto& [i, k] : pieces) {
        if (static_cast<int>(k.size()) >= minLen) open.push_back(k);
        else for (int p : k) dir[size_t(p)] = -1;
    }
    for (const auto& k : rings)
        if (k.size() < 6)
            for (int p : k) dir[size_t(p)] = -1;

    // 3. In tight zones extend up to just before the next wall.
    std::vector<uint8_t> taken(size_t(h) * q, 0);
    for (size_t p = 0; p < dir.size(); ++p) taken[p] = dir[p] >= 0;
    std::stable_sort(open.begin(), open.end(), [](const auto& u, const auto& v) { return u.size() < v.size(); });
    for (auto& chain : open)
        for (const bool back : {true, false}) {
            Vec t = endDir(chain, back);
            if (!back) t = {-t.x, -t.y};
            const int e = back ? chain.back() : chain.front();
            double y = Y(e), x = X(e);
            const int steps = static_cast<int>(rng.integers(4, 13));
            const double turn = rng.normal() * .06;
            std::vector<uint8_t> own(size_t(h) * q, 0);
            for (int p : chain) own[size_t(p)] = 1;
            std::vector<int> added;
            for (int s = 0; s < steps; ++s) {
                const double ang = std::atan2(t.y, t.x) + turn;
                t = {std::cos(ang), std::sin(ang)};
                x += t.x;
                y += t.y;
                const int py = roundEven(y), px = roundEven(x);
                if (py < 0 || py >= h || px < 0 || px >= q || zone.at(px, py) != 1) break;
                const int p = py * q + px;
                if (own[size_t(p)]) continue;
                bool hit = false;
                for (int c = py - 2; c <= py + 2 && !hit; ++c)
                    for (int d = px - 2; d <= px + 2 && !hit; ++d)
                        hit = c >= 0 && c < h && d >= 0 && d < q && taken[size_t(c) * q + d] && !own[size_t(c) * q + d];
                if (hit) break;
                added.push_back(p);
                own[size_t(p)] = 1;
            }
            if (added.size() < 2) continue;
            const int fd = fallDirection(back ? t : Vec{-t.x, -t.y});
            for (int p : added) {
                dir[size_t(p)] = fd;
                taken[size_t(p)] = 1;
            }
            if (back) chain.insert(chain.end(), added.begin(), added.end());
            else chain.insert(chain.begin(), added.rbegin(), added.rend());
        }
    Grid<int32_t> out(q, h, 0);
    for (size_t p = 0; p < dir.size(); ++p) out.v[p] = dir[p] >= 0 ? dir[p] + 1 : 0;
    return out;
}

/// nordkanten_ausduennen: parallel north edges close together; the shorter one gives way.
Grid<int32_t> thinNorthEdges(const Grid<int32_t>& wv, int gap = 3) {
    Grid<int32_t> out = wv;
    const int h = wv.h, q = wv.w;
    std::vector<int> len(size_t(h) * q, 0);
    const ChainList cl = chainsOf(wv);
    for (const auto& c : cl.chains)
        for (int p : c) len[size_t(p)] = static_cast<int>(c.size());
    std::vector<int> cells;
    for (size_t p = 0; p < wv.v.size(); ++p) if (isNorth(wv.v[p])) cells.push_back(static_cast<int>(p));
    std::stable_sort(cells.begin(), cells.end(), [&](int a, int b) { return len[size_t(a)] < len[size_t(b)]; });
    for (int p : cells) {
        const int y = p / q, x = p % q;
        if (!out.at(x, y)) continue;
        for (int dy = -gap; dy <= gap; ++dy) {
            if (dy >= -1 && dy <= 1) continue;
            const int c = y + dy;
            if (c >= 0 && c < h && isNorth(out.at(x, c)) && len[size_t(c) * q + x] >= len[size_t(p)]) {
                out.at(x, y) = 0;
                break;
            }
        }
    }
    return out;
}

/// ein_boden_je_wand: every wall and its rim get one ground (mud or sand, by
/// majority); then single 4x4 blocks are dissolved.
Grid<int32_t> oneGroundPerWall(const Grid<int32_t>& wv, const Grid<int32_t>& ground) {
    const int h = wv.h, q = wv.w;
    Mask wide(q, h, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < q; ++x) {
            if (wv.at(x, y) <= 0) continue;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if (wide.inside(x + dx, y + dy)) wide.at(x + dx, y + dy) = 1;
        }
    Grid<int32_t> labels;
    const int n = label8(wide, labels);
    std::vector<std::vector<int>> cellsOf(size_t(n) + 1);
    for (size_t p = 0; p < labels.v.size(); ++p)
        if (wv.v[p] > 0) cellsOf[size_t(labels.v[p])].push_back(static_cast<int>(p));
    Grid<int32_t> b = ground;
    struct Wall { size_t n; int g; const std::vector<int>* cells; };
    std::vector<Wall> walls;
    for (int i = 1; i <= n; ++i) {
        const auto& z = cellsOf[size_t(i)];
        if (z.empty()) continue;
        int f[5] = {0, 0, 0, 0, 0}, sum = 0;
        for (int p : z) { f[b.v[size_t(p)]] += 1; sum += 1; }
        walls.push_back({z.size(), 2 * f[kSand] > sum ? kSand : kMud, &z});
    }
    std::stable_sort(walls.begin(), walls.end(), [](const Wall& u, const Wall& v) { return u.n < v.n; });   // large ones win
    for (const Wall& wl : walls)
        for (int p : *wl.cells) {
            const int y = p / q, x = p % q;
            for (int yy = std::max(0, y - 2); yy < std::min(h, y + 3); ++yy)
                for (int xx = std::max(0, x - 2); xx < std::min(q, x + 3); ++xx) b.at(xx, yy) = wl.g;
        }
    const int gh = h / 2, gw = q / 2;
    Grid<int32_t> coarse(gw, gh, 0);
    for (int y = 0; y < gh; ++y)
        for (int x = 0; x < gw; ++x) {
            int c[5] = {0, 0, 0, 0, 0};
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) c[b.at(2 * x + dx, 2 * y + dy)] += 1;
            coarse.at(x, y) = static_cast<int>(std::max_element(c, c + 5) - c);
        }
    for (int pass = 0; pass < 2; ++pass) {
        const Grid<int32_t> before = coarse;
        auto at = [&](int x, int y) { return before.at(std::clamp(x, 0, gw - 1), std::clamp(y, 0, gh - 1)); };
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                const int nb[4] = {at(x + 1, y), at(x, y + 1), at(x - 1, y), at(x, y - 1)};
                int same = 0;
                for (int v : nb) same += v == before.at(x, y);
                if (same) continue;
                int c[5] = {0, 0, 0, 0, 0};
                for (int v : nb) c[v] += 1;
                coarse.at(x, y) = static_cast<int>(std::max_element(c, c + 5) - c);
            }
    }
    return upscale2(coarse);
}

}  // namespace

Sketch drawSketch(const LearnedModel& m, const Params& p0, nn::Pool& pool, const ProgressFn& progress) {
    if (p0.width % 8 || p0.height % 8) throw std::runtime_error("the learned generator needs map sizes that are multiples of 8");
    // With the large shape, mountains are spread over the whole world (Python: _ketten_verteilt).
    Params p = p0;
    if (p.coherence > 0) p.mountainSpread = true;
    Rng rng = stream(p.seed, 60);
    const MountainPlan world = planWorld(p, Mask());
    const Grid<int8_t> z = worldZones(p, world);
    Grid<int32_t> raster = zoneRaster(z, p, rng);
    Grid<int32_t> ground0 = mountainGround(raster, rng);
    // Mix sliders -> wanted share per ground (mud, rock, sand, vegetation; no snow).
    double target[5] = {0, 0, 0, 0, 0}, sum = 0;
    const char* ids[4] = {"mud", "roc", "san", "veg"};
    for (int k = 0; k < 4; ++k)
        for (const auto& [id, share] : p.mix)
            if (id == ids[k]) target[k] = share;
    for (double t : target) sum += t;
    const double* want = nullptr;
    if (sum > 0) {
        for (double& t : target) t /= sum;
        want = target;
    }
    const bool small = p.width * p.height <= 128 * 128;
    const int R = kWorldMargin / 2, sh = raster.h, sw = raster.w;
    const double calmT = std::clamp(p.calmTerrain, 0.0, 1.0);
    const double area = double(p.width) * p.height / 4;

    // Gradient on the sketch raster: a preference per ground and cell, the terrain per cell.
    Floats u, rough;
    std::vector<Floats> pref, bias;
    const bool gradient = gradientField(p, 2 * sw, 2 * sh, kWorldMargin, u);
    if (gradient) {
        Floats full;
        gradientTerrain(p, 2 * sw, 2 * sh, kWorldMargin, full);
        Floats uu(sw, sh);
        rough = Floats(sw, sh);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x) {
                uu.at(x, y) = u.at(2 * x, 2 * y);
                rough.at(x, y) = full.at(2 * x, 2 * y);
            }
        double base[5];
        double bsum = 0;
        for (int k = 0; k < 5; ++k) bsum += kGroundShare[k];
        for (int k = 0; k < 5; ++k) base[k] = want ? target[k] : kGroundShare[k] / bsum;
        double ends[2][5];
        for (int e = 0; e < 2; ++e)
            for (int k = 0; k < 5; ++k) {
                const int g = p.gradientGround[e];
                ends[e][k] = (g >= 0 && g < 4) ? .2 * base[k] + (k == g ? .8 : 0.0) : base[k];
            }
        pref.assign(5, Floats(sw, sh));
        for (int k = 0; k < 5; ++k) {
            double mean = 0;
            for (size_t i = 0; i < uu.v.size(); ++i) {
                pref[size_t(k)].v[i] = (1 - uu.v[i]) * ends[0][k] + uu.v[i] * ends[1][k];
                mean += pref[size_t(k)].v[i];
            }
            target[k] = mean / double(uu.v.size());
        }
        want = target;
        bias.assign(5, Floats(sw, sh));
        for (int k = 0; k < 5; ++k)
            for (size_t i = 0; i < uu.v.size(); ++i)
                bias[size_t(k)].v[i] = static_cast<float>(1.2 * nn::detLog((pref[size_t(k)].v[i] + .004) / (target[k] + .004)));
    }

    // Large shape: its contours go into the network as fixed walls.
    Rng shapeRng = stream(p.seed, 66);
    Grid<int32_t> lines = extra::largeShape(raster, std::clamp(p.coherence, 0.0, 1.0), shapeRng,
                                            std::clamp(p.mountainDensity, 0.0, 1.0), calmT, gradient ? &rough : nullptr);
    auto isLine = [&](size_t i) { return lines.v[i] < 9; };
    if (gradient) {
        // Flat side: fewer large-shape lines and fewer of the network's own cliffs.
        Rng r70 = stream(p.seed, 70);
        const Floats loose = extra::softNoise(sw, sh, r70, {{5, 1.0}});
        std::vector<int> order(loose.v.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return loose.v[size_t(a)] < loose.v[size_t(b)]; });
        Floats rank(sw, sh);
        for (size_t k = 0; k < order.size(); ++k) rank.v[size_t(order[k])] = double(k) / double(order.size());
        for (size_t i = 0; i < lines.v.size(); ++i) {
            const double flat = std::clamp((.5 - rough.v[i]) * 2, 0.0, 1.0);
            if (rank.v[i] < flat && isLine(i)) lines.v[i] = 9;
            if (raster.v[i] == 0 && rank.v[i] < 1.1 * flat) raster.v[i] = 2;
        }
        lines = extra::dropShortWalls(lines, 9, 9);
    }
    Mask lineMask(sw, sh);
    for (size_t i = 0; i < lineMask.v.size(); ++i) lineMask.v[i] = isLine(i);
    {
        const Mask near1 = dilate4(lineMask, 1);
        for (size_t i = 0; i < raster.v.size(); ++i)
            if (near1.v[i] && raster.v[i] == 2) raster.v[i] = 0;
    }
    if (calmT > 0 && lineMask.any()) {
        // Away from the large lines the network draws many short cliffs of its own;
        // calm terrain leaves more of that open. Inside mountains the terraces give the shape.
        const Mask near5 = dilate4(lineMask, 5), near3 = dilate4(lineMask, 3);
        Rng r67 = stream(p.seed, 67);
        const Floats pick = extra::softNoise(sw, sh, r67, {{6, 1.0}});
        const double q1 = extra::quantileOf(pick.v, .9 * calmT);
        for (size_t i = 0; i < raster.v.size(); ++i)
            if (!near5.v[i] && raster.v[i] == 0 && pick.v[i] < q1) raster.v[i] = 2;
        const double q2 = extra::quantileOf(pick.v, std::min(1.0, 1.2 * calmT));
        for (size_t i = 0; i < raster.v.size(); ++i)
            if (raster.v[i] == 1 && !near3.v[i] && pick.v[i] < q2) raster.v[i] = 2;
    }
    int tightCells = 0, lineCells = 0;
    for (int y = R; y < sh - R; ++y)
        for (int x = R; x < sw - R; ++x) {
            tightCells += raster.at(x, y) == 1;
            lineCells += lines.at(x, y) < 9;
        }
    const double mapCells = double(sh - 2 * R) * (sw - 2 * R);
    const double goal = .035 + .20 * p.cliffs + .3 * tightCells / mapCells + .9 * lineCells / mapCells;
    const Grid<int32_t>* fixedWalls = lineMask.any() ? &lines : nullptr;
    const std::vector<Floats>* gBias = gradient ? &bias : nullptr;

    const double calmB = std::clamp(p.biomeCalm, 0.0, 1.0);
    if (calmB > 0) {
        // Coarse to fine: a draft, the calm areas from it, then walls drawn to match.
        auto [dw, dg] = drawFromNet(m, raster, ground0, want, rng, 10, pool, progress, fixedWalls, gBias);
        dw = tidyWalls(dw, raster, rng);
        Rng r63 = stream(p.seed, 63);
        std::vector<Floats> pref4;
        if (gradient) pref4.assign(pref.begin(), pref.begin() + 4);
        Grid<int32_t> land = extra::calmBiomes(dw, dg, calmB, r63, gradient ? &pref4 : nullptr);
        for (size_t i = 0; i < land.v.size(); ++i)
            if (ground0.v[i] < 5) land.v[i] = ground0.v[i];
        land = extra::dropSmallPatches(land, Grid<int32_t>(sw, sh, 0), .02 * calmB * area);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x) ground0.at(x, y) = land.at(x / 2 * 2, y / 2 * 2);
    }
    bool have = false;
    double best = 0;
    Grid<int32_t> wall, ground;
    for (int c = 0; c < (small ? 2 : 1); ++c) {
        auto [wv, gv] = drawFromNet(m, raster, ground0, want, rng, 16, pool, progress, fixedWalls, gBias);
        wv = thinNorthEdges(tidyWalls(wv, raster, rng));
        if (p.coherence > 0) wv = extra::dropShortWalls(wv, static_cast<int>(12 + 16 * calmT), 0);
        int walls = 0;
        for (int y = R; y < sh - R; ++y)
            for (int x = R; x < sw - R; ++x) walls += wv.at(x, y) > 0;
        const double note = std::abs(walls / mapCells - goal);
        if (!have || note < best) {
            have = true;
            best = note;
            wall = std::move(wv);
            ground = std::move(gv);
        }
    }
    for (int32_t& g : ground.v) if (g > 4) g = 0;
    if (p.symmetry != Sym::None) {
        mirrorSketch(wall, p.symmetry, true);
        mirrorSketch(ground, p.symmetry, false);
    }
    if (calmB > 0) {
        for (int32_t& g : ground.v) if (g > 3) g = 0;
        ground = extra::narrowRim(wall, ground);
        ground = extra::dropSmallPatches(ground, wall, .02 * calmB * area);
        ground = extra::narrowRim(wall, ground);
    } else {
        ground = oneGroundPerWall(wall, ground);
    }

    // Volcano sites (with symmetry in the base half, mirrored along), then the lava cores.
    std::vector<extra::VolcanoSite> sites;
    const int counts[3] = {std::max(0, p.volcanoSizes[0]), std::max(0, p.volcanoSizes[1]), std::max(0, p.volcanoSizes[2])};
    if (counts[0] + counts[1] + counts[2] > 0) {
        Grid<int32_t> index(sw, sh);
        std::iota(index.v.begin(), index.v.end(), 0);
        Grid<int32_t> image = index;
        if (p.symmetry != Sym::None) mirrorSketch(image, p.symmetry, false);
        Mask moved(sw, sh);
        for (size_t i = 0; i < moved.v.size(); ++i) moved.v[i] = image.v[i] != index.v[i];
        const Mask movedNear = dilate4(moved, 8);
        Mask allowed(sw, sh);
        for (size_t i = 0; i < allowed.v.size(); ++i) allowed.v[i] = !movedNear.v[i];
        Rng r68 = stream(p.seed, 68);
        sites = extra::volcanoSites(ground, wall, counts, R, R, sw - R, sh - R, r68, &allowed);
        if (p.symmetry != Sym::None) {
            mirrorSketch(ground, p.symmetry, false);
            std::vector<extra::VolcanoSite> all;
            for (const auto& s : sites)
                for (size_t i = 0; i < image.v.size(); ++i)
                    if (image.v[i] == index.at(s.x, s.y))
                        all.push_back({static_cast<int>(i) / sw, static_cast<int>(i) % sw, s.size, s.lava, s.r});
            sites = std::move(all);
        }
    }
    Rng r64 = stream(p.seed, 64);
    Mask lava = extra::lavaPlan(ground, wall, std::clamp(p.lavaRock, 0.0, 1.0), r64);
    for (const auto& s : sites)
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x) {
                const int d = (y - s.y) * (y - s.y) + (x - s.x) * (x - s.x);
                if (d <= (s.r + 2) * (s.r + 2)) lava.at(x, y) = 0;
                if (s.lava && d <= s.r * s.r) lava.at(x, y) = 1;
            }
    if (p.symmetry != Sym::None) mirrorSketch(lava, p.symmetry, false);

    // Crop only now: walls run off the map.
    Sketch sk;
    sk.wall = Grid<int32_t>(sw - 2 * R, sh - 2 * R);
    sk.ground = Grid<int32_t>(sw - 2 * R, sh - 2 * R);
    sk.lava = Mask(sw - 2 * R, sh - 2 * R);
    for (int y = 0; y < sk.wall.h; ++y)
        for (int x = 0; x < sk.wall.w; ++x) {
            sk.wall.at(x, y) = wall.at(x + R, y + R);
            sk.ground.at(x, y) = ground.at(x + R, y + R);
            sk.lava.at(x, y) = lava.at(x + R, y + R);
        }
    for (const auto& s : sites) sk.volcanoes.push_back({s.y - R, s.x - R, s.size, s.lava, s.r});
    sk.zones = Grid<int8_t>(p.width, p.height);
    for (int y = 0; y < p.height; ++y)
        for (int x = 0; x < p.width; ++x) sk.zones.at(x, y) = z.at(x + kWorldMargin, y + kWorldMargin);
    sk.plan = cutOutPlan(world, p.width, p.height);
    return sk;
}

}  // namespace om::gen
