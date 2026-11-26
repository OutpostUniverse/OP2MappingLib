// Mountains, gorges, plateaus and valleys (Python: mapgen_berge.py).
// Ported from fefi-byte's op2-codegen-editor (MIT, see THIRD-PARTY-NOTICES.md).
#include "mountains.hpp"

#include "../blend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace om::gen {

namespace {

constexpr double kPi = 3.14159265358979323846;
// Distance between two terraces: (sides and south, north). North walls are one
// cell thick, the others three; below 4 cells the walls cut into each other.
constexpr double kTightSide = 4.0, kTightNorth = 3.0;   // ABSTAND_ENG
constexpr double kWideSide = 7.0, kWideNorth = 5.0;     // ABSTAND_WEIT
constexpr double kNeckHalf = 4.5;                       // HALS_HALB
constexpr double kGorgeHalf = 4.6;                      // SCHLUCHT_HALB
constexpr double kValleyHalf = 13.0;                    // TAL_HALB
constexpr int kBuildSquare = 7;                         // BAUPLATZ
// Attempts per wall and runs per mountain of the library drawer (editor: maplib.Zeichner).
constexpr int kAttempts = 4, kRuns = 4;
constexpr int kNorthCliffs = 8, kCliffsLowSide = 10;

/// Python's round(): halves to even (the default floating-point rounding mode).
inline int roundEven(double v) { return static_cast<int>(std::nearbyint(v)); }

Mask maskNot(const Mask& m) {
    Mask o(m.w, m.h);
    for (size_t i = 0; i < m.v.size(); ++i) o.v[i] = m.v[i] ? 0 : 1;
    return o;
}
void andWith(Mask& a, const Mask& b) { for (size_t i = 0; i < a.v.size(); ++i) a.v[i] = (a.v[i] && b.v[i]) ? 1 : 0; }
void orWith(Mask& a, const Mask& b) { for (size_t i = 0; i < a.v.size(); ++i) a.v[i] = (a.v[i] || b.v[i]) ? 1 : 0; }
Mask maskAnd(Mask a, const Mask& b) { andWith(a, b); return a; }
int countAnd(const Mask& a, const Mask& b) {
    int n = 0;
    for (size_t i = 0; i < a.v.size(); ++i) n += (a.v[i] && b.v[i]) ? 1 : 0;
    return n;
}
bool anyAnd(const Mask& a, const Mask& b) {
    for (size_t i = 0; i < a.v.size(); ++i) if (a.v[i] && b.v[i]) return true;
    return false;
}
Mask fromIndices(const std::vector<int>& idx, int w, int h) {
    Mask m(w, h, 0);
    for (int i : idx) m.v[size_t(i)] = 1;
    return m;
}
Mask crop(const Mask& m, int x0, int y0, int w, int h) {
    Mask o(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) o.at(x, y) = m.at(x + x0, y + y0);
    return o;
}
template <class T>
Grid<T> cropGrid(const Grid<T>& m, int x0, int y0, int w, int h) {
    Grid<T> o(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) o.at(x, y) = m.at(x + x0, y + y0);
    return o;
}
template <class T>
void paste(Grid<T>& dst, const Grid<T>& src, int x0, int y0) {
    for (int y = 0; y < src.h; ++y)
        for (int x = 0; x < src.w; ++x) dst.at(x + x0, y + y0) = src.at(x, y);
}

/// numpy.quantile with linear interpolation.
double quantile(std::vector<double> v, double q) {
    if (v.empty()) return std::numeric_limits<double>::infinity();
    std::sort(v.begin(), v.end());
    const double pos = q * double(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(v.size() - 1, lo + 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
}

/// _rauschen: smooth noise with a fixed cell SIZE, so shapes are equally large on every map size.
Floats cellNoise(int h, int w, Rng& rng, int cell) {
    const int gh = h / cell + 3, gw = w / cell + 3;
    std::vector<double> g(size_t(gh) * gw);
    for (double& x : g) x = rng.uniform();
    Floats out(w, h);
    for (int y = 0; y < h; ++y) {
        const double ys = double(y) / cell;
        const int y0 = static_cast<int>(ys);
        double fy = ys - y0;
        fy = fy * fy * (3 - 2 * fy);
        for (int x = 0; x < w; ++x) {
            const double xs = double(x) / cell;
            const int x0 = static_cast<int>(xs);
            double fx = xs - x0;
            fx = fx * fx * (3 - 2 * fx);
            const double a = g[size_t(y0) * gw + x0], b = g[size_t(y0) * gw + x0 + 1];
            const double c = g[size_t(y0 + 1) * gw + x0], d = g[size_t(y0 + 1) * gw + x0 + 1];
            out.at(x, y) = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
        }
    }
    return out;
}

/// _schrumpf: shrink by the true distance; by `north` on the north side. Off the map counts as empty.
Mask shrink(const Mask& m, double side, double north) {
    const int w = m.w, h = m.h;
    Mask out = m;
    for (uint8_t& c : out.v) c = c ? 1 : 0;
    const int r = static_cast<int>(std::ceil(std::max(side, north)));
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            const double ry = dy < 0 ? north : side;
            if (!(dx || dy) || (dx / side) * (dx / side) + (dy / ry) * (dy / ry) > 1) continue;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    if (!out.at(x, y)) continue;
                    const int sx = x + dx, sy = y + dy;
                    if (sx < 0 || sy < 0 || sx >= w || sy >= h || !m.at(sx, sy)) out.at(x, y) = 0;
                }
        }
    return out;
}

/// _weite: round dilation; off the map counts as empty.
Mask widen(const Mask& m, double r) {
    const int w = m.w, h = m.h;
    Mask out = m;
    for (uint8_t& c : out.v) c = c ? 1 : 0;
    const int n = static_cast<int>(std::ceil(r));
    for (int dy = -n; dy <= n; ++dy)
        for (int dx = -n; dx <= n; ++dx) {
            if (!(dx || dy) || (dx / r) * (dx / r) + (dy / r) * (dy / r) > 1) continue;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    if (out.at(x, y)) continue;
                    const int sx = x + dx, sy = y + dy;
                    if (sx >= 0 && sy >= 0 && sx < w && sy < h && m.at(sx, sy)) out.at(x, y) = 1;
                }
        }
    return out;
}

/// _erodiere: m & ~dehne(~m, n); the map border does not erode.
Mask erode(const Mask& m, int n) { return maskAnd(m, maskNot(dilate4(maskNot(m), n))); }

/// _saeubern: smooth, drop specks and holes, keep a free border.
Mask tidy(const Mask& m, int minCells, int border = 5) {
    Mask f = removeSmallBits(smoothMask(m, 3), minCells);
    for (int y = 0; y < f.h; ++y)
        for (int x = 0; x < f.w; ++x)
            if (y < border || x < border || y >= f.h - border || x >= f.w - border) f.at(x, y) = 0;
    return f;
}

Mask mirrored(Mask m, Sym s) {
    if (s != Sym::None) applySymmetry(m, s);
    return m;
}

struct Chain { double cx, cy, angle, len, wid, tilt; };

/// With symmetry only the base half counts, the rest is mirrored.
void mirrorBase(double& cx, double& cy, int w, int h, Sym sym) {
    if ((sym == Sym::MirrorX || sym == Sym::Quad) && cx >= w / 2.0) cx = w - 1 - cx;
    if ((sym == Sym::MirrorY || sym == Sym::Rot180 || sym == Sym::Quad) && cy >= h / 2.0) cy = h - 1 - cy;
}

double nearestChain(const std::vector<Chain>& chains, double cx, double cy) {
    double best = 1e9;
    for (const Chain& k : chains) best = std::min(best, std::hypot(cx - k.cx, cy - k.cy));
    return best;
}

/// _ketten: one mountain chain per 128x128 of world area.
std::vector<Chain> chainsCentred(int h, int w, Rng& rng, Sym sym) {
    const int count = std::max(1, roundEven(double(w) * h / (128.0 * 128.0)));
    const double len = 0.45 * std::min(std::min(w, h), 128);
    const double wid = 0.38 * len;
    std::vector<Chain> chains;
    for (int i = 0; i < count; ++i) {
        double bestD = -1, bx = 0, by = 0;
        for (int v = 0; v < 3; ++v) {
            double cx = rng.uniform(kWorldMargin, w - kWorldMargin);
            double cy = rng.uniform(kWorldMargin, h - kWorldMargin);
            mirrorBase(cx, cy, w, h, sym);
            const double d = nearestChain(chains, cx, cy);
            if (bestD < 0 || d > bestD) { bestD = d; bx = cx; by = cy; }
        }
        double angle = rng.uniform(-.6, .6);
        if ((h >= w && rng.uniform() < .5) || h > w) angle += kPi / 2;
        const double l = std::min(w, h) >= 128 ? len : len * 1.6;
        chains.push_back({bx, by, angle, l, wid, rng.uniform(0, 2 * kPi)});
    }
    return chains;
}

/// _ketten_verteilt: more and more varied mountains spread over the whole world.
std::vector<Chain> chainsSpread(int h, int w, Rng& rng, Sym sym) {
    const int count = std::max(2, roundEven(1.4 * w * h / (128.0 * 128.0)));
    std::vector<Chain> chains;
    for (int i = 0; i < count; ++i) {
        double bestD = -1, bx = 0, by = 0;
        for (int v = 0; v < 4; ++v) {
            double cx = rng.uniform(8, w - 8), cy = rng.uniform(8, h - 8);
            mirrorBase(cx, cy, w, h, sym);
            const double d = nearestChain(chains, cx, cy);
            if (bestD < 0 || d > bestD) { bestD = d; bx = cx; by = cy; }
        }
        const double len = rng.uniform(.22, .45) * std::min(std::min(w, h), 160);
        const double angle = rng.uniform(0, kPi);
        const double wid = len * rng.uniform(.45, .95);
        chains.push_back({bx, by, angle, len, wid, rng.uniform(0, 2 * kPi)});
    }
    return chains;
}

/// Einschnitt.bauen: ends where the sole stops; false if hardly anything is left.
bool buildCut(Cut& out, const Mask& sole, const Mask& area, const Floats& along, const Mask& allowed) {
    if (sole.count() < 20) return false;
    double a = std::numeric_limits<double>::infinity(), b = -a;
    for (size_t i = 0; i < sole.v.size(); ++i)
        if (sole.v[i]) { a = std::min(a, along.v[i]); b = std::max(b, along.v[i]); }
    out.sole = sole;
    out.area = area;
    out.along = along;
    out.endA = Mask(sole.w, sole.h);
    out.endB = Mask(sole.w, sole.h);
    for (size_t i = 0; i < sole.v.size(); ++i) {
        const bool base = area.v[i] && allowed.v[i];
        out.endA.v[i] = base && std::abs(along.v[i] - a) < 1.5;
        out.endB.v[i] = base && std::abs(along.v[i] - b) < 1.5;
    }
    return true;
}

struct CutPlan {
    bool valley;
    Floats d, vv;
    double half;
    Mask sole;
};

/// _plane_berge
void planRanges(MountainPlan& plan, const Params& p, Rng& rng, const Mask& freeCells) {
    const int w = freeCells.w, h = freeCells.h, R = kWorldMargin;
    const Sym sym = p.symmetry;
    Floats X(w, h), Y(w, h);
    {
        const Floats a = cellNoise(h, w, rng, 16), b = cellNoise(h, w, rng, 8);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) X.at(x, y) = x + (a.at(x, y) - .5) * 22 + (b.at(x, y) - .5) * 9;
        const Floats c = cellNoise(h, w, rng, 16), d = cellNoise(h, w, rng, 8);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) Y.at(x, y) = y + (c.at(x, y) - .5) * 22 + (d.at(x, y) - .5) * 9;
    }
    const std::vector<Chain> chains = p.mountainSpread ? chainsSpread(h, w, rng, sym) : chainsCentred(h, w, rng, sym);
    Floats fmax(w, h, -std::numeric_limits<double>::infinity());
    Grid<int> nearest(w, h, 0);
    for (size_t nr = 0; nr < chains.size(); ++nr) {
        const Chain& k = chains[nr];
        const double c = std::cos(k.angle), s = std::sin(k.angle);
        for (size_t i = 0; i < X.v.size(); ++i) {
            const double u = (X.v[i] - k.cx) * c + (Y.v[i] - k.cy) * s;
            const double v = -(X.v[i] - k.cx) * s + (Y.v[i] - k.cy) * c;
            const double f = 1 - std::sqrt((u / k.len) * (u / k.len) + (v / k.wid) * (v / k.wid));
            if (f > fmax.v[i]) { fmax.v[i] = f; nearest.v[i] = static_cast<int>(nr); }
        }
    }
    const Floats ridgeNoise = cellNoise(h, w, rng, 14);
    const Floats n18 = cellNoise(h, w, rng, 18), n7 = cellNoise(h, w, rng, 7);
    Floats f(w, h), tilt(w, h, 0.0);
    for (size_t i = 0; i < f.v.size(); ++i) {
        const double ridge = 1 - std::abs(2 * ridgeNoise.v[i] - 1);
        f.v[i] = fmax.v[i] + .42 * (n18.v[i] - .5) + .32 * (ridge - .6) + .18 * (n7.v[i] - .5);
    }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const Chain& k = chains[size_t(nearest.at(x, y))];
            tilt.at(x, y) = (std::cos(k.tilt) * (x - k.cx) + std::sin(k.tilt) * (y - k.cy)) / 60.;
        }
    if (sym != Sym::None) {
        applySymmetry(f, sym);
        applySymmetry(tilt, sym);
    }
    const double density = std::clamp(p.mountainDensity, 0.0, 1.0);
    const double side = kWideSide + density * (kTightSide - kWideSide);
    const double north = kWideNorth + density * (kTightNorth - kWideNorth);
    const Mask open = maskNot(dilate4(freeCells, 6));
    if (!open.any()) return;
    Mask onMap(w, h, 0);
    for (int y = R; y < h - R; ++y)
        for (int x = R; x < w - R; ++x) onMap.at(x, y) = 1;
    const double mapCells = double(onMap.count());

    // Threshold chosen so the requested share of the map results, measured on the map.
    auto threshold = [&](const Mask& area) {
        std::vector<double> vals;
        for (size_t i = 0; i < f.v.size(); ++i) if (area.v[i] && onMap.v[i]) vals.push_back(f.v[i]);
        if (vals.empty()) return std::numeric_limits<double>::infinity();
        return quantile(vals, 1 - std::min(.9, p.mountains * mapCells / double(vals.size())));
    };
    auto above = [&](const Mask& area) {
        const double t = threshold(area);
        Mask m(w, h);
        for (size_t i = 0; i < m.v.size(); ++i) m.v[i] = (area.v[i] && f.v[i] > t) ? 1 : 0;
        return m;
    };
    Mask raw = above(open);
    Mask keep = open;

    // Put valleys and gorges where they cut through the most mountain.
    std::vector<CutPlan> cuts;
    std::vector<bool> kinds;
    for (int i = 0; i < p.valleys; ++i) kinds.push_back(true);
    for (int i = 0; i < p.gorges; ++i) kinds.push_back(false);
    Floats uu(w, h), vv(w, h), d(w, h);
    for (const bool valley : kinds) {
        const double half = valley ? kValleyHalf : kGorgeHalf;
        bool have = false;
        double bestNote = 0;
        CutPlan best;
        for (const Chain& k : chains) {
            const double c = std::cos(k.angle), sn = std::sin(k.angle);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    uu.at(x, y) = (x - k.cx) * c + (y - k.cy) * sn;
                    vv.at(x, y) = -(x - k.cx) * sn + (y - k.cy) * c;
                }
            const double ph = rng.uniform(0, 2 * kPi);
            for (int step = 0; step < 18; ++step) {
                const double place = (-.85 + 1.7 * step / 17.0) * k.len;
                for (size_t i = 0; i < d.v.size(); ++i) {
                    const double dv = uu.v[i] - place - 4 * std::sin(vv.v[i] / 9 + ph) - 2 * std::sin(vv.v[i] / 5 - ph);
                    d.v[i] = std::abs(vv.v[i]) < k.wid * 2.4 ? dv : 1e9;
                }
                double far = 1e9;
                for (const CutPlan& s : cuts)
                    for (size_t i = 0; i < d.v.size(); ++i)
                        if (s.sole.v[i]) far = std::min(far, std::abs(d.v[i]));
                if (far < half + 12) continue;
                // Mountain has to remain on both sides.
                int left = 0, right = 0;
                for (size_t i = 0; i < d.v.size(); ++i) {
                    if (!raw.v[i] || !onMap.v[i]) continue;
                    left += d.v[i] < -half && d.v[i] > -half - 10;
                    right += d.v[i] > half && d.v[i] < half + 10;
                }
                const double note = std::min(left, right) * rng.uniform(.8, 1.0);
                if (!have || note > bestNote) {
                    have = true;
                    bestNote = note;
                    best.valley = valley;
                    best.d = d;
                    best.vv = vv;
                    best.half = half;
                    best.sole = Mask(w, h);
                    for (size_t i = 0; i < d.v.size(); ++i) best.sole.v[i] = std::abs(d.v[i]) < 1.5;
                }
            }
        }
        if (!have || bestNote < 40) continue;
        for (size_t i = 0; i < keep.v.size(); ++i) keep.v[i] = keep.v[i] && std::abs(best.d.v[i]) > best.half;
        cuts.push_back(std::move(best));
        // What the cut removes is added elsewhere, the share stays.
        raw = above(keep);
    }
    Mask m = tidy(mirrored(raw, sym), 90);
    // Close narrow bays, take off thin necks and spurs: the walls of both sides
    // overlap there and the mountain leaks.
    m = maskAnd(maskNot(widen(maskNot(widen(m, kNeckHalf)), kNeckHalf)), mirrored(keep, sym));
    m = maskAnd(widen(shrink(m, kNeckHalf, kNeckHalf), kNeckHalf), m);
    m = mirrored(tidy(m, 90), sym);
    if (!m.any()) return;
    plan.terraces.push_back(m);
    Floats g(w, h);
    for (size_t i = 0; i < g.v.size(); ++i) g.v[i] = f.v[i] + .5 * tilt.v[i];
    for (int t = 0; t < 7; ++t) {
        const Mask prev = plan.terraces.back();
        // North and south distance differ; with symmetry the terrace has to keep both.
        Mask inner = shrink(prev, side, north);
        if (sym == Sym::MirrorY || sym == Sym::Quad || sym == Sym::Rot180) {
            Mask flipped = inner;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    flipped.at(x, y) = sym == Sym::Rot180 ? inner.at(w - 1 - x, h - 1 - y) : inner.at(x, h - 1 - y);
            andWith(inner, flipped);
        }
        Mask n(w, h, 0);
        for (const auto& idx : components(prev)) {
            const Mask comp = fromIndices(idx, w, h);
            const Mask in = maskAnd(inner, comp);
            if (in.count() < 14) continue;
            std::vector<double> vals;
            for (int i : idx) vals.push_back(g.v[size_t(i)]);
            const double q = quantile(vals, rng.uniform(.02, .30));
            for (size_t i = 0; i < n.v.size(); ++i) if (in.v[i] && g.v[i] > q) n.v[i] = 1;
        }
        n = maskAnd(smoothMask(n, 1), inner);
        n = removeSmallBits(n, 12);
        n = maskAnd(mirrored(n, sym), inner);
        if (!n.any()) break;
        plan.terraces.push_back(n);
    }

    Mask baseHalf(w, h, 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if ((sym == Sym::MirrorX || sym == Sym::Quad) && x >= w / 2) baseHalf.at(x, y) = 0;
            if ((sym == Sym::MirrorY || sym == Sym::Rot180 || sym == Sym::Quad) && y >= h / 2) baseHalf.at(x, y) = 0;
        }
    for (const CutPlan& s : cuts) {
        // Only what has mountain on BOTH sides counts.
        const int reach = static_cast<int>(s.half) + 4;
        Mask a(w, h), b(w, h);
        for (size_t i = 0; i < a.v.size(); ++i) {
            a.v[i] = m.v[i] && s.d.v[i] < 0 && s.d.v[i] > -s.half - 8;
            b.v[i] = m.v[i] && s.d.v[i] > 0 && s.d.v[i] < s.half + 8;
        }
        const Mask between = maskAnd(dilate4(a, reach), dilate4(b, reach));
        Mask sole(w, h), area(w, h);
        for (size_t i = 0; i < sole.v.size(); ++i) {
            sole.v[i] = std::abs(s.d.v[i]) < (s.valley ? 9 : 3) && between.v[i] && baseHalf.v[i];
            area.v[i] = std::abs(s.d.v[i]) <= s.half + 2;
        }
        // With symmetry the cut counts in the base half; its area is mirrored.
        area = mirrored(area, sym);
        Cut e;
        if (!buildCut(e, sole, area, s.vv, baseHalf)) continue;
        (s.valley ? plan.valleys : plan.gorges).push_back(std::move(e));
    }
}

/// _plane_hochebenen: plateaus apart from mountains, valleys and blocked cells.
void planPlateaus(MountainPlan& plan, const Params& p, Rng& rng, const Mask& freeCells) {
    const int w = freeCells.w, h = freeCells.h, R = kWorldMargin;
    Mask blocked = dilate4(freeCells, 7);
    if (!plan.terraces.empty()) orWith(blocked, dilate4(plan.terraces[0], 9));
    for (const auto* list : {&plan.valleys, &plan.gorges})
        for (const Cut& e : *list) orWith(blocked, maskAnd(e.area, dilate4(e.sole, 4)));
    const int small = std::min(w, h) - 2 * R;
    for (int t = 0; t < 400; ++t) {
        if (static_cast<int>(plan.plateaus.size()) >= p.plateaus) break;
        const int r = small >= 128 ? static_cast<int>(rng.integers(13, 18)) : static_cast<int>(rng.integers(9, 12));
        // May be cut by the border, but most of it lies on the map.
        const int x = static_cast<int>(rng.integers(R + r / 4, w - R - r / 4));
        const int y = static_cast<int>(rng.integers(R + r / 4, h - R - r / 4));
        Mask m = blob(w, h, {x, y}, r, rng, .7);
        m = tidy(mirrored(m, p.symmetry), 60);
        int inside = 0;
        for (int yy = R; yy < h - R; ++yy)
            for (int xx = R; xx < w - R; ++xx) inside += m.at(xx, yy) != 0;
        // Mirrored areas must not touch across the axis.
        if (inside < .55 * double(m.count()) || inside < 60 || anyAnd(m, blocked)) continue;
        orWith(blocked, dilate4(m, 9));
        plan.plateaus.push_back(std::move(m));
    }
}

/// BergPlan.ausschneiden: the map's cutout of the world plan.
MountainPlan cutOut(const MountainPlan& world, int w, int h) {
    const int R = kWorldMargin;
    MountainPlan out;
    for (const Mask& m : world.terraces) {
        const Mask part = crop(m, R, R, w, h);
        // Slivers left by the cut are no mountain.
        Mask small = maskAnd(removeSmallBits(part, out.terraces.empty() ? 90 : 12), part);
        if (!out.terraces.empty()) andWith(small, out.terraces.back());
        if (!small.any()) break;
        out.terraces.push_back(std::move(small));
    }
    for (const Mask& m : world.plateaus) {
        Mask part = crop(m, R, R, w, h);
        if (part.any()) out.plateaus.push_back(std::move(part));
    }
    for (const auto& [src, dst] : {std::pair{&world.valleys, &out.valleys}, std::pair{&world.gorges, &out.gorges}})
        for (const Cut& e : *src) {
            // The ends lie in the base half.
            const Mask base = crop(dilate4(e.sole, 8), R, R, w, h);
            Cut c;
            if (buildCut(c, crop(e.sole, R, R, w, h), crop(e.area, R, R, w, h), cropGrid(e.along, R, R, w, h), base))
                dst->push_back(std::move(c));
        }
    return out;
}

}  // namespace

Mask MountainPlan::all(int w, int h) const {
    Mask m(w, h, 0);
    if (!terraces.empty()) orWith(m, terraces[0]);
    for (const Mask& p : plateaus) orWith(m, p);
    return m;
}

Mask MountainPlan::seam(int w, int h) const {
    // Mountains entirely, plateaus along the edge only; their top keeps its own ground.
    Mask m(w, h, 0);
    if (!terraces.empty()) orWith(m, widen(terraces[0], 5));
    for (const Mask& p : plateaus) orWith(m, maskAnd(widen(p, 5), maskNot(shrink(p, 6, 6))));
    return m;
}

MountainPlan planMountains(const Params& p, const Mask& blocked) {
    MountainPlan plan;
    if (!mountainsWanted(p)) return plan;
    const int R = kWorldMargin, h = p.height + 2 * R, w = p.width + 2 * R;
    // Own randomness: without mountains every map stays bit-identical.
    Rng rng(p.seed ^ 0x6265726765ull);   // "berge"
    Mask freeCells(w, h, 0);
    if (blocked.w == p.width && blocked.h == p.height) paste(freeCells, blocked, R, R);
    freeCells = mirrored(freeCells, p.symmetry);
    if (p.mountains > 0) planRanges(plan, p, rng, freeCells);
    planPlateaus(plan, p, rng, freeCells);
    return cutOut(plan, p.width, p.height);
}

// ---------------------------------------------------------------------------
// Walls

namespace {

/// _boegen: split the boundary cycle into arcs with randomly placed gaps. No
/// wall along the map border; the last border point stays so walls reach it.
std::vector<Path> arcs(const Path& cycle, int w, int h, int count, int lenLo, int lenHi, Rng& rng) {
    const int n = static_cast<int>(cycle.size());
    std::vector<uint8_t> border(static_cast<size_t>(n)), drop(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto [x, y] = cycle[size_t(i)];
        border[size_t(i)] = x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1;
    }
    for (int i = 0; i < n; ++i)
        drop[size_t(i)] = border[size_t(i)] && border[size_t((i + n - 1) % n)] && border[size_t((i + 1) % n)];
    const int start = static_cast<int>(rng.integers(0, n));
    for (int i = 0; i < count; ++i) {
        const int pos = start + static_cast<int>((i + rng.uniform(-.25, .25)) * n / count);
        const int len = static_cast<int>(rng.integers(lenLo, lenHi + 1));
        for (int q = pos; q < pos + len; ++q) drop[size_t(((q % n) + n) % n)] = 1;
    }
    int s = -1;
    for (int i = 0; i < n && s < 0; ++i) if (drop[size_t(i)]) s = i;
    if (s < 0) {
        Path closed = cycle;
        closed.push_back(cycle.front());
        return {closed};
    }
    std::vector<Path> out;
    Path run;
    for (int i = 0; i <= n; ++i) {
        const int j = (s + i) % n;
        if (i == n || drop[size_t(j)]) {
            if (run.size() >= 8) out.push_back(run);
            run.clear();
        } else {
            run.push_back(cycle[size_t(j)]);
        }
    }
    return out;
}

/// _wellig: deflect the contour sideways, otherwise it runs straight between corners.
Path wavy(const Path& b, Rng& rng, int w, int h, double lift = 1.3) {
    const bool closed = b.front() == b.back();
    const int n = static_cast<int>(b.size()) - (closed ? 1 : 0);
    if (n < 14) return b;
    const int k = std::max(3, n / 5);
    std::vector<double> st(static_cast<size_t>(k));
    for (double& v : st) v = rng.uniform(-1, 1);
    if (!closed) st.front() = st.back() = 0;
    Path out;
    for (int i = 0; i < n; ++i) {
        const double x = closed ? double(i) / n * k : double(i) / (n - 1) * (k - 1);
        const int ii = static_cast<int>(x);
        double f = x - ii;
        f = f * f * (3 - 2 * f);
        const double off = (st[size_t(ii % k)] * (1 - f) + st[size_t((ii + 1) % k)] * f) * lift;
        double tx, ty;
        if (closed) {
            const Point& a = b[size_t((i + 3) % n)];
            const Point& c = b[size_t((i - 3 + n) % n)];
            tx = a.first - c.first;
            ty = a.second - c.second;
        } else if (i == 0) {
            tx = b[1].first - b[0].first;
            ty = b[1].second - b[0].second;
        } else if (i == n - 1) {
            tx = b[size_t(n - 1)].first - b[size_t(n - 2)].first;
            ty = b[size_t(n - 1)].second - b[size_t(n - 2)].second;
        } else {
            tx = (b[size_t(i + 1)].first - b[size_t(i - 1)].first) / 2.0;
            ty = (b[size_t(i + 1)].second - b[size_t(i - 1)].second) / 2.0;
        }
        const double ln = std::max(std::hypot(tx, ty), 1e-6);
        const double qx = b[size_t(i)].first + (-ty / ln) * off;
        const double qy = b[size_t(i)].second + (tx / ln) * off;
        out.emplace_back(roundEven(std::min(std::max(qx, 0.0), double(w - 1))),
                         roundEven(std::min(std::max(qy, 0.0), double(h - 1))));
    }
    if (closed) out.push_back(out.front());
    return out;
}

/// _weich: moving average over 2r+1 points.
Path soften(const Path& b, int r) {
    const bool closed = b.front() == b.back();
    const int n = static_cast<int>(b.size()) - (closed ? 1 : 0);
    Path out;
    for (int i = 0; i < n; ++i) {
        double sx = 0, sy = 0;
        for (int o = -r; o <= r; ++o) {
            int j = i + o;
            j = closed ? ((j % n) + n) % n : std::clamp(j, 0, n - 1);
            sx += b[size_t(j)].first;
            sy += b[size_t(j)].second;
        }
        out.emplace_back(roundEven(sx / (2 * r + 1)), roundEven(sy / (2 * r + 1)));
    }
    if (closed) out.push_back(out.front());
    return out;
}

/// _linie: simplified polyline without repeated points.
Path line(const Path& b, double eps) {
    Path pt;
    if (b.front() == b.back()) {
        const size_t m = b.size() / 2;
        pt = simplifyContour(Path(b.begin(), b.begin() + long(m) + 1), eps);
        pt.pop_back();
        const Path rest = simplifyContour(Path(b.begin() + long(m), b.end()), eps);
        pt.insert(pt.end(), rest.begin(), rest.end());
    } else {
        pt = simplifyContour(b, eps);
    }
    Path out;
    for (const Point& q : pt)
        if (out.empty() || out.back() != q) out.push_back(q);
    return out;
}

/// _ziehe: rasterise a polyline into the mask.
void rasterise(Mask& m, const Path& pts) {
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        const auto [x0, y0] = pts[i];
        const auto [x1, y1] = pts[i + 1];
        const int n = std::max({std::abs(x1 - x0), std::abs(y1 - y0), 1});
        for (int s = 0; s <= n; ++s) {
            const double t = double(s) / n;
            m.at(roundEven(x0 + (x1 - x0) * t), roundEven(y0 + (y1 - y0) * t)) = 1;
        }
    }
}

/// _fluten: flood fill inside `open` from `start`, with or without diagonals.
Mask flood(const Mask& open, const Mask& start, bool diagonal) {
    const int w = open.w, h = open.h;
    Mask r(w, h, 0);
    std::vector<int> stack;
    for (size_t i = 0; i < r.v.size(); ++i)
        if (start.v[i] && open.v[i]) { r.v[i] = 1; stack.push_back(static_cast<int>(i)); }
    static const int k8[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
    while (!stack.empty()) {
        const int c = stack.back();
        stack.pop_back();
        const int x = c % w, y = c / w;
        for (int k = 0; k < (diagonal ? 8 : 4); ++k) {
            const int nx = x + k8[k][0], ny = y + k8[k][1];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const int nc = ny * w + nx;
            if (open.v[size_t(nc)] && !r.v[size_t(nc)]) { r.v[size_t(nc)] = 1; stack.push_back(nc); }
        }
    }
    return r;
}

/// One window of the map: tiles, cell types, wall cell types (-1 = not set) and wall mask.
struct Window {
    int x0 = 0, y0 = 0;
    Grid<int32_t> tiles, cells;
    Grid<int16_t> walls;
    Mask mask;
};

/// _kern_erreichbar: can the inside of the area be reached from outside (diagonal steps count)?
bool coreReachable(const Window& win, const Mask& comp) {
    const int w = comp.w, h = comp.h;
    const Mask near6 = dilate4(comp, 6), near4 = dilate4(comp, 4);
    Grid<uint8_t> ct(w, h);
    for (size_t i = 0; i < ct.v.size(); ++i) ct.v[i] = static_cast<uint8_t>(win.cells.v[i] & 31);
    const Mask passable = passableMask(ct);
    Mask open(w, h), outside(w, h);
    for (size_t i = 0; i < open.v.size(); ++i) {
        open.v[i] = passable.v[i] && near6.v[i];
        outside.v[i] = near6.v[i] && !near4.v[i];
    }
    return anyAnd(flood(open, outside, true), erode(comp, 3));
}

struct Drawer {
    WallCanvas& c;
    Rng rng;
    std::vector<uint16_t> cellsBuf;

    /// Bounding box of the component plus 10 cells (_ausschnitt).
    Window window(const Mask& comp) const {
        int x0 = comp.w, y0 = comp.h, x1 = -1, y1 = -1;
        for (int y = 0; y < comp.h; ++y)
            for (int x = 0; x < comp.w; ++x)
                if (comp.at(x, y)) { x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y); }
        Window win;
        win.x0 = std::max(0, x0 - 10);
        win.y0 = std::max(0, y0 - 10);
        const int ww = std::min(comp.w, x1 + 11) - win.x0, wh = std::min(comp.h, y1 + 11) - win.y0;
        win.tiles = cropGrid(c.tiles, win.x0, win.y0, ww, wh);
        win.cells = cropGrid(c.cells, win.x0, win.y0, ww, wh);
        return win;
    }

    /// Zeichner.wand: draws one wall into the window; returns the wall mask, sets `rough` if the
    /// solver only found the fallback.
    Mask wall(Window& win, Grid<int16_t>& walls, const Path& pts, int side, uint64_t seed, double height, bool& rough) {
        const int w = win.tiles.w, h = win.tiles.h;
        cellsBuf.resize(win.tiles.v.size());
        for (size_t i = 0; i < cellsBuf.size(); ++i) cellsBuf[i] = static_cast<uint16_t>(win.tiles.v[i]);
        BlendMap bm(c.profile, w, h, c.table.data(), static_cast<int>(c.table.size()), cellsBuf.data());
        std::vector<int32_t> xy;
        for (const auto& [x, y] : pts) { xy.push_back(x); xy.push_back(y); }
        Mask m(w, h, 0);
        bm.drawCliff(xy.data(), static_cast<int>(pts.size()), side, height, seed);
        for (const om_change& ch : bm.changes()) {
            win.tiles.at(ch.x, ch.y) = ch.new_mapping;
            int ct = ch.cell_type;
            if (ct >= kNorthCliffs && ct <= kCliffsLowSide) m.at(ch.x, ch.y) = 1;
            if (ct >= 0) walls.at(ch.x, ch.y) = static_cast<int16_t>(ct);
            if (ct < 0 && ch.new_mapping >= 0 && ch.new_mapping < static_cast<int>(c.cellOfMapping.size()))
                ct = c.cellOfMapping[size_t(ch.new_mapping)];
            win.cells.at(ch.x, ch.y) = ct >= 0 ? ct : 4;
        }
        rough = bm.lastCliffStage() < 0;
        return m;
    }

    /// _fremde_paare: tile pairs at the wall no original map has.
    int foreignPairs(const Grid<int32_t>& g, const Mask& band) const {
        const GeneratorData& gd = c.profile.generator;
        int s = 0;
        for (int y = 0; y < g.h; ++y)
            for (int x = 0; x < g.w; ++x) {
                const uint32_t a = c.packedOfMapping[size_t(g.at(x, y))];
                if (x + 1 < g.w && (band.at(x, y) || band.at(x + 1, y))) {
                    const uint32_t b = c.packedOfMapping[size_t(g.at(x + 1, y))];
                    s += !gd.known(true, a) || !gd.seen(true, a, b);
                }
                if (y + 1 < g.h && (band.at(x, y) || band.at(x, y + 1))) {
                    const uint32_t b = c.packedOfMapping[size_t(g.at(x, y + 1))];
                    s += !gd.known(false, a) || !gd.seen(false, a, b);
                }
            }
        return s;
    }

    enum class Goal { None, Closed, Open };

    /// _setze: encloses one area with a wall; the best of several attempts wins.
    bool enclose(Grid<int16_t>& walls, Mask& mask, const Mask& compFull, int gaps, int lenLo, int lenHi, Goal goal) {
        Window base = window(compFull);
        const int w = base.tiles.w, h = base.tiles.h;
        const Mask comp = crop(compFull, base.x0, base.y0, w, h);
        const Path cycle = boundaryCycle(comp);
        if (cycle.size() < 9) return true;
        const Grid<int16_t> wallsBase = cropGrid(walls, base.x0, base.y0, w, h);
        const Mask maskBase = crop(mask, base.x0, base.y0, w, h);
        const Mask near5 = dilate4(comp, 5), core5 = erode(comp, 5);
        bool have = false, bestBad = true;
        double bestNote = 0;
        Window best;
        const int rounds = kAttempts * (goal != Goal::None ? 3 : 1);
        for (int v = 0; v < rounds; ++v) {
            // One more gap only if the requested number does not work.
            const int more = goal == Goal::Open ? v / kAttempts : 0;
            const std::vector<Path> bows = arcs(cycle, w, h, gaps + more, lenLo, lenHi, rng);
            Window win = base;
            win.walls = wallsBase;
            win.mask = maskBase;
            int over = 0, rough = 0;
            Mask target(w, h, 0);
            for (Path b : bows) {
                if (v % 3) b = wavy(b, rng, w, h);
                const Path pt = line(soften(b, 1 + v % 2), 1.0 + .25 * (v % 4));
                if (pt.size() < 2) continue;
                // Late attempts with a thicker wall, which closes gaps.
                const double height = goal == Goal::Closed && v >= 2 * kAttempts ? 4.0 : 3.0;
                const int side = contourHighLeft(pt, comp) ? OM_CLIFF_HIGH_LEFT : OM_CLIFF_HIGH_RIGHT;
                const uint64_t seed = static_cast<uint64_t>(rng.integers(0, int64_t(1) << 31));
                bool r = false;
                const Mask m = wall(win, win.walls, pt, side, seed, height, r);
                rough += r;
                rasterise(target, pt);
                over += countAnd(m, dilate4(win.mask, 1));
                orWith(win.mask, m);
            }
            bool good = true;
            if (goal != Goal::None) {
                const bool reach = coreReachable(win, comp);
                good = goal == Goal::Open ? reach : !reach;
            }
            // Rated per wall cell; as a sum the shortest wall would win. And where
            // the solver drew nothing, wall is missing.
            Mask band = maskAnd(maskAnd(near5, maskNot(core5)), win.mask);
            const int bandCells = std::max(1, static_cast<int>(band.count()));
            const Mask drawn = dilate4(win.mask, 1);
            int missing = 0;
            for (size_t i = 0; i < target.v.size(); ++i) missing += target.v[i] && !drawn.v[i];
            const double miss = double(missing) / std::max(1, static_cast<int>(target.count()));
            const double note = (3.0 * over + foreignPairs(win.tiles, band)) / bandCells + 4 * miss + .5 * rough;
            const bool bad = !good;
            if (!have || bad < bestBad || (bad == bestBad && note < bestNote)) {
                have = true;
                bestBad = bad;
                bestNote = note;
                best = std::move(win);
            }
            if (!bestBad && v >= kAttempts - 1) break;
        }
        paste(c.tiles, best.tiles, base.x0, base.y0);
        paste(c.cells, best.cells, base.x0, base.y0);
        paste(walls, best.walls, base.x0, base.y0);
        paste(mask, best.mask, base.x0, base.y0);
        return !bestBad;
    }

    /// _ein_berg: one mountain from the inside out; the outer wall comes last. -> closed?
    bool mountain(Grid<int16_t>& walls, Mask& mask, const Mask& range, const MountainPlan& plan, double loose) {
        for (size_t k = plan.terraces.size() - 1; k >= 1; --k) {
            for (const auto& idx : components(maskAnd(plan.terraces[k], range))) {
                const Mask comp = fromIndices(idx, range.w, range.h);
                const int n = static_cast<int>(boundaryCycle(comp).size());
                const int gaps = n < 60 ? 1 : roundEven(n / (60 - 25 * loose));
                enclose(walls, mask, comp, gaps, 4, 8 + static_cast<int>(6 * loose), Goal::None);
            }
        }
        // Without a core there is nothing to leak into.
        const bool core = erode(range, 3).any();
        return enclose(walls, mask, range, 0, 0, 0, core ? Goal::Closed : Goal::None);
    }
};

}  // namespace

Mask drawMountains(WallCanvas& c, const MountainPlan& plan, const Params& p, MountainReport& report) {
    const int w = c.tiles.w, h = c.tiles.h;
    Mask mask(w, h, 0);
    Grid<int16_t> walls(w, h, -1);
    Drawer d{c, Rng(p.seed ^ 0x7761656e6465ull), {}};   // "waende"
    const double loose = 1 - std::clamp(p.mountainDensity, 0.0, 1.0);
    std::vector<Mask> ranges;
    if (!plan.terraces.empty())
        for (const auto& idx : components(plan.terraces[0])) ranges.push_back(fromIndices(idx, w, h));
    for (const Mask& range : ranges) {
        // Each mountain repeated until it is closed, at most kRuns times.
        const Window area = d.window(range);
        const int x0 = area.x0, y0 = area.y0, ww = area.tiles.w, wh = area.tiles.h;
        const auto tiles0 = area.tiles, cells0 = area.cells;
        const auto walls0 = cropGrid(walls, x0, y0, ww, wh);
        const auto mask0 = crop(mask, x0, y0, ww, wh);
        bool have = false;
        Grid<int32_t> bt, bc;
        Grid<int16_t> bw;
        Mask bm;
        for (int run = 0; run < kRuns; ++run) {
            paste(c.tiles, tiles0, x0, y0);
            paste(c.cells, cells0, x0, y0);
            paste(walls, walls0, x0, y0);
            paste(mask, mask0, x0, y0);
            const bool closed = d.mountain(walls, mask, range, plan, loose);
            if (!have || closed) {
                have = true;
                bt = cropGrid(c.tiles, x0, y0, ww, wh);
                bc = cropGrid(c.cells, x0, y0, ww, wh);
                bw = cropGrid(walls, x0, y0, ww, wh);
                bm = crop(mask, x0, y0, ww, wh);
            }
            if (closed) break;
        }
        paste(c.tiles, bt, x0, y0);
        paste(c.cells, bc, x0, y0);
        paste(walls, bw, x0, y0);
        paste(mask, bm, x0, y0);
    }
    std::vector<Mask> plateaus;
    for (const Mask& m : plan.plateaus)
        for (const auto& idx : components(m)) plateaus.push_back(fromIndices(idx, w, h));
    for (const Mask& comp : plateaus) d.enclose(walls, mask, comp, 2, 8, 11, Drawer::Goal::Open);

    auto inside = [&](const Mask& comp) {
        Window win = d.window(comp);
        return coreReachable(win, crop(comp, win.x0, win.y0, win.tiles.w, win.tiles.h));
    };
    report.mountains = static_cast<int>(ranges.size());
    report.plateaus = static_cast<int>(plateaus.size());
    for (const Mask& r : ranges) report.leaky += inside(r);
    for (const Mask& r : plateaus) report.unreachable += !inside(r);
    return mask;
}

Mask buildingGround(const Grid<uint8_t>& cellTypes) {
    const Mask area = largestArea(passableMask(cellTypes));
    const int w = area.w, h = area.h, n = kBuildSquare;
    std::vector<int64_t> s(size_t(w + 1) * (h + 1), 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            s[size_t(y + 1) * (w + 1) + x + 1] = area.at(x, y) + s[size_t(y) * (w + 1) + x + 1] +
                                                 s[size_t(y + 1) * (w + 1) + x] - s[size_t(y) * (w + 1) + x];
    Mask out(w, h, 0);
    for (int y = 0; y + n <= h; ++y)
        for (int x = 0; x + n <= w; ++x) {
            const int64_t sum = s[size_t(y + n) * (w + 1) + x + n] - s[size_t(y) * (w + 1) + x + n] -
                                s[size_t(y + n) * (w + 1) + x] + s[size_t(y) * (w + 1) + x];
            if (sum != n * n) continue;
            for (int dy = 0; dy < n; ++dy)
                for (int dx = 0; dx < n; ++dx) out.at(x + dx, y + dy) = 1;
        }
    return out;
}

void zoneReport(const Grid<uint8_t>& cellTypes, const MountainPlan& plan, MountainReport& report) {
    const Mask build = buildingGround(cellTypes);
    const Mask open = passableMask(cellTypes);
    report.build = double(build.count()) / double(build.v.size());
    auto share = [&](const std::vector<const Mask*>& masks) {
        if (masks.empty()) return -1.0;
        Mask m(build.w, build.h, 0);
        for (const Mask* x : masks) orWith(m, *x);
        return double(countAnd(build, m)) / std::max<double>(1, double(m.count()));
    };
    std::vector<const Mask*> mountain, plateau, valley, gorge;
    if (!plan.terraces.empty()) mountain.push_back(&plan.terraces[0]);
    for (const Mask& p : plan.plateaus) plateau.push_back(&p);
    for (const Cut& e : plan.valleys) valley.push_back(&e.sole);
    for (const Cut& e : plan.gorges) gorge.push_back(&e.sole);
    report.buildMountain = share(mountain);
    report.buildPlateau = share(plateau);
    report.buildValley = share(valley);
    report.buildGorge = share(gorge);
    report.gorges = static_cast<int>(plan.gorges.size());
    report.gorgesPassable = 0;
    // Passage is tested without diagonals, i.e. strictly.
    for (const Cut& e : plan.gorges)
        report.gorgesPassable += anyAnd(flood(maskAnd(open, e.area), e.endA, false), e.endB);
}

}  // namespace om::gen
