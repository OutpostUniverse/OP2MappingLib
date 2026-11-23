// Relief: height field, landscape mode, contours and features.
//
// Port of op2-codegen-editor's mapview/mapgen_relief.py. The height field H
// marks highland (1) and lowland (0); features (mountain, valley, ...) add
// deliberate shapes, the cliff slider scatters extra blobs. Landscape mode
// builds one massif at a map side plus mesas, cut by dry riverbeds. The
// contour helpers turn a highland component into the polylines the cliff
// drawing walks along.
//
// Differences from the Python original:
//   - Random numbers come from om::Rng, so shapes match in kind only.
//   - Python's round() rounds half to even; std::nearbyint does the same in
//     the default rounding mode and is used wherever the original rounds.
//   - landscapeFamilies / steerFeatureFamilies take the mud and sand family
//     indices instead of the TileModel (that is all they read from it).
#include "mapgen.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace om::gen {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Python round(): half to even.
int pyRound(double v) { return static_cast<int>(std::nearbyint(v)); }

// Python a // b for ints (floor division).
int floorDiv(int a, int b) {
    int q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

// Python a % n, never negative.
int pyMod(int a, int n) { return ((a % n) + n) % n; }

int sign(int v) { return (v > 0) - (v < 0); }

// a |= b, cell by cell.
void unite(Mask& a, const Mask& b) {
    for (size_t i = 0; i < a.v.size(); ++i)
        if (b.v[i]) a.v[i] = 1;
}

// SEKTOREN: 3x3 grid of rough positions.
struct Sector {
    const char* name;
    double fx, fy;
};
const Sector kSectors[] = {
    {"nw", 0.22, 0.22}, {"n", 0.50, 0.20}, {"no", 0.78, 0.22},
    {"w", 0.20, 0.50},  {"c", 0.50, 0.50}, {"o", 0.80, 0.50},
    {"sw", 0.22, 0.78}, {"s", 0.50, 0.80}, {"so", 0.78, 0.78},
};

const Sector* findSector(const std::string& where) {
    for (const Sector& s : kSectors)
        if (where == s.name) return &s;
    return nullptr;
}

// GEGENUEBER: opposite sector, "" if `where` is no sector.
std::string opposite(const std::string& where) {
    static const char* const pairs[][2] = {
        {"nw", "so"}, {"n", "s"}, {"no", "sw"}, {"w", "o"}, {"c", "so"},
        {"o", "w"},   {"sw", "no"}, {"s", "n"}, {"so", "nw"},
    };
    for (const auto& p : pairs)
        if (where == p[0]) return p[1];
    return "";
}

// _MOORE: Moore neighbourhood clockwise, starting west.
const int kMoore[8][2] = {{-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}};

int mooreIndex(int dx, int dy) {
    for (int i = 0; i < 8; ++i)
        if (kMoore[i][0] == dx && kMoore[i][1] == dy) return i;
    return -1;
}

// Mask of one component given as cell indices.
Mask componentMask(const std::vector<int>& cells, int w, int h) {
    Mask m(w, h, 0);
    for (int i : cells) m.v[size_t(i)] = 1;
    return m;
}

// np.unique(values)[argmax(counts)]: most frequent family, ties to the smallest.
int majority(const Field& f, const Mask& m) {
    std::vector<int> counts;
    for (size_t i = 0; i < f.v.size(); ++i) {
        if (!m.v[i]) continue;
        const int v = f.v[i];
        if (v < 0) continue;
        if (size_t(v) >= counts.size()) counts.resize(size_t(v) + 1, 0);
        ++counts[size_t(v)];
    }
    int best = 0;
    for (size_t k = 1; k < counts.size(); ++k)
        if (counts[k] > counts[size_t(best)]) best = int(k);
    return best;
}

}  // namespace

// _hoehenfeld: where the highland goes, from the features and the cliff slider.
Mask heightField(const Params& p, Rng& rng, const std::vector<Placement>& layout) {
    const int w = p.width, h = p.height;
    Mask H(w, h, 0);
    for (const Placement& e : layout) {
        switch (e.feat.kind) {
        case FeatKind::Mountain:
            unite(H, blob(w, h, {e.x, e.y}, e.radius, rng, 0.5));
            break;
        case FeatKind::Plateau:
            // Flatter and rounder than a mountain, so you can build on top.
            unite(H, blob(w, h, {e.x, e.y}, int(e.radius * 1.3), rng, 0.25));
            break;
        case FeatKind::Valley:
            unite(H, corridor(w, h, e.path, e.width, rng, 0.35));
            break;
        case FeatKind::Choke: {
            const int n = int(e.path.size());
            const int mid = n / 2 + int(rng.integers(floorDiv(-n, 6), n / 6 + 1));
            unite(H, barrier(w, h, e.path, mid, std::max(2, e.width / 2), rng));
            break;
        }
        case FeatKind::Clearing:
            break;
        }
    }
    const int blobs = pyRound(p.cliffs * w * h / 900.0);
    for (int i = 0; i < blobs; ++i) {
        const int cx = int(rng.integers(6, std::max(7, w - 6)));
        const int cy = int(rng.integers(6, std::max(7, h - 6)));
        const int r = int(rng.integers(6, 16));
        unite(H, blob(w, h, {cx, cy}, r, rng, 0.55));
    }
    // Punch out clearings, with a margin: the wall band is three cells thick
    // and its foot would otherwise still lie inside the clearing.
    for (const Placement& e : layout) {
        if (e.feat.kind != FeatKind::Clearing) continue;
        const Mask m = featureMask(e, w, h, 4);
        for (size_t i = 0; i < H.v.size(); ++i)
            if (m.v[i]) H.v[i] = 0;
    }
    if (H.any()) {
        // Snap the edge, otherwise the cliff frays cell by cell.
        H = snapEdge(H, 3);
        H = removeSmallBits(H, 24);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                if (y < 3 || y >= h - 3 || x < 3 || x >= w - 3) H.at(x, y) = 0;
    }
    return H;
}

// _fluss_pfad: meandering path from start towards goal - heading plus a slow sideways wave.
Path riverPath(Point start, Point goal, int w, int h, Rng& rng) {
    double x = start.first, y = start.second;
    const double tx = goal.first, ty = goal.second;
    Path pts{{pyRound(x), pyRound(y)}};
    const double phase = rng.uniform(0.0, 2 * kPi);
    const double pace = rng.uniform(0.05, 0.09);
    const double swing = rng.uniform(0.5, 0.85);
    for (int step = 0; step < 4 * (w + h); ++step) {
        const double dx = tx - x, dy = ty - y;
        if (std::hypot(dx, dy) < 2.0) break;
        double angle = std::atan2(dy, dx);
        angle += swing * std::sin(phase + step * pace);
        x = std::min(std::max(x + std::cos(angle), 1.0), w - 2.0);
        y = std::min(std::max(y + std::sin(angle), 1.0), h - 2.0);
        const Point q{pyRound(x), pyRound(y)};
        if (q != pts.back()) pts.push_back(q);
        if (q.first <= 1 || q.first >= w - 2 || q.second <= 1 || q.second >= h - 2) break;
    }
    return pts;
}

// _pfad_band: cells within `radius` of any path point (the bed or the gorge).
Mask pathBand(int w, int h, const Path& pts, double radius) {
    Mask band(w, h, 0);
    const int r = int(std::ceil(radius));
    const double r2 = radius * radius;
    for (const Point& pt : pts) {
        const int px = pt.first, py = pt.second;
        const int y0 = std::max(0, py - r), y1 = std::min(h, py + r + 1);
        const int x0 = std::max(0, px - r), x1 = std::min(w, px + r + 1);
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const int dx = x - px, dy = y - py;
                if (double(dx * dx + dy * dy) <= r2) band.at(x, y) = 1;
            }
    }
    return band;
}

// _dehne: 4-neighbour dilation by `steps` cells, nothing wraps around the edge.
Mask dilate4(const Mask& m, int steps) {
    Mask out = m;
    const int w = m.w, h = m.h;
    for (int s = 0; s < steps; ++s) {
        Mask grown = out;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                if (grown.at(x, y)) continue;
                if ((y > 0 && out.at(x, y - 1)) || (y < h - 1 && out.at(x, y + 1)) ||
                    (x > 0 && out.at(x - 1, y)) || (x < w - 1 && out.at(x + 1, y)))
                    grown.at(x, y) = 1;
            }
        out = std::move(grown);
    }
    return out;
}

// _landschaft_hoehenfeld: massif at one map side, mesas, riverbeds cutting gorges.
std::pair<Mask, Mask> landscapeHeightField(const Params& p, Rng& rng) {
    const int w = p.width, h = p.height;
    const int area = w * h;
    Mask H(w, h, 0);
    // 0 = west, 1 = east, 2 = north, 3 = south.
    const int side = int(rng.integers(0, 4));
    // Target share of the massif; the cliff slider adds to it.
    const double target = 0.16 + 0.45 * p.cliffs;
    const int baseRadius = pyRound(std::sqrt(double(area)) / 5.5);
    for (int i = 0; i < 10; ++i) {
        if (double(H.count()) / area >= target) break;
        const int r = int(rng.integers(std::max(4, baseRadius - 4), baseRadius + 5));
        const double along = rng.uniform(0.08, 0.92);
        const double depth = rng.uniform(0.0, 0.55) * r;
        int cx, cy;
        if (side == 0) {
            cx = int(depth);
            cy = int(along * h);
        } else if (side == 1) {
            cx = w - 1 - int(depth);
            cy = int(along * h);
        } else if (side == 2) {
            cx = int(along * w);
            cy = int(depth);
        } else {
            cx = int(along * w);
            cy = h - 1 - int(depth);
        }
        unite(H, blob(w, h, {cx, cy}, r, rng, 0.35));
    }
    const Mask massif = H;

    // Mesas: free-standing table mountains in the lowland, clear of the massif.
    int mesas = std::max(1, pyRound(area / 12000.0));
    Mask blocked = dilate4(massif, 8);
    const int tries = mesas * 6;
    for (int t = 0; t < tries; ++t) {
        if (mesas <= 0) break;
        const int r = int(rng.integers(5, 9));
        const int cx = int(rng.integers(r + 4, w - r - 4));
        const int cy = int(rng.integers(r + 4, h - r - 4));
        if (blocked.at(cx, cy)) continue;
        const Mask mesa = blob(w, h, {cx, cy}, r, rng, 0.3);
        unite(H, mesa);
        unite(blocked, dilate4(mesa, 8));
        --mesas;
    }

    // Riverbeds: source inside the massif, mouth at the opposite border. The
    // gorge is wide enough for two wall bands (3 each) plus a walkable floor.
    int rivers = p.rivers;
    if (rivers < 0) rivers = std::max(1, pyRound(std::sqrt(double(area)) / 128.0));
    Mask beds(w, h, 0);
    Mask notMassif(w, h, 0);
    for (size_t i = 0; i < massif.v.size(); ++i) notMassif.v[i] = massif.v[i] ? 0 : 1;
    const Mask nearLow = dilate4(notMassif, 6);
    Mask inner(w, h, 0);
    for (size_t i = 0; i < massif.v.size(); ++i) inner.v[i] = (massif.v[i] && !nearLow.v[i]) ? 1 : 0;
    const Mask& srcMask = inner.any() ? inner : massif;
    std::vector<Point> sources;   // np.nonzero order: row-major
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (srcMask.at(x, y)) sources.emplace_back(x, y);
    for (int k = 0; k < rivers; ++k) {
        if (sources.empty()) break;
        const Point start = sources[size_t(rng.integers(0, int64_t(sources.size())))];
        Point mouth;
        if (side == 0)
            mouth = {w - 2, int(rng.integers(h / 6, h - h / 6))};
        else if (side == 1)
            mouth = {1, int(rng.integers(h / 6, h - h / 6))};
        else if (side == 2)
            mouth = {int(rng.integers(w / 6, w - w / 6)), h - 2};
        else
            mouth = {int(rng.integers(w / 6, w - w / 6)), 1};
        const Path path = riverPath(start, mouth, w, h, rng);
        const Mask gorge = pathBand(w, h, path, 5.0);
        for (size_t i = 0; i < H.v.size(); ++i)
            if (gorge.v[i]) H.v[i] = 0;
        unite(beds, pathBand(w, h, path, 2.1));
        // Occasional widenings: a dry river has scour pools.
        const size_t stride = std::max<size_t>(1, path.size() / 4);
        for (size_t i = stride; i < path.size(); i += stride) {
            if (rng.uniform() < 0.5) unite(beds, pathBand(w, h, {path[i]}, rng.uniform(3.0, 4.0)));
        }
    }

    if (H.any()) {
        // Smooth first, then snap to a five-cell grid: the vertical mud run
        // block is exactly 5 cells tall, so every contour run can be covered
        // with whole blocks.
        H = smoothMask(H, 2);
        H = snapEdge(H, 5);
        H = removeSmallBits(H, std::max(24, pyRound(0.0015 * area)));
        // Only the anchored side may reach the border; the contour is not
        // drawn along it.
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                if ((side != 0 && x < 3) || (side != 1 && x >= w - 3) ||
                    (side != 2 && y < 3) || (side != 3 && y >= h - 3))
                    H.at(x, y) = 0;
            }
    }
    // One cell clear of the wall: the footing belongs to the wall family.
    const Mask wall = dilate4(H, 1);
    for (size_t i = 0; i < beds.v.size(); ++i)
        if (wall.v[i]) beds.v[i] = 0;
    return {H, beds};
}

// _landschaft_familien: each highland component plus a rim of 4 takes its
// surroundings' majority family (mud or sand only, else mud); riverbeds become
// sand. `mud` < 0 means no mud family (nothing happens), `sand` < 0 no sand.
void landscapeFamilies(Field& f, const Mask& H, const Mask& beds, int mud, int sand) {
    if (mud < 0) return;
    for (const std::vector<int>& cells : components(H)) {
        // Rim 4: wall band of 3 plus one cell of footing.
        const Mask rim = dilate4(componentMask(cells, H.w, H.h), 4);
        const int maj = majority(f, rim);
        const int16_t fam = static_cast<int16_t>((maj == mud || maj == sand) ? maj : mud);
        for (size_t i = 0; i < f.v.size(); ++i)
            if (rim.v[i]) f.v[i] = fam;
    }
    if (beds.w == f.w && beds.h == f.h && beds.any() && sand >= 0) {
        for (size_t i = 0; i < f.v.size(); ++i)
            if (beds.v[i]) f.v[i] = static_cast<int16_t>(sand);
    }
}

// _rand_zyklus: ordered boundary cycle of one component (Moore tracing).
Path boundaryCycle(const Mask& comp) {
    const int w = comp.w, h = comp.h;
    int sx = -1, sy = -1;
    for (int y = 0; y < h && sy < 0; ++y)
        for (int x = 0; x < w; ++x)
            if (comp.at(x, y)) {
                sx = x;
                sy = y;
                break;
            }
    if (sy < 0) return {};
    auto in = [&](int x, int y) { return comp.inside(x, y) && comp.at(x, y) != 0; };

    Path cycle{{sx, sy}};
    int bx = sx - 1, by = sy;
    int cx = sx, cy = sy;
    const size_t limit = comp.count() * 8 + 8;
    for (size_t it = 0; it < limit; ++it) {
        const int startI = mooreIndex(bx - cx, by - cy);
        if (startI < 0) return cycle;   // cannot happen: the backtrack is always a neighbour
        bool found = false;
        int nx = 0, ny = 0;
        for (int k = 1; k < 9; ++k) {
            const int* d = kMoore[(startI + k) % 8];
            nx = cx + d[0];
            ny = cy + d[1];
            if (in(nx, ny)) {
                const int* pd = kMoore[(startI + k - 1) % 8];
                bx = cx + pd[0];
                by = cy + pd[1];
                found = true;
                break;
            }
        }
        if (!found) return cycle;   // single cell
        cx = nx;
        cy = ny;
        if (cx == sx && cy == sy) return cycle;
        cycle.emplace_back(cx, cy);
    }
    return cycle;
}

// _kontur_vereinfachen: Ramer-Douglas-Peucker. eps = 1 keeps every true corner.
Path simplifyContour(const Path& pts, double eps) {
    if (pts.size() < 3) return pts;
    const double ax = pts.front().first, ay = pts.front().second;
    const double abx = pts.back().first - ax, aby = pts.back().second - ay;
    double len = std::hypot(abx, aby);
    if (len == 0) len = 1.0;
    size_t best = 0;
    double bestDist = -1;
    for (size_t i = 0; i < pts.size(); ++i) {
        const double d = std::abs(abx * (pts[i].second - ay) - aby * (pts[i].first - ax)) / len;
        if (d > bestDist) {
            bestDist = d;
            best = i;
        }
    }
    if (bestDist <= eps) return {pts.front(), pts.back()};
    Path left = simplifyContour(Path(pts.begin(), pts.begin() + long(best) + 1), eps);
    const Path right = simplifyContour(Path(pts.begin() + long(best), pts.end()), eps);
    left.pop_back();
    left.insert(left.end(), right.begin(), right.end());
    return left;
}

// _kontur_hand: counts component cells 1..3 steps left and right of each
// segment midpoint. Left of (dx, dy) is (dy, -dx), y grows downwards.
bool contourHighLeft(const Path& pts, const Mask& comp) {
    int left = 0, right = 0;
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        const int x1 = pts[i].first, y1 = pts[i].second;
        const int x2 = pts[i + 1].first, y2 = pts[i + 1].second;
        const int dx = sign(x2 - x1), dy = sign(y2 - y1);
        const int mx = floorDiv(x1 + x2, 2), my = floorDiv(y1 + y2, 2);
        for (int k = 1; k <= 3; ++k) {
            const int lx = mx + dy * k, ly = my - dx * k;
            const int rx = mx - dy * k, ry = my + dx * k;
            if (comp.inside(lx, ly) && comp.at(lx, ly)) ++left;
            if (comp.inside(rx, ry) && comp.at(rx, ry)) ++right;
        }
    }
    return left >= right;
}

// _kontur_boegen: drop stretches hugging the map border, cut `gaps` ramp gaps
// (preferably in the middle of horizontal runs, which have end caps) and
// return the remaining open arcs of at least 6 points. With nothing removed
// the closed cycle comes back with its first point appended.
std::vector<Path> contourArcs(const Path& cycle, const Mask& comp, int gaps, Rng& rng, int gapLength) {
    (void)rng;   // unused in the original as well (gap positions are evenly spaced)
    const int w = comp.w, h = comp.h;
    const int n = int(cycle.size());
    if (n == 0) return {};
    std::vector<char> drop(size_t(n), 0);
    for (int i = 0; i < n; ++i) {
        const int x = cycle[size_t(i)].first, y = cycle[size_t(i)].second;
        drop[size_t(i)] = (x <= 1 || y <= 1 || x >= w - 2 || y >= h - 2) ? 1 : 0;
    }
    if (gaps > 0 && n >= 40) {
        std::vector<int> cand;
        for (int i = 0; i < n; ++i) {
            if (drop[size_t(i)]) continue;
            const int y = cycle[size_t(i)].second;
            if (y == cycle[size_t((i + 1) % n)].second && y == cycle[size_t(pyMod(i - 1, n))].second)
                cand.push_back(i);
        }
        if (cand.empty())
            for (int i = 0; i < n; ++i)
                if (!drop[size_t(i)]) cand.push_back(i);
        for (int k = 0; k < gaps; ++k) {
            if (cand.empty()) break;
            const int target = pyRound((k + 0.5) * n / gaps);
            int bestI = cand.front(), bestD = -1;
            for (int j : cand) {
                const int a = std::abs(j - target);
                const int d = std::min(a, n - a);
                if (bestD < 0 || d < bestD) {
                    bestD = d;
                    bestI = j;
                }
            }
            for (int d = -floorDiv(gapLength, 2); d <= floorDiv(gapLength, 2); ++d)
                drop[size_t(pyMod(bestI + d, n))] = 1;
        }
    }
    int offset = -1;
    for (int i = 0; i < n; ++i)
        if (drop[size_t(i)]) {
            offset = i;
            break;
        }
    if (offset < 0) {   // closed figure
        Path closed = cycle;
        closed.push_back(cycle.front());
        return {closed};
    }
    // Start at the first removed point, so no arc wraps around the cycle start.
    std::vector<Path> arcs;
    Path run;
    for (int i = 0; i <= n; ++i) {
        const int j = (offset + i) % n;
        if (i == n || drop[size_t(j)]) {
            if (run.size() >= 6) arcs.push_back(run);
            run.clear();
        } else {
            run.push_back(cycle[size_t(j)]);
        }
    }
    return arcs;
}

// _sektor_punkt: a point in the requested sector, slightly jittered, folded
// into the base half/quarter under symmetry (mirroring copies it over later).
Point sectorPoint(const std::string& where, int w, int h, Rng& rng, Sym s) {
    // Python evaluates the random fallback even for a known sector; drawn
    // always here as well to keep the call sequence the same.
    const double rx = rng.uniform(), ry = rng.uniform();
    double fx, fy;
    if (const Sector* sec = findSector(where)) {
        fx = sec->fx;
        fy = sec->fy;
    } else {
        fx = 0.15 + 0.70 * rx;
        fy = 0.15 + 0.70 * ry;
    }
    const double vx = fx * w + 0.04 * w * rng.normal();
    int x = int(std::min(std::max(vx, 4.0), w - 5.0));
    const double vy = fy * h + 0.04 * h * rng.normal();
    int y = int(std::min(std::max(vy, 4.0), h - 5.0));
    if ((s == Sym::MirrorX || s == Sym::Rot180 || s == Sym::Quad) && x >= w / 2) x = w - 1 - x;
    if ((s == Sym::MirrorY || s == Sym::Quad) && y >= h / 2) y = h - 1 - y;
    return {x, y};
}

// _feature_layout: concrete geometry per feature; runs before everything else.
std::vector<Placement> featureLayout(const Params& p, Rng& rng) {
    const int w = p.width, h = p.height;
    const int small = std::min(w, h);
    std::vector<Placement> layout;
    for (const Feature& feat : p.features) {
        const double size = std::min(std::max(feat.size, 0.0), 1.0);
        const Point at = sectorPoint(feat.where, w, h, rng, p.symmetry);
        Placement e;
        e.feat = feat;
        e.x = at.first;
        e.y = at.second;
        if (feat.kind == FeatKind::Valley || feat.kind == FeatKind::Choke) {
            const std::string to = !feat.to.empty() ? feat.to : opposite(feat.where.empty() ? "c" : feat.where);
            const Point end = sectorPoint(to, w, h, rng, p.symmetry);
            e.path = featurePath(at, end, rng);
            e.width = feat.kind == FeatKind::Valley ? 3 + pyRound(5 * size) : 2 + pyRound(9 * size);
        } else {
            e.radius = pyRound((0.06 + 0.10 * size) * small);
        }
        layout.push_back(std::move(e));
    }
    return layout;
}

// _merkmal_pfad: jittered line from A to B - control points offset across the
// axis, then interpolated cell by cell.
Path featurePath(Point start, Point end, Rng& rng) {
    const int x0 = start.first, y0 = start.second;
    const int x1 = end.first, y1 = end.second;
    const int len = std::max({std::abs(x1 - x0), std::abs(y1 - y0), 1});
    const int steps = std::max(4, len / 8);
    const double dx = double(x1 - x0) / len, dy = double(y1 - y0) / len;
    const double qx = -dy, qy = dx;
    std::vector<std::pair<double, double>> ctrl;
    for (int i = 0; i <= steps; ++i) {
        const double t = double(i) / steps;
        const double off = (i == 0 || i == steps) ? 0.0 : 0.06 * len * rng.normal();
        ctrl.emplace_back(x0 + (x1 - x0) * t + qx * off, y0 + (y1 - y0) * t + qy * off);
    }
    Path out;
    for (size_t i = 0; i + 1 < ctrl.size(); ++i) {
        const double ax = ctrl[i].first, ay = ctrl[i].second;
        const double bx = ctrl[i + 1].first, by = ctrl[i + 1].second;
        const int n = std::max(2, int(std::max(std::abs(bx - ax), std::abs(by - ay))));
        for (int k = 0; k < n; ++k) {
            const double t = double(k) / n;
            out.emplace_back(pyRound(ax + (bx - ax) * t), pyRound(ay + (by - ay) * t));
        }
    }
    out.emplace_back(pyRound(ctrl.back().first), pyRound(ctrl.back().second));
    return out;
}

// _steer_feature_families: ground under each feature (except clearings) becomes
// mud, or sand where the area is mostly sand already, so formations can be
// placed; then one smoothing pass blurs the seam. `mud` < 0 counts as family 0,
// `sand` < 0 as mud (the Python fallbacks for a missing family).
void steerFeatureFamilies(Field& f, const std::vector<Placement>& layout, int mud, int sand) {
    if (layout.empty()) return;
    const int dirt = mud >= 0 ? mud : 0;
    const int sandFam = sand >= 0 ? sand : dirt;
    bool touched = false;
    for (const Placement& e : layout) {
        if (e.feat.kind == FeatKind::Clearing) continue;
        const Mask m = featureMask(e, f.w, f.h, 3);
        size_t total = 0, sandy = 0;
        for (size_t i = 0; i < m.v.size(); ++i)
            if (m.v[i]) {
                ++total;
                if (f.v[i] == sandFam) ++sandy;
            }
        if (total == 0) continue;
        // Keep sand if the area is mostly sand anyway.
        const int16_t fam = static_cast<int16_t>(double(sandy) / double(total) > 0.5 ? sandFam : dirt);
        for (size_t i = 0; i < m.v.size(); ++i)
            if (m.v[i]) f.v[i] = fam;
        touched = true;
    }
    // Blur the edge, otherwise the outline of the steered area stays as a seam.
    // _smooth_families(f, 1): one pass with the Python default radius 2.
    if (touched) f = smoothFamilies(f, 1, 2);
}

// _feature_maske: the area a feature claims, grown by `margin`. Valleys and
// chokes are discs along every second path point (squares would paint a
// hard-edged rectangle into the biome map), the rest one disc.
Mask featureMask(const Placement& e, int w, int h, int margin) {
    Mask m(w, h, 0);
    auto disc = [&](int cx, int cy, int r) {
        const int a = std::abs(r);
        const int y0 = std::max(0, cy - a), y1 = std::min(h - 1, cy + a);
        const int x0 = std::max(0, cx - a), x1 = std::min(w - 1, cx + a);
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) m.at(x, y) = 1;
    };
    if (e.feat.kind == FeatKind::Valley || e.feat.kind == FeatKind::Choke) {
        const int r = e.width + margin + (e.feat.kind == FeatKind::Valley ? 4 : 0);
        for (size_t i = 0; i < e.path.size(); i += 2) disc(e.path[i].first, e.path[i].second, r);
    } else {
        disc(e.x, e.y, e.radius + margin);
    }
    return m;
}

}  // namespace om::gen
