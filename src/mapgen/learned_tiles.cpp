// Learned map generator, part 2: painting the tiles from the sketch and the
// finishing touches (Python: mapgen_netz._Maler, _Nacharbeit, karte).
// Ported from fefi-byte's op2-codegen-editor (MIT, see THIRD-PARTY-NOTICES.md).
#include "learned.hpp"
#include "canvas.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

namespace om::gen {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kWallBoost = 30.0;    // WAND_VERSTAERKUNG: sketch says wall -> prefer wall tiles
constexpr double kNorthBoost = 2.0;    // NORD_VERSTAERKUNG: north edges are one cell thick
constexpr int kSand = 2;
constexpr int kImpassable1 = 6, kImpassable2 = 1;
constexpr int kBuildSquare = 7;

Rng stream(uint64_t seed, uint64_t k) { return Rng(seed * 0x9E3779B97F4A7C15ull ^ (0x6e65747a00000000ull + k)); }

inline bool isCliff(int ct) { return ct == 8 || ct == 9 || ct == 10; }
inline bool isNorth(int v) { return v == 6 || v == 7 || v == 8; }

const double kDirX[8] = {1, 0.70710678118654752440, 0, -0.70710678118654752440, -1, -0.70710678118654752440, 0, 0.70710678118654752440};
const double kDirY[8] = {0, 0.70710678118654752440, 1, 0.70710678118654752440, 0, -0.70710678118654752440, -1, -0.70710678118654752440};

int fallDirection(double tx, double ty) {
    int best = 0;
    double bv = -1e300;
    for (int r = 0; r < 8; ++r) {
        const double v = kDirX[r] * -ty + kDirY[r] * tx;
        if (v > bv) { bv = v; best = r; }
    }
    return best;
}

template <class T>
void shuffle(std::vector<T>& v, Rng& rng) {
    for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[size_t(rng.integers(0, int64_t(i)))]);
}

/// Gumbel(0, 1) draw.
double gumbel(Rng& rng) {
    double u = rng.uniform();
    if (u <= 0) u = 1e-300;
    return -nn::detLog(-nn::detLog(u));
}

/// _Maler: paints tiles after the sketch. Grid values are vocabulary indices;
/// V + 1 marks an open cell.
struct Painter {
    const LearnedModel& m;
    nn::Pool& pool;
    const ProgressFn& progress;
    double done = 0;
    int h, w, V, open;
    Grid<int32_t> sw, sb;     ///< sketch per cell (full resolution)
    nn::Act pos, sketchFeat;
    Mask nah, nord, must;

    Painter(const LearnedModel& model, const Sketch& sk, nn::Pool& p, const ProgressFn& pr)
        : m(model), pool(p), progress(pr), h(sk.wall.h * 2), w(sk.wall.w * 2), V(model.vocab), open(model.vocab + 1) {
        sw = Grid<int32_t>(w, h);
        sb = Grid<int32_t>(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                sw.at(x, y) = sk.wall.at(x / 2, y / 2);
                sb.at(x, y) = sk.ground.at(x / 2, y / 2);
            }
        pos = nn::borderPlanes(h, w, 12.f);
        sketchFeat = m.tiles.sketchFeatures(h, w, sw.v.data(), sb.v.data(), pool);
        masks();
    }

    void check() const {
        if (progress && !progress(done)) throw Cancelled();
    }

    void masks() {
        Mask any(w, h);
        nord = Mask(w, h);
        must = Mask(w, h);
        for (size_t i = 0; i < sw.v.size(); ++i) {
            any.v[i] = sw.v[i] > 0;
            nord.v[i] = isNorth(sw.v[i]);
            must.v[i] = any.v[i] && !nord.v[i];
        }
        nah = dilate4(any, 3);
    }

    /// The sketch changed inside [x0, x1] x [y0, y1]: recompute its features there.
    void sketchChanged(int x0, int y0, int x1, int y1) {
        const int cx0 = std::max(0, x0 - 4), cy0 = std::max(0, y0 - 4);
        const int cx1 = std::min(w - 1, x1 + 4), cy1 = std::min(h - 1, y1 + 4);
        const int cw = cx1 - cx0 + 1, ch = cy1 - cy0 + 1;
        std::vector<int32_t> a(size_t(cw) * ch), b(size_t(cw) * ch);
        for (int y = 0; y < ch; ++y)
            for (int x = 0; x < cw; ++x) {
                a[size_t(y) * cw + x] = sw.at(cx0 + x, cy0 + y);
                b[size_t(y) * cw + x] = sb.at(cx0 + x, cy0 + y);
            }
        const nn::Act f = m.tiles.sketchFeatures(ch, cw, a.data(), b.data(), pool);
        for (int c = 0; c < f.c; ++c)
            for (int y = std::max(0, y0 - 2); y <= std::min(h - 1, y1 + 2); ++y)
                for (int x = std::max(0, x0 - 2); x <= std::min(w - 1, x1 + 2); ++x)
                    sketchFeat.ch(c)[size_t(y) * w + x] = f.ch(c)[size_t(y - cy0) * cw + (x - cx0)];
    }

    /// The window around the cells (rounded to multiples of 4) and the network's features there.
    struct Window {
        int x0, y0, ww, wh;
        nn::Act feat;
    };
    Window window(const Grid<int32_t>& g, const std::vector<int>& cells, int rim) const {
        int miny = h, minx = w, maxy = -1, maxx = -1;
        for (int c : cells) {
            miny = std::min(miny, c / w);
            maxy = std::max(maxy, c / w);
            minx = std::min(minx, c % w);
            maxx = std::max(maxx, c % w);
        }
        Window win;
        win.y0 = std::max(miny - rim, 0) / 4 * 4;
        win.x0 = std::max(minx - rim, 0) / 4 * 4;
        const int y1 = std::min(h, (maxy + 1 + rim + 3) / 4 * 4), x1 = std::min(w, (maxx + 1 + rim + 3) / 4 * 4);
        win.wh = y1 - win.y0;
        win.ww = x1 - win.x0;
        std::vector<int32_t> t(size_t(win.ww) * win.wh);
        nn::Act p(4, win.wh, win.ww), s(sketchFeat.c, win.wh, win.ww);
        for (int y = 0; y < win.wh; ++y)
            for (int x = 0; x < win.ww; ++x) {
                const size_t src = size_t(win.y0 + y) * w + (win.x0 + x), dst = size_t(y) * win.ww + x;
                t[dst] = g.v[src];
                for (int c = 0; c < 4; ++c) p.ch(c)[dst] = pos.ch(c)[src];
                for (int c = 0; c < s.c; ++c) s.ch(c)[dst] = sketchFeat.ch(c)[src];
            }
        win.feat = m.tiles.features(win.wh, win.ww, t.data(), p, s, pool);
        return win;
    }

    /// Probabilities of the cells (rows of V), the wall tiles weighted by the sketch.
    void probabilities(const Window& win, const std::vector<int>& cells, double temperature, std::vector<double>& out) {
        std::vector<int> idx(cells.size());
        for (size_t i = 0; i < cells.size(); ++i)
            idx[i] = (cells[i] / w - win.y0) * win.ww + (cells[i] % w - win.x0);
        std::vector<float> logits;
        m.tiles.project(win.feat, idx, logits, pool);
        out.resize(cells.size() * size_t(V));
        pool.run(static_cast<int>(cells.size()), [&](int i) {
            double* p = &out[size_t(i) * V];
            softmax(&logits[size_t(i) * V], V, temperature, p);
            const int c = cells[size_t(i)];
            const double f = (nah.v[size_t(c)] ? 1.0 : .01) * (must.v[size_t(c)] ? kWallBoost : 1.0) *
                             (nord.v[size_t(c)] ? kNorthBoost : 1.0);
            for (int j : m.wallIdx) p[j] *= f;
            double s = 0;
            for (int j = 0; j < V; ++j) s += p[j];
            for (int j = 0; j < V; ++j) p[j] /= s;
        });
    }

    /// erlaubt: tiles that fit all neighbours already set, as bits.
    void allowed(const Grid<int32_t>& g, int x, int y, std::vector<uint64_t>& e) const {
        const int W = m.words;
        e.assign(size_t(W), ~uint64_t(0));
        if (V % 64) e.back() = (uint64_t(1) << (V % 64)) - 1;
        auto andRow = [&](const std::vector<uint64_t>& t, int a) {
            for (int k = 0; k < W; ++k) e[size_t(k)] &= t[size_t(a) * W + k];
        };
        if (x > 0 && g.at(x - 1, y) < V) andRow(m.right, g.at(x - 1, y));
        if (x < w - 1 && g.at(x + 1, y) < V) andRow(m.left, g.at(x + 1, y));
        if (y > 0 && g.at(x, y - 1) < V) andRow(m.down, g.at(x, y - 1));
        if (y < h - 1 && g.at(x, y + 1) < V) andRow(m.up, g.at(x, y + 1));
    }

    /// fuelle: fills the open cells, certain ones first; reopens dead ends.
    /// `fixed(c)`: cell c must not be reopened.
    template <class Fixed>
    void fill(Grid<int32_t>& g, Rng& rng, int rounds, double temperature, Fixed fixed, int rim = 20,
              double from = -1, double to = -1) {
        int n0 = 0;
        for (int32_t v : g.v) n0 += v == open;
        std::vector<int> cells;
        std::vector<double> pmax, rows;
        std::vector<uint64_t> e;
        for (int r = 1; r <= rounds * 2; ++r) {
            cells.clear();
            for (int i = 0; i < static_cast<int>(g.v.size()); ++i)
                if (g.v[size_t(i)] == open) cells.push_back(i);
            if (cells.empty()) return;
            check();
            const double t = std::min(1.0, double(r) / rounds);
            const int n = static_cast<int>(cells.size());
            const int count = t >= 1 ? n : std::max(1, n - static_cast<int>(n0 * std::cos(kPi / 2 * t)));
            const Window win = window(g, cells, rim);
            // Certainty of every open cell; only the chosen ones need their full row.
            pmax.assign(size_t(n), 0);
            for (int s = 0; s < n; s += 2048) {
                const std::vector<int> part(cells.begin() + s, cells.begin() + std::min(n, s + 2048));
                probabilities(win, part, temperature, rows);
                for (size_t i = 0; i < part.size(); ++i)
                    pmax[size_t(s) + i] = *std::max_element(rows.begin() + long(i) * V, rows.begin() + long(i + 1) * V);
            }
            std::vector<double> sure(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) sure[size_t(i)] = nn::detLog(pmax[size_t(i)] + 1e-9) + gumbel(rng) * 1.2 * (1 - t);
            std::vector<int> order(static_cast<size_t>(n));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sure[size_t(a)] > sure[size_t(b)]; });
            order.resize(size_t(count));
            std::vector<int> chosen(order.size());
            for (size_t i = 0; i < order.size(); ++i) chosen[i] = cells[size_t(order[i])];
            for (size_t s = 0; s < chosen.size(); s += 2048) {
                const std::vector<int> part(chosen.begin() + long(s), chosen.begin() + long(std::min(chosen.size(), s + 2048)));
                probabilities(win, part, temperature, rows);
                for (size_t i = 0; i < part.size(); ++i) {
                    const int c = part[i], y = c / w, x = c % w;
                    allowed(g, x, y, e);
                    const double* p = &rows[i * size_t(V)];
                    double total = 0;
                    for (int j = 0; j < V; ++j)
                        if ((e[size_t(j) / 64] >> (j % 64)) & 1) total += p[j];
                    if (total < 1e-12) {
                        for (int yy = std::max(0, y - 1); yy <= std::min(h - 1, y + 1); ++yy)
                            for (int xx = std::max(0, x - 1); xx <= std::min(w - 1, x + 1); ++xx)
                                if (!fixed(yy * w + xx)) g.at(xx, yy) = open;
                        continue;
                    }
                    const double u = rng.uniform() * total;
                    double cum = 0;
                    int pick = V - 1;
                    for (int j = 0; j < V; ++j) {
                        if ((e[size_t(j) / 64] >> (j % 64)) & 1) cum += p[j];
                        if (cum > u) { pick = j; break; }
                    }
                    g.at(x, y) = std::min(pick, V - 1);
                }
            }
            if (from >= 0) done = from + (to - from) * t;
        }
        cells.clear();
        for (int i = 0; i < static_cast<int>(g.v.size()); ++i)
            if (g.v[size_t(i)] == open) cells.push_back(i);
        if (cells.empty()) return;
        const Window win = window(g, cells, rim);
        probabilities(win, cells, temperature, rows);
        for (size_t i = 0; i < cells.size(); ++i) {
            const int c = cells[i], y = c / w, x = c % w;
            allowed(g, x, y, e);
            const double* p = &rows[i * size_t(V)];
            int best = -1, any = -1;
            for (int j = 0; j < V; ++j) {
                if (any < 0 || p[j] > p[any]) any = j;
                if (((e[size_t(j) / 64] >> (j % 64)) & 1) && (best < 0 || p[j] > p[best])) best = j;
            }
            g.at(x, y) = best >= 0 ? best : any;
        }
    }

    /// falsch: cells with a neighbour pair no training map has, inside [x0, x1) x [y0, y1).
    Mask wrong(const Grid<int32_t>& g, int x0 = 0, int y0 = 0, int x1 = -1, int y1 = -1) const {
        if (x1 < 0) { x1 = w; y1 = h; }
        Mask b(w, h, 0);
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                if (x + 1 < x1 && !m.seenRight(g.at(x, y), g.at(x + 1, y))) b.at(x, y) = b.at(x + 1, y) = 1;
                if (y + 1 < y1 && !m.seenDown(g.at(x, y), g.at(x, y + 1))) b.at(x, y) = b.at(x, y + 1) = 1;
            }
        return b;
    }

    /// flicke: repaints unknown tile pairs, spot by spot.
    void patch(Grid<int32_t>& g, Rng& rng, int times = 6) {
        for (int k = 0; k < times; ++k) {
            const Mask b = wrong(g);
            if (!b.any()) break;
            Grid<int32_t> labels;
            const int n = label8(dilate4(b, 2), labels);
            std::vector<std::vector<int>> spots(size_t(n) + 1);
            for (size_t i = 0; i < b.v.size(); ++i)
                if (b.v[i]) spots[size_t(labels.v[i])].push_back(static_cast<int>(i));
            for (int i = 1; i <= n; ++i) {
                const auto& z = spots[size_t(i)];
                if (z.empty()) continue;
                int miny = h, minx = w, maxy = 0, maxx = 0;
                for (int c : z) {
                    miny = std::min(miny, c / w);
                    maxy = std::max(maxy, c / w);
                    minx = std::min(minx, c % w);
                    maxx = std::max(maxx, c % w);
                }
                const int y0 = std::max(miny - 8, 0), x0 = std::max(minx - 8, 0);
                const int y1 = std::min(h, maxy + 9), x1 = std::min(w, maxx + 9);
                Mask mm(w, h, 0);
                for (int c : z) mm.v[size_t(c)] = 1;
                mm = dilate4(mm, 1 + k / 2);
                std::vector<std::pair<int, int32_t>> saved;
                for (size_t c = 0; c < mm.v.size(); ++c)
                    if (mm.v[c]) {
                        saved.emplace_back(static_cast<int>(c), g.v[c]);
                        g.v[c] = open;
                    }
                const size_t before = [&] {
                    Grid<int32_t> old = g;
                    for (const auto& [c, v] : saved) old.v[size_t(c)] = v;
                    return wrong(old, x0, y0, x1, y1).count();
                }();
                fill(g, rng, 4, 1.0, [&](int c) { return !mm.v[size_t(c)]; }, 10);
                if (wrong(g, x0, y0, x1, y1).count() > before)
                    for (const auto& [c, v] : saved) g.v[size_t(c)] = v;
            }
        }
    }

    /// reste_weg: paints tiny wall bits over with ground.
    void dropScraps(Grid<int32_t>& g, Rng& rng, int minCells = 10) {
        for (int r = 0; r < 4; ++r) {
            Mask wall(w, h);
            for (size_t i = 0; i < g.v.size(); ++i) wall.v[i] = m.vocabWall[size_t(g.v[i])];
            Grid<int32_t> labels;
            const int n = label8(wall, labels);
            std::vector<int> size(size_t(n) + 1, 0), limit(size_t(n) + 1, minCells);
            for (int32_t l : labels.v) size[size_t(l)] += 1;
            // At the map border a wall continues off the map.
            for (int x = 0; x < w; ++x) limit[size_t(labels.at(x, 0))] = limit[size_t(labels.at(x, h - 1))] = 4;
            for (int y = 0; y < h; ++y) limit[size_t(labels.at(0, y))] = limit[size_t(labels.at(w - 1, y))] = 4;
            Mask small(w, h);
            for (size_t i = 0; i < small.v.size(); ++i)
                small.v[i] = wall.v[i] && size[size_t(labels.v[i])] < limit[size_t(labels.v[i])];
            if (!small.any()) break;
            Mask mm = dilate4(small, 1 + r / 2);
            int x0 = w, y0 = h, x1 = -1, y1 = -1;
            for (size_t i = 0; i < mm.v.size(); ++i) {
                if (wall.v[i] && !small.v[i]) mm.v[i] = 0;
                if (!mm.v[i]) continue;
                const int x = static_cast<int>(i) % w, y = static_cast<int>(i) / w;
                x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y);
                sw.v[i] = 0;
            }
            const Mask oldNah = nah;
            masks();
            for (size_t i = 0; i < nah.v.size(); ++i) nah.v[i] = oldNah.v[i] && !mm.v[i] && nah.v[i];
            if (x1 >= 0) sketchChanged(x0, y0, x1, y1);
            for (size_t i = 0; i < mm.v.size(); ++i) if (mm.v[i]) g.v[i] = open;
            fill(g, rng, 5, .8, [&](int c) { return !mm.v[size_t(c)]; });
        }
        masks();
    }

    /// bauplatz: cells covered by a free 7x7 square, and the squares' corners.
    void buildSquares(const Grid<int32_t>& g, Mask& covered, Mask& corner) const {
        const int n = kBuildSquare;
        std::vector<int64_t> s(size_t(w + 1) * (h + 1), 0);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                s[size_t(y + 1) * (w + 1) + x + 1] = m.vocabFree[size_t(g.at(x, y))] + s[size_t(y) * (w + 1) + x + 1] +
                                                     s[size_t(y + 1) * (w + 1) + x] - s[size_t(y) * (w + 1) + x];
        covered = Mask(w, h, 0);
        corner = Mask(std::max(0, w - n + 1), std::max(0, h - n + 1), 0);
        for (int y = 0; y + n <= h; ++y)
            for (int x = 0; x + n <= w; ++x) {
                const int64_t v = s[size_t(y + n) * (w + 1) + x + n] - s[size_t(y) * (w + 1) + x + n] -
                                  s[size_t(y + n) * (w + 1) + x] + s[size_t(y) * (w + 1) + x];
                if (v != n * n) continue;
                corner.at(x, y) = 1;
                for (int dy = 0; dy < n; ++dy)
                    for (int dx = 0; dx < n; ++dx) covered.at(x + dx, y + dy) = 1;
            }
    }

    /// bergstufen: finds building ground inside a mountain on the finished map,
    /// draws a terrace there and repaints only that spot. Never on sand.
    void terraces(Grid<int32_t>& g, const Mask& mountain, double density, Rng& rng) {
        const int rounds = static_cast<int>(std::nearbyint(14 * density * density));
        const double goal = .16 - .12 * density;
        const size_t shortest = density < .8 ? 5 : 3;
        int gap = 3;
        const int hh = h / 2, hw = w / 2, mid = kBuildSquare / 2;
        const size_t mountainCells = mountain.count();
        for (int it = 0; it < rounds; ++it) {
            check();
            Mask covered, corner;
            buildSquares(g, covered, corner);
            if (!mountainCells) break;
            size_t inside = 0;
            for (size_t i = 0; i < covered.v.size(); ++i) inside += covered.v[i] && mountain.v[i];
            if (double(inside) <= goal * double(mountainCells)) break;
            Mask wall(w, h);
            for (size_t i = 0; i < g.v.size(); ++i) wall.v[i] = m.vocabWall[size_t(g.v[i])];
            const Mask near = dilate4(wall, gap);
            std::vector<int> walls;
            for (size_t i = 0; i < wall.v.size(); ++i) if (wall.v[i]) walls.push_back(static_cast<int>(i));
            Mask painted(w, h, 0);
            Grid<int32_t> sw2 = sw;
            std::vector<int> spots;
            for (int y = 0; y < corner.h; ++y)
                for (int x = 0; x < corner.w; ++x)
                    if (corner.at(x, y) && mountain.at(x + mid, y + mid)) spots.push_back((y + mid) * w + (x + mid));
            shuffle(spots, rng);
            for (int c : spots) {
                const int y = c / w, x = c % w;
                if (sb.at(x, y) == kSand) continue;
                bool busy = false;
                for (int yy = std::max(0, y - 7); yy < std::min(h, y + 8) && !busy; ++yy)
                    for (int xx = std::max(0, x - 7); xx < std::min(w, x + 8) && !busy; ++xx) busy = painted.at(xx, yy);
                if (busy) continue;
                int r = static_cast<int>(rng.integers(0, 5));
                if (!walls.empty()) {
                    int best = walls[0], bd = 1 << 30;
                    for (int q : walls) {
                        const int d = std::abs(q / w - y) + std::abs(q % w - x);
                        if (d < bd) { bd = d; best = q; }
                    }
                    const int v = sw.v[size_t(best)];
                    if (v && !isNorth(v)) r = v - 1;
                }
                std::map<std::pair<int, int>, int> line;   // keyed like the Python dict; order does not matter here
                for (const int sense : {-1, 1}) {
                    double tx = kDirY[r] * sense, ty = -kDirX[r] * sense;
                    double fy = y / 2.0, fx = x / 2.0;
                    const double turn = rng.normal() * .10;
                    const int steps = static_cast<int>(rng.integers(3, 9));
                    for (int k = 0; k < steps; ++k) {
                        const int py = static_cast<int>(fy), px = static_cast<int>(fx);
                        if (!(py >= 0 && py < hh && px >= 0 && px < hw) || !mountain.at(2 * px, 2 * py)) break;
                        bool close = false;
                        for (int dy = 0; dy < 2; ++dy)
                            for (int dx = 0; dx < 2; ++dx) close = close || near.at(2 * px + dx, 2 * py + dy);
                        if (close) break;
                        line[{py, px}] = fallDirection(tx * sense, ty * sense) + 1;
                        const double ang = std::atan2(ty, tx) + turn;
                        tx = std::cos(ang);
                        ty = std::sin(ang);
                        fx += tx;
                        fy += ty;
                    }
                }
                if (line.size() < shortest) continue;
                for (const auto& [q, v] : line)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            sw2.at(2 * q.second + dx, 2 * q.first + dy) = v;
                            painted.at(2 * q.second + dx, 2 * q.first + dy) = 1;
                        }
            }
            if (!painted.any()) {
                --gap;
                if (density < .8 || gap < 0) break;
                continue;
            }
            int x0 = w, y0 = h, x1 = -1, y1 = -1;
            for (size_t i = 0; i < sw.v.size(); ++i)
                if (sw.v[i] != sw2.v[i]) {
                    const int x = static_cast<int>(i) % w, y = static_cast<int>(i) / w;
                    x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y);
                }
            sw = std::move(sw2);
            masks();
            if (x1 >= 0) sketchChanged(x0, y0, x1, y1);
            Mask mm = dilate4(painted, 4);
            for (size_t i = 0; i < mm.v.size(); ++i) {
                if (wall.v[i] && !painted.v[i]) mm.v[i] = 0;
                if (mm.v[i]) g.v[i] = open;
            }
            fill(g, rng, 10, .9, [&](int c) { return !mm.v[size_t(c)]; });
        }
    }
};

// ---------------------------------------------------------------------------
// Finishing (_Nacharbeit)

struct Finisher {
    const LearnedModel& m;
    Canvas& c;
    BlendMap& bm;
    Rng& rng;
    const GeneratorData& gd;
    std::vector<uint8_t> wall, transition, object;
    std::vector<int> pure;                       ///< ground of plain tiles, -1 otherwise (rein)
    std::vector<std::vector<int>> fillTiles;     ///< per ground (fuell)

    Finisher(const LearnedModel& model, Canvas& canvas, BlendMap& blend, Rng& r)
        : m(model), c(canvas), bm(blend), rng(r), gd(canvas.profile.generator) {
        const size_t n = m.mapCell.size();
        wall.resize(n);
        transition.resize(n);
        object.resize(n);
        pure.assign(n, -1);
        fillTiles.resize(c.profile.grounds.size());
        for (size_t t = 0; t < n; ++t) {
            const int ct = m.mapCell[t], g = m.mapGround[t], f = m.mapFlags[t];
            wall[t] = isCliff(ct);
            transition[t] = (f & LearnedModel::kTransition) != 0;
            const bool fill = g >= 0 && (f & LearnedModel::kPlaceable);
            object[t] = (((f & LearnedModel::kDetail) && !fill) || ct == kImpassable1 || ct == kImpassable2) && !wall[t] &&
                        !transition[t];
            if (g >= 0 && !transition[t] && !wall[t]) pure[t] = g;
        }
        for (size_t t = 0; t < n; ++t) {
            const int g = m.mapGround[t];
            if (g >= 0 && size_t(g) < fillTiles.size() && (m.mapFlags[t] & LearnedModel::kPlaceable) && !wall[t] &&
                !transition[t] && !object[t])
                fillTiles[size_t(g)].push_back(static_cast<int>(t));
        }
    }

    void put(int x, int y, int t) {
        c.put(x, y, t, m.mapCell[size_t(t)]);
        bm.setTile(x, y, t);
    }

    /// _passt: does tile k fit the neighbours (pairs seen in the original maps)?
    bool fits(int x, int y, int k) const {
        auto ok = [&](bool horizontal, int a, int b) {
            const uint32_t pa = c.packedOfMapping[size_t(a)], pb = c.packedOfMapping[size_t(b)];
            return !gd.known(horizontal, pa) || gd.seen(horizontal, pa, pb);
        };
        if (x > 0 && !ok(true, c.tiles.at(x - 1, y), k)) return false;
        if (x < c.w - 1 && !ok(true, k, c.tiles.at(x + 1, y))) return false;
        if (y > 0 && !ok(false, c.tiles.at(x, y - 1), k)) return false;
        if (y < c.h - 1 && !ok(false, k, c.tiles.at(x, y + 1))) return false;
        return true;
    }

    /// _boden: a fill tile of `ground` that fits, else the first candidate.
    void groundTile(int x, int y, int ground) {
        if (ground < 0 || size_t(ground) >= fillTiles.size() || fillTiles[size_t(ground)].empty()) return;
        std::vector<int> probe = fillTiles[size_t(ground)];
        shuffle(probe, rng);
        probe.resize(std::min<size_t>(40, probe.size()));
        for (int k : probe)
            if (fits(x, y, k)) {
                put(x, y, k);
                return;
            }
        put(x, y, probe[0]);
    }

    /// _familie: ground per cell; walls and objects take their neighbour's.
    Grid<int32_t> grounds() const {
        Grid<int32_t> f(c.w, c.h);
        for (size_t i = 0; i < f.v.size(); ++i) {
            const int t = c.tiles.v[i];
            f.v[i] = object[size_t(t)] ? -1 : pure[size_t(t)];
        }
        const int d[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int it = 0; it < 24; ++it) {
            bool empty = false;
            for (int32_t v : f.v) empty = empty || v < 0;
            if (!empty) break;
            for (const auto& dd : d) {
                const Grid<int32_t> before = f;
                for (int y = 0; y < c.h; ++y)
                    for (int x = 0; x < c.w; ++x) {
                        const int sy = y - dd[0], sx = x - dd[1];
                        if (f.at(x, y) < 0 && sy >= 0 && sy < c.h && sx >= 0 && sx < c.w && before.at(sx, sy) >= 0)
                            f.at(x, y) = before.at(sx, sy);
                    }
            }
        }
        return f;
    }

    size_t objectCount() const {
        size_t n = 0;
        for (int32_t t : c.tiles.v) n += object[size_t(t)];
        return n;
    }

    Mask objectMask() const {
        Mask o(c.w, c.h);
        for (size_t i = 0; i < o.v.size(); ++i) o.v[i] = object[size_t(c.tiles.v[i])];
        return o;
    }

    /// objekte: removes whole objects, only the share `keep` stays.
    void objects(double keep) {
        Grid<int32_t> labels;
        const int n = label8(objectMask(), labels);
        std::vector<uint8_t> gone(size_t(n) + 1, 0);
        for (int i = 1; i <= n; ++i) gone[size_t(i)] = rng.uniform() >= keep;
        const Grid<int32_t> fam = grounds();
        for (int pass = 0; pass < 2; ++pass)
            for (int y = 0; y < c.h; ++y)
                for (int x = 0; x < c.w; ++x)
                    if (gone[size_t(labels.at(x, y))] && object[size_t(c.tiles.at(x, y))]) groundTile(x, y, fam.at(x, y));
    }

    /// sprenkel: tiny patches of a foreign ground dissolve into the surrounding ground.
    void specks(int big = 30) {
        std::vector<int> fams;
        for (const char* id : {"mud", "roc", "san"}) {
            const int g = c.profile.findGround(id);
            if (g >= 0) fams.push_back(g);
        }
        for (int pass = 0; pass < 2; ++pass)
            for (const int f : fams) {
                Mask foreign(c.w, c.h);
                for (size_t i = 0; i < foreign.v.size(); ++i) {
                    const int t = c.tiles.v[i];
                    foreign.v[i] = pure[size_t(t)] != f && !wall[size_t(t)];
                }
                Grid<int32_t> labels;
                const int n = label8(foreign, labels);
                std::vector<int> size(size_t(n) + 1, 0);
                for (int32_t l : labels.v) size[size_t(l)] += 1;
                size[0] = big;
                std::vector<uint8_t> border(size_t(n) + 1, 0);
                for (int x = 0; x < c.w; ++x) border[size_t(labels.at(x, 0))] = border[size_t(labels.at(x, c.h - 1))] = 1;
                for (int y = 0; y < c.h; ++y) border[size_t(labels.at(0, y))] = border[size_t(labels.at(c.w - 1, y))] = 1;
                for (int i = 0; i <= n; ++i) if (border[size_t(i)]) size[size_t(i)] += 12;
                // groups in order of their first cell
                std::vector<int> order;
                std::vector<std::vector<int>> cells(size_t(n) + 1);
                for (size_t i = 0; i < labels.v.size(); ++i) {
                    const int l = labels.v[i];
                    if (l > 0 && size[size_t(l)] < big) {
                        if (cells[size_t(l)].empty()) order.push_back(l);
                        cells[size_t(l)].push_back(static_cast<int>(i));
                    }
                }
                for (const int l : order) {
                    const auto& z = cells[size_t(l)];
                    bool other = false;
                    for (int p : z) {
                        const int t = c.tiles.v[size_t(p)];
                        other = other || transition[size_t(t)] ||
                                (pure[size_t(t)] >= 0 && pure[size_t(t)] != f &&
                                 std::find(fams.begin(), fams.end(), pure[size_t(t)]) != fams.end());
                    }
                    if (!other) continue;   // objects only
                    int y0 = c.h, x0 = c.w, y1 = 0, x1 = 0;
                    for (int p : z) {
                        y0 = std::min(y0, p / c.w); y1 = std::max(y1, p / c.w);
                        x0 = std::min(x0, p % c.w); x1 = std::max(x1, p % c.w);
                    }
                    y0 = std::max(0, y0 - 1); x0 = std::max(0, x0 - 1);
                    y1 = std::min(c.h, y1 + 2); x1 = std::min(c.w, x1 + 2);
                    int same = 0;
                    for (int yy = y0; yy < y1; ++yy)
                        for (int xx = x0; xx < x1; ++xx) same += pure[size_t(c.tiles.at(xx, yy))] == f;
                    const int around = (y1 - y0) * (x1 - x0) - static_cast<int>(z.size());
                    if (same < .6 * std::max(1, around)) continue;
                    for (int p : z) groundTile(p % c.w, p / c.w, f);
                }
            }
    }

    /// felsen: last resort, impassable rocks where one could still build inside
    /// a mountain or gorge. Never on sand.
    void rocks(Mask tight, double goal = .04) {
        std::vector<const Stamp*> stamps;
        for (const Stamp& s : c.profile.generator.details) {
            if (std::max(s.w, s.h) > 4) continue;
            bool blocks = false;
            for (const StampTile& t : s.tiles) {
                int ct = t.cellType;
                if (ct < 0) {
                    const int mp = stampMapping(c, t);
                    if (mp >= 0 && size_t(mp) < m.mapCell.size()) ct = m.mapCell[size_t(mp)];
                }
                blocks = blocks || ct == kImpassable1 || ct == kImpassable2;
            }
            if (blocks) stamps.push_back(&s);
        }
        const int sand = c.profile.findGround("san");
        const Grid<int32_t> fam = grounds();
        for (size_t i = 0; i < tight.v.size(); ++i) if (fam.v[i] == sand && sand >= 0) tight.v[i] = 0;
        const int n = kBuildSquare, mid = n / 2, w = c.w, h = c.h;
        const size_t tightCells = tight.count();
        int fails = 0;
        while (!stamps.empty() && tightCells && fails < 60) {
            std::vector<int64_t> s(size_t(w + 1) * (h + 1), 0);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    Grid<uint8_t> one(1, 1, static_cast<uint8_t>(c.cells.at(x, y) & 31));
                    s[size_t(y + 1) * (w + 1) + x + 1] = passableMask(one).v[0] + s[size_t(y) * (w + 1) + x + 1] +
                                                         s[size_t(y + 1) * (w + 1) + x] - s[size_t(y) * (w + 1) + x];
                }
            std::vector<std::pair<int, int>> spots;
            for (int y = 0; y + n <= h; ++y)
                for (int x = 0; x + n <= w; ++x) {
                    const int64_t v = s[size_t(y + n) * (w + 1) + x + n] - s[size_t(y) * (w + 1) + x + n] -
                                      s[size_t(y + n) * (w + 1) + x] + s[size_t(y) * (w + 1) + x];
                    if (v == n * n && tight.at(x + mid, y + mid)) spots.emplace_back(y + mid, x + mid);
                }
            if (double(spots.size()) <= goal * double(tightCells) / 4) break;
            shuffle(spots, rng);
            spots.resize(std::min<size_t>(40, spots.size()));
            std::vector<std::pair<int, int>> placed;
            for (const auto& [y, x] : spots) {
                bool close = false;
                for (const auto& [a, b] : placed) close = close || (std::abs(y - a) < 8 && std::abs(x - b) < 8);
                if (close) continue;
                const Stamp& b = *stamps[size_t(rng.integers(0, int64_t(stamps.size())))];
                const int y0 = y - b.h / 2, x0 = x - b.w / 2;
                if (y0 < 0 || x0 < 0 || y0 + b.h > h || x0 + b.w > w) continue;
                bool walls = false;
                for (int yy = y0; yy < y0 + b.h; ++yy)
                    for (int xx = x0; xx < x0 + b.w; ++xx) walls = walls || wall[size_t(c.tiles.at(xx, yy))];
                if (walls || !seamFits(c, b, x0, y0)) continue;
                if (!placeStamp(c, bm, b, x0, y0)) continue;
                placed.emplace_back(y, x);
            }
            fails = placed.empty() ? fails + 1 : 0;
        }
    }
};

}  // namespace

Result paintMap(const Profile& profile, const LearnedModel& m, const Params& p, const Sketch& sk, int quality,
                nn::Pool& pool, const ProgressFn& progress) {
    Rng rng = stream(p.seed, 62);
    Painter painter(m, sk, pool, progress);
    const int w = painter.w, h = painter.h;
    Grid<int32_t> g(w, h, painter.open);
    painter.fill(g, rng, quality >= 2 ? 24 : 6, .95, [](int) { return false; }, 20, 0.0, .6);
    painter.done = .6;
    painter.patch(g, rng);
    painter.done = .75;
    painter.dropScraps(g, rng);
    Mask mountain(w, h);
    for (size_t i = 0; i < mountain.v.size(); ++i) mountain.v[i] = sk.zones.v[i] == 1;
    painter.terraces(g, mountain, std::clamp(p.mountainDensity, 0.0, 1.0), rng);
    painter.done = .88;
    painter.dropScraps(g, rng);
    painter.patch(g, rng, 8);
    painter.done = .95;
    painter.check();

    // --- finishing
    Canvas c(profile, w, h);
    for (size_t i = 0; i < g.v.size(); ++i) {
        const int t = m.vocabMapping[size_t(g.v[i])];
        c.tiles.v[i] = t;
        c.cells.v[i] = m.mapCell[size_t(t)];
    }
    const std::vector<om_mapping> table = standardTable(profile);
    std::vector<uint16_t> init(c.tiles.v.begin(), c.tiles.v.end());
    BlendMap bm(profile, w, h, table.data(), static_cast<int>(table.size()), init.data());
    Finisher fin(m, c, bm, rng);
    fin.specks();
    const double goal = p.details * .06;
    const double share = double(fin.objectCount()) / double(w * h);
    fin.objects(share > 0 ? std::min(1.0, goal / share) : 0.0);
    if (p.learnedRocks) {
        Mask tight(w, h);
        for (size_t i = 0; i < tight.v.size(); ++i) tight.v[i] = sk.zones.v[i] == 1 || sk.zones.v[i] == 4;
        fin.rocks(tight);
    }
    const int more = static_cast<int>(std::nearbyint((goal - double(fin.objectCount()) / double(w * h)) * w * h));
    if (more > 0 && goal > share) {
        Field fam(w, h);
        for (size_t i = 0; i < fam.v.size(); ++i) fam.v[i] = static_cast<int16_t>(m.mapGround[size_t(c.tiles.v[i])]);
        Mask wallNear(w, h);
        for (size_t i = 0; i < wallNear.v.size(); ++i) wallNear.v[i] = fin.wall[size_t(c.tiles.v[i])];
        Mask blocked = dilate4(wallNear, 1);
        for (size_t i = 0; i < blocked.v.size(); ++i) blocked.v[i] |= sk.zones.v[i] == 3 || sk.zones.v[i] == 4;
        scatterDetails(c, bm, fam, rng, more, blocked, Sym::None);
    }

    Result r;
    r.tiles = c.tiles;
    r.cells = Grid<uint8_t>(w, h);
    for (size_t i = 0; i < r.cells.v.size(); ++i) r.cells.v[i] = static_cast<uint8_t>(c.cells.v[i] & 31);
    const Grid<uint8_t>& cells = r.cells;

    // --- report (_bericht)
    MountainReport& z = r.zones;
    zoneReport(cells, sk.plan, z);
    const Mask open = passableMask(cells);
    z.gorgesPassable = 0;
    for (const Cut& e : sk.plan.gorges) {
        Mask area = dilate4(e.area, 5);
        for (size_t i = 0; i < area.v.size(); ++i) area.v[i] = area.v[i] && open.v[i];
        const Mask a = dilate4(e.endA, 5), b = dilate4(e.endB, 5);
        // 4-neighbour flood from the first end
        Mask reach(w, h, 0);
        std::vector<int> stack;
        for (size_t i = 0; i < a.v.size(); ++i)
            if (a.v[i] && area.v[i]) { reach.v[i] = 1; stack.push_back(static_cast<int>(i)); }
        while (!stack.empty()) {
            const int q = stack.back();
            stack.pop_back();
            const int x = q % w, y = q / w;
            const int nb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& d : nb) {
                const int nx = x + d[0], ny = y + d[1];
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                const size_t ni = size_t(ny) * w + nx;
                if (area.v[ni] && !reach.v[ni]) { reach.v[ni] = 1; stack.push_back(static_cast<int>(ni)); }
            }
        }
        bool through = false;
        for (size_t i = 0; i < b.v.size(); ++i) through = through || (b.v[i] && reach.v[i]);
        z.gorgesPassable += through;
    }
    const Mask area = largestArea(open);
    Mask zone1(w, h), zone2(w, h);
    for (size_t i = 0; i < zone1.v.size(); ++i) {
        zone1.v[i] = sk.zones.v[i] == 1;
        zone2.v[i] = sk.zones.v[i] == 2;
    }
    z.mountains = static_cast<int>(components(zone1).size());
    z.leaky = 0;
    const auto plateaus = components(zone2);
    z.plateaus = static_cast<int>(plateaus.size());
    z.unreachable = 0;
    for (const auto& e : plateaus) {
        size_t in = 0;
        for (int i : e) in += area.v[size_t(i)];
        z.unreachable += double(in) < .3 * double(e.size());
    }

    // --- spawns: with symmetry only where the mirror image is free too
    Grid<uint8_t> forSpawns = cells;
    if (p.symmetry != Sym::None) {
        Mask all = open;
        auto andImage = [&](bool fx, bool fy) {
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    if (!open.at(fx ? w - 1 - x : x, fy ? h - 1 - y : y)) all.at(x, y) = 0;
        };
        if (p.symmetry == Sym::MirrorX || p.symmetry == Sym::Quad) andImage(true, false);
        if (p.symmetry == Sym::MirrorY || p.symmetry == Sym::Quad) andImage(false, true);
        if (p.symmetry == Sym::Rot180 || p.symmetry == Sym::Quad) andImage(true, true);
        for (size_t i = 0; i < all.v.size(); ++i) if (!all.v[i]) forSpawns.v[i] = kImpassable1;
    }
    r.spawns = pickSpawns(forSpawns, p.spawns, p.spawnClearance, p.symmetry, rng);
    const double cellsAll = double(w) * h;
    r.passableShare = double(open.count()) / cellsAll;
    r.connected = connected(cells, r.spawns);
    size_t cliff = 0;
    for (uint8_t ct : cells.v) cliff += isCliff(ct);
    r.plateauShare = double(cliff) / cellsAll;
    r.detailShare = double(fin.objectCount()) / cellsAll;
    r.familyShare.assign(profile.grounds.size(), 0.0);
    for (int32_t t : c.tiles.v) {
        const int gr = m.mapGround[size_t(t)];
        if (gr >= 0 && size_t(gr) < r.familyShare.size()) r.familyShare[size_t(gr)] += 1.0 / cellsAll;
    }
    r.sketchWall = sk.wall;
    r.sketchGround = sk.ground;
    painter.done = 1.0;
    if (progress) progress(1.0);
    return r;
}

}  // namespace om::gen
