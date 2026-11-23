// Height-field shapes for the map generator (Python: shapegen.py).
//
// Masks are 0/1 highland fields: blobs for mountains, a corridor for valleys,
// a barrier with a gap for chokes, plus the clean-up passes (majority
// smoothing, snapping edges to a coarser grid, removing specks and holes).
#include "mapgen.hpp"

#include <cmath>
#include <deque>

namespace om::gen {

namespace {

// Squared distance of (x, y) to (cx, cy).
inline double dist2(int x, int y, double cx, double cy) {
    const double dx = x - cx, dy = y - cy;
    return dx * dx + dy * dy;
}

// Sets every cell within `r` of (cx, cy) in m (Python: feld |= (d^2 <= r^2)).
void stampDisc(Mask& m, double cx, double cy, double r) {
    const double r2 = r * r;
    for (int y = 0; y < m.h; ++y)
        for (int x = 0; x < m.w; ++x)
            if (dist2(x, y, cx, cy) <= r2) m.at(x, y) = 1;
}

}  // namespace

// glaette: majority vote over the 3x3 neighbourhood. Like the Python's
// np.roll the neighbourhood wraps around the map edges.
Mask smoothMask(const Mask& m, int passes) {
    Mask f = m;
    for (uint8_t& c : f.v) c = c ? 1 : 0;
    if (f.w == 0 || f.h == 0) return f;
    for (int p = 0; p < passes; ++p) {
        Mask out(f.w, f.h);
        for (int y = 0; y < f.h; ++y) {
            for (int x = 0; x < f.w; ++x) {
                int sum = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int yy = (y + dy + f.h) % f.h;
                    for (int dx = -1; dx <= 1; ++dx) sum += f.at((x + dx + f.w) % f.w, yy);
                }
                out.at(x, y) = sum * 2 > 9 ? 1 : 0;
            }
        }
        f = std::move(out);
    }
    return f;
}

// begradige: majority vote per block x block square, scaled back up. Blocks
// cut off by the map edge count the missing cells as 0 (zero padding).
Mask snapEdge(const Mask& m, int block) {
    if (block < 2) return m;
    Mask out(m.w, m.h);
    for (int by = 0; by < m.h; by += block) {
        for (int bx = 0; bx < m.w; bx += block) {
            int sum = 0;
            for (int y = by; y < std::min(by + block, m.h); ++y)
                for (int x = bx; x < std::min(bx + block, m.w); ++x) sum += m.at(x, y) ? 1 : 0;
            if (sum * 2 <= block * block) continue;
            for (int y = by; y < std::min(by + block, m.h); ++y)
                for (int x = bx; x < std::min(bx + block, m.w); ++x) out.at(x, y) = 1;
        }
    }
    return out;
}

// blob: round but irregular area, radius perturbed by four harmonics per direction.
Mask blob(int w, int h, Point centre, int radius, Rng& rng, double roughness) {
    const double cx = centre.first, cy = centre.second;
    double a[4], b[4];   // sin and cos weights of harmonics 1..4, drawn in the Python's order
    for (int k = 0; k < 4; ++k) {
        a[k] = rng.normal();
        b[k] = rng.normal();
    }
    Mask m(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double angle = std::atan2(y - cy, x - cx);
            double noise = 0;
            for (int k = 1; k <= 4; ++k) noise += (a[k - 1] * std::sin(k * angle) + b[k - 1] * std::cos(k * angle)) / k;
            const double r = std::max(radius * (1.0 + roughness * noise / 3.0), 2.0);
            if (dist2(x, y, cx, cy) <= r * r) m.at(x, y) = 1;
        }
    }
    return smoothMask(m, 1);
}

// korridor: a wide band of highland along the path with the corridor punched out of it.
Mask corridor(int w, int h, const Path& path, int width, Rng& rng, double roughness) {
    Mask band(w, h), hole(w, h);
    // std::nearbyint rounds half to even, like Python's round().
    const int outer = width + 8 + static_cast<int>(std::nearbyint(6 * roughness));
    for (size_t i = 0; i < path.size(); ++i) {
        if (i % 2) continue;
        const double wobble = roughness * 2.0 * rng.normal();
        const double px = path[i].first, py = path[i].second;
        stampDisc(band, px, py, outer + wobble);
        stampDisc(hole, px, py, width + wobble * 0.5);
    }
    for (size_t i = 0; i < band.v.size(); ++i) band.v[i] = band.v[i] && !hole.v[i];
    return smoothMask(band, 1);
}

// riegel: a barrier of discs along the path, leaving out the points within `gap` of `gapAt`.
Mask barrier(int w, int h, const Path& path, int gapAt, int gap, Rng& rng) {
    Mask m(w, h);
    for (size_t i = 0; i < path.size(); ++i) {
        if (std::abs(static_cast<int>(i) - gapAt) <= gap) continue;
        const double r = 6 + 1.2 * rng.normal();
        stampDisc(m, path[i].first, path[i].second, std::max(3.0, r));
    }
    return smoothMask(m, 1);
}

// entferne_kleinkram: flips 4-connected highland patches, then lowland holes,
// smaller than minCells. The hole pass sees the result of the patch pass.
Mask removeSmallBits(const Mask& m, int minCells) {
    Mask out = m;
    for (uint8_t& c : out.v) c = c ? 1 : 0;
    const int w = out.w, h = out.h;
    std::vector<Point> cells;
    std::deque<Point> q;
    for (int value : {1, 0}) {
        std::vector<uint8_t> seen(out.v.size(), 0);
        for (int sy = 0; sy < h; ++sy) {
            for (int sx = 0; sx < w; ++sx) {
                if (seen[size_t(sy) * w + sx] || out.at(sx, sy) != value) continue;
                q.assign(1, Point(sx, sy));
                seen[size_t(sy) * w + sx] = 1;
                cells.clear();
                while (!q.empty()) {
                    const auto [x, y] = q.front();
                    q.pop_front();
                    cells.push_back({x, y});
                    static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                    for (const auto& d : dirs) {
                        const int nx = x + d[0], ny = y + d[1];
                        if (out.inside(nx, ny) && !seen[size_t(ny) * w + nx] && out.at(nx, ny) == value) {
                            seen[size_t(ny) * w + nx] = 1;
                            q.push_back({nx, ny});
                        }
                    }
                }
                if (static_cast<int>(cells.size()) < minCells)
                    for (const auto& [x, y] : cells) out.at(x, y) = static_cast<uint8_t>(1 - value);
            }
        }
    }
    return out;
}

}  // namespace om::gen
