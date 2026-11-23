// Passability, spawn points and connectivity (Python: mapgen_wege.py).
#include "mapgen.hpp"

#include <limits>

namespace om::gen {

namespace {

// Cell types (0..31) vehicles can cross: the fast, medium and slow speed
// classes of celltypes.SPEED_CLASS (Python: PASSABLE_CLASSES).
//   fast:   0 FAST_PASSIBLE_1, 7 FAST_PASSIBLE_2, 21 DOZED_AREA, 26 TUBE_0
//   medium: 4 MEDIUM_PASSIBLE_1, 5 MEDIUM_PASSIBLE_2
//   slow:   2 SLOW_PASSIBLE_1, 3 SLOW_PASSIBLE_2, 22 RUBBLE
// Everything else (impassable, cliffs, vents, walls, unknown) blocks.
constexpr uint32_t kPassableBits = (1u << 0) | (1u << 2) | (1u << 3) | (1u << 4) | (1u << 5) |
                                   (1u << 7) | (1u << 21) | (1u << 22) | (1u << 26);

bool passableType(uint8_t cellType) {
    return cellType < 32 && ((kPassableBits >> cellType) & 1u);
}

const int kDirs4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

}  // namespace

// _passable_mask
Mask passableMask(const Grid<uint8_t>& cellTypes) {
    Mask m(cellTypes.w, cellTypes.h);
    for (size_t i = 0; i < m.v.size(); ++i) m.v[i] = passableType(cellTypes.v[i]) ? 1 : 0;
    return m;
}

// _clearance: summed-area table over the mask; cells off the map count as blocked.
Floats clearance(const Mask& passable, int radius) {
    const int w = passable.w, h = passable.h;
    const int size = 2 * radius + 1;
    // integral(x, y) = number of set cells in [0, x) x [0, y) of the zero-padded mask,
    // stored with the padding offset so I(px, py) uses padded coordinates.
    const int pw = w + 2 * radius, ph = h + 2 * radius;
    std::vector<int> integral(size_t(pw + 1) * (ph + 1), 0);
    auto I = [&](int x, int y) -> int& { return integral[size_t(y) * (pw + 1) + x]; };
    for (int y = 1; y <= ph; ++y) {
        int row = 0;
        for (int x = 1; x <= pw; ++x) {
            const int mx = x - 1 - radius, my = y - 1 - radius;
            row += passable.get(mx, my, 0) ? 1 : 0;
            I(x, y) = I(x, y - 1) + row;
        }
    }
    Floats out(w, h);
    const double area = double(size) * size;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int total = I(x + size, y + size) - I(x + size, y) - I(x, y + size) + I(x, y);
            out.at(x, y) = total / area;
        }
    return out;
}

// _groesstes_gebiet: the first-found largest 4-connected area.
Mask largestArea(const Mask& m) {
    if (!m.any()) return m;
    const int w = m.w, h = m.h;
    std::vector<int> label(m.v.size(), 0);
    int next = 1, best = 0, bestSize = 0;
    std::vector<Point> stack;
    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            if (!m.at(sx, sy) || label[size_t(sy) * w + sx]) continue;
            const int id = next++;
            label[size_t(sy) * w + sx] = id;
            stack.assign(1, Point(sx, sy));
            int size = 0;
            while (!stack.empty()) {
                const auto [x, y] = stack.back();
                stack.pop_back();
                ++size;
                for (const auto& d : kDirs4) {
                    const int nx = x + d[0], ny = y + d[1];
                    if (m.inside(nx, ny) && m.at(nx, ny) && !label[size_t(ny) * w + nx]) {
                        label[size_t(ny) * w + nx] = id;
                        stack.push_back({nx, ny});
                    }
                }
            }
            if (size > bestSize) {   // strict: ties keep the earlier area, like np.argmax
                bestSize = size;
                best = id;
            }
        }
    }
    Mask out(w, h);
    for (size_t i = 0; i < out.v.size(); ++i) out.v[i] = label[i] == best ? 1 : 0;
    return out;
}

// _pick_spawns: start positions in the largest passable area, fully free within
// `clearance`, spread out by farthest-point picking; mirrored under symmetry.
std::vector<Point> pickSpawns(const Grid<uint8_t>& cellTypes, int count, int clearance_, Sym s, Rng& rng) {
    if (count <= 0) return {};
    if (clearance_ < 0) clearance_ = 0;
    const Mask mask = largestArea(passableMask(cellTypes));
    const Floats clear = clearance(mask, clearance_);
    const int w = mask.w, h = mask.h;
    const int margin = clearance_ + 2;

    const bool halfX = s == Sym::MirrorX || s == Sym::Rot180 || s == Sym::Quad;
    const bool halfY = s == Sym::MirrorY || s == Sym::Quad;
    // Candidates in row-major order, like np.nonzero.
    std::vector<Point> cand;
    for (int y = margin; y < h - margin; ++y) {
        if (halfY && y >= h / 2) break;
        for (int x = margin; x < w - margin; ++x) {
            if (halfX && x >= w / 2) break;
            if (clear.at(x, y) >= 0.995) cand.push_back({x, y});
        }
    }
    if (cand.empty()) return {};

    auto mirror = [&](Point p) {
        const int x = p.first, y = p.second;
        std::vector<Point> m;
        switch (s) {
        case Sym::MirrorX: m = {{w - 1 - x, y}}; break;
        case Sym::MirrorY: m = {{x, h - 1 - y}}; break;
        case Sym::Rot180: m = {{w - 1 - x, h - 1 - y}}; break;
        case Sym::Quad: m = {{w - 1 - x, y}, {x, h - 1 - y}, {w - 1 - x, h - 1 - y}}; break;
        case Sym::None: break;
        }
        return m;
    };

    std::vector<Point> picked;
    picked.push_back(cand[size_t(rng.integers(0, int64_t(cand.size())))]);
    const int perMirror = 1 + static_cast<int>(mirror(picked[0]).size());
    const int needed = std::max(1, (count + perMirror - 1) / perMirror);
    // Squared distance of each candidate to its nearest picked point.
    std::vector<int64_t> dist(cand.size(), std::numeric_limits<int64_t>::max());
    while (static_cast<int>(picked.size()) < needed) {
        const Point& last = picked.back();
        size_t best = 0;
        for (size_t i = 0; i < cand.size(); ++i) {
            const int64_t dx = cand[i].first - last.first, dy = cand[i].second - last.second;
            dist[i] = std::min(dist[i], dx * dx + dy * dy);
            if (dist[i] > dist[best]) best = i;
        }
        if (dist[best] == 0) break;
        picked.push_back(cand[best]);
    }

    std::vector<Point> spawns;
    for (const Point& p : picked) {
        spawns.push_back(p);
        for (const Point& q : mirror(p)) spawns.push_back(q);
    }
    if (static_cast<int>(spawns.size()) > count) spawns.resize(size_t(count));
    return spawns;
}

// _connected: flood fill (4 neighbours) from the first spawn over passable cells.
bool connected(const Grid<uint8_t>& cellTypes, const std::vector<Point>& spawns) {
    if (spawns.size() < 2) return true;
    const Mask mask = passableMask(cellTypes);
    const int w = mask.w;
    std::vector<uint8_t> seen(mask.v.size(), 0);
    std::vector<Point> stack{spawns[0]};
    seen[size_t(spawns[0].second) * w + spawns[0].first] = 1;
    while (!stack.empty()) {
        const auto [x, y] = stack.back();
        stack.pop_back();
        for (const auto& d : kDirs4) {
            const int nx = x + d[0], ny = y + d[1];
            if (mask.inside(nx, ny) && mask.at(nx, ny) && !seen[size_t(ny) * w + nx]) {
                seen[size_t(ny) * w + nx] = 1;
                stack.push_back({nx, ny});
            }
        }
    }
    for (const Point& p : spawns)
        if (!seen[size_t(p.second) * w + p.first]) return false;
    return true;
}

}  // namespace om::gen
