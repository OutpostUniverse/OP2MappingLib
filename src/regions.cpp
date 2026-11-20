// Passable regions: connected components of passable cells, largest first.
//
// Moved here from OP2Mapper's "Find & Fix Enclosed Areas" (fMain.MapAnalysis.vb)
// so every map tool shares one implementation. Same rules as the original:
// breadth-first flood fill, 8-connected, the bottom row skipped, and the left
// and right edges joined only on the 512x256 world map. Unlike the original,
// equal-sized regions are ordered by their first cell so the result never
// depends on sort stability.
//
// Independent of the blend and cliff code: it needs only the public header.
#include "op2maplib/op2maplib.h"

#include <algorithm>
#include <deque>
#include <new>
#include <vector>

// The opaque handle behind om_regions_find. Owned by the caller, freed with
// om_regions_free; it holds its own copy of the result, so the cell_types
// buffer can be released as soon as om_regions_find returns.
struct om_regions {
    int width = 0, height = 0;
    std::vector<int> label;                     ///< per cell: region index or -1
    std::vector<std::vector<int>> cells;        ///< per region: cell indices (y*width + x), in BFS order
};

namespace {

// Outpost 2 cell types that units can cross.
constexpr uint32_t kOp2Passable =
    (1u << 0) |                                   // FastPassible1
    (1u << 2) | (1u << 3) |                       // SlowPassible1, SlowPassible2
    (1u << 4) | (1u << 5) |                       // MediumPassible1, MediumPassible2
    (1u << 7) |                                   // FastPassible2
    (1u << 21) | (1u << 22) |                     // DozedArea, Rubble
    (1u << 26) | (1u << 27) | (1u << 28) | (1u << 29) | (1u << 30) | (1u << 31);   // Tube0-5

// Cell types of 32 and above are never passable (they have no bit in the mask).
bool passable(uint32_t mask, uint8_t cellType) {
    return cellType < 32 && (mask & (1u << cellType)) != 0;
}

}  // namespace

extern "C" {

// Only the 512x256 world map wraps around; other sizes have hard edges.
OM_API uint32_t OM_CALL om_regions_op2_flags(int32_t width, int32_t height) {
    uint32_t f = OM_REGIONS_CONNECT8 | OM_REGIONS_SKIP_BOTTOM_ROW;
    if (width == 512 && height == 256) f |= OM_REGIONS_WRAP_X;
    return f;
}

OM_API uint32_t OM_CALL om_cell_types_passable_mask(void) { return kOp2Passable; }

// Labels the regions with a breadth-first flood fill and sorts them largest
// first. Returns NULL for invalid input or when memory runs out.
OM_API om_regions* OM_CALL om_regions_find(int32_t width, int32_t height, const uint8_t* cellTypes,
                                           uint32_t passableMask, uint32_t flags) {
    if (width <= 0 || height <= 0 || !cellTypes) return nullptr;
    if (passableMask == 0) passableMask = kOp2Passable;
    const bool connect8 = (flags & OM_REGIONS_CONNECT8) != 0;
    const bool wrapX = (flags & OM_REGIONS_WRAP_X) != 0;
    // Last row searched (inclusive).
    const int maxY = (flags & OM_REGIONS_SKIP_BOTTOM_ROW) ? height - 2 : height - 1;

    om_regions* r = new (std::nothrow) om_regions;
    if (!r) return nullptr;
    try {
        r->width = width;
        r->height = height;
        r->label.assign(static_cast<size_t>(width) * height, -1);
        std::vector<uint8_t> visited(r->label.size(), 0);

        static const int d8x[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        static const int d8y[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        static const int d4x[4] = {0, 1, 0, -1};
        static const int d4y[4] = {-1, 0, 1, 0};
        const int* dx = connect8 ? d8x : d4x;
        const int* dy = connect8 ? d8y : d4y;
        const int dirs = connect8 ? 8 : 4;

        std::deque<int> queue;
        for (int y = 0; y <= maxY; ++y) {
            for (int x = 0; x < width; ++x) {
                const int start = y * width + x;
                // Cells are marked visited when queued, passable or not, so each
                // cell is looked at once; blocked cells simply start no region.
                if (visited[start]) continue;
                visited[start] = 1;
                if (!passable(passableMask, cellTypes[start])) continue;

                std::vector<int> region;
                queue.push_back(start);
                while (!queue.empty()) {
                    const int c = queue.front();
                    queue.pop_front();
                    region.push_back(c);
                    const int cx = c % width, cy = c / width;
                    for (int k = 0; k < dirs; ++k) {
                        int nx = cx + dx[k];
                        const int ny = cy + dy[k];
                        if (wrapX) nx = ((nx % width) + width) % width;
                        else if (nx < 0 || nx >= width) continue;
                        if (ny < 0 || ny > maxY) continue;
                        const int n = ny * width + nx;
                        if (visited[n]) continue;
                        visited[n] = 1;
                        if (passable(passableMask, cellTypes[n])) queue.push_back(n);
                    }
                }
                r->cells.push_back(std::move(region));
            }
        }

        // Regions were found in row order of their first cell, so a stable sort
        // by size alone gives the documented tie order.
        // Largest first; equal sizes keep their discovery order (first cell, row by row).
        std::vector<size_t> order(r->cells.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return r->cells[a].size() > r->cells[b].size(); });
        std::vector<std::vector<int>> sorted;
        sorted.reserve(order.size());
        for (size_t i : order) sorted.push_back(std::move(r->cells[i]));
        r->cells = std::move(sorted);
        for (size_t i = 0; i < r->cells.size(); ++i)
            for (int c : r->cells[i]) r->label[c] = static_cast<int>(i);
        return r;
    } catch (...) {
        delete r;
        return nullptr;
    }
}

OM_API void OM_CALL om_regions_free(om_regions* regions) { delete regions; }

// Queries. Region 0 is the main area; regions 1.. are enclosed.
OM_API int OM_CALL om_regions_count(const om_regions* regions) {
    return regions ? static_cast<int>(regions->cells.size()) : 0;
}

OM_API int OM_CALL om_regions_size(const om_regions* regions, int region) {
    if (!regions || region < 0 || region >= static_cast<int>(regions->cells.size())) return 0;
    return static_cast<int>(regions->cells[region].size());
}

// Cells come in flood-fill order (starting from the region's first cell).
OM_API int OM_CALL om_regions_cells(const om_regions* regions, int region, int32_t* xy, int32_t max) {
    if (!regions || !xy || max <= 0 || region < 0 || region >= static_cast<int>(regions->cells.size())) return 0;
    const std::vector<int>& cells = regions->cells[region];
    const int n = std::min<int>(max, static_cast<int>(cells.size()));
    for (int i = 0; i < n; ++i) {
        xy[2 * i] = cells[i] % regions->width;
        xy[2 * i + 1] = cells[i] / regions->width;
    }
    return n;
}

OM_API int OM_CALL om_regions_at(const om_regions* regions, int32_t x, int32_t y) {
    if (!regions || x < 0 || y < 0 || x >= regions->width || y >= regions->height) return -1;
    return regions->label[static_cast<size_t>(y) * regions->width + x];
}

OM_API int OM_CALL om_regions_enclosed_size(const om_regions* regions) {
    if (!regions) return 0;
    int n = 0;
    for (size_t i = 1; i < regions->cells.size(); ++i) n += static_cast<int>(regions->cells[i].size());
    return n;
}

// Regions 1.. in order, concatenated, stopping once max cells are written.
OM_API int OM_CALL om_regions_enclosed_cells(const om_regions* regions, int32_t* xy, int32_t max) {
    if (!regions || !xy || max <= 0) return 0;
    int n = 0;
    for (size_t i = 1; i < regions->cells.size() && n < max; ++i)
        for (int c : regions->cells[i]) {
            if (n >= max) break;
            xy[2 * n] = c % regions->width;
            xy[2 * n + 1] = c / regions->width;
            ++n;
        }
    return n;
}

}  // extern "C"
