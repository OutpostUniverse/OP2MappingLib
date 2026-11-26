// The map generator's tile canvas and its stamp helpers (generate.cpp), shared
// by the classic and the learned generator.
#pragma once

#include "mapgen.hpp"

#include "../blend.hpp"

#include <unordered_map>

namespace om::gen {

constexpr int kMediumPassable1 = 4;   // celltypes.MEDIUM_PASSIBLE_1: fallback cell type

/// The generator's view of the map: tiles as standard mapping indices, cell
/// types, and the BlendMap that picks ground tiles.
struct Canvas {
    const Profile& profile;
    int w, h;
    Grid<int32_t> tiles;          ///< standard mapping index per cell
    Grid<int32_t> cells;          ///< cell type per cell
    std::vector<int> cellOfMapping;              ///< standard mapping -> cell type (-1 unknown)
    std::vector<uint32_t> packedOfMapping;       ///< standard mapping -> tileset * 4096 + image
    std::unordered_map<uint32_t, int> mappingOfPacked;

    Canvas(const Profile& p, int width, int height)
        : profile(p), w(width), h(height), tiles(width, height, 0), cells(width, height, kMediumPassable1) {
        const auto& sm = p.standardMapping;
        cellOfMapping.assign(sm.size(), -1);
        packedOfMapping.resize(sm.size());
        for (size_t m = 0; m < sm.size(); ++m) {
            const int t = p.findTile(sm[m].first, sm[m].second);
            if (t >= 0) cellOfMapping[m] = p.tiles[t].cellType;
            const uint32_t k = GeneratorData::pack(sm[m].first, sm[m].second);
            packedOfMapping[m] = k;
            mappingOfPacked.emplace(k, static_cast<int>(m));   // first wins
        }
    }

    /// Writes one tile with a known cell type (-1: the table's, else the fallback).
    void put(int x, int y, int mapping, int cellType) {
        tiles.at(x, y) = mapping;
        if (cellType < 0 && mapping >= 0 && mapping < static_cast<int>(cellOfMapping.size()))
            cellType = cellOfMapping[mapping];
        cells.at(x, y) = cellType >= 0 ? cellType : kMediumPassable1;
    }
};

/// Standard mapping index of a stamp tile, or -1 if the table lacks it.
int stampMapping(const Canvas& c, const StampTile& t);
/// Writes a stamp at (x0, y0) into canvas and blend map. False (and nothing
/// written) if one of its tiles is missing from the mapping table.
bool placeStamp(Canvas& c, BlendMap& bm, const Stamp& s, int x0, int y0);
/// _naht_passt: do the stamp's borders fit the tiles around it?
bool seamFits(const Canvas& c, const Stamp& s, int x, int y);
/// _scatter_details: rocks, cracks and craters in whole stamps on matching
/// ground, until `target` cells are covered. Returns the cells placed.
int scatterDetails(Canvas& c, BlendMap& bm, const Field& fam, Rng& rng, int target, Mask& blocked, Sym sym);
/// The profile's standard mapping table as om_mapping entries.
std::vector<om_mapping> standardTable(const Profile& profile);

}  // namespace om::gen
