// Map generator entry point (Python: mapgen.generate / GeneratedMap).
#pragma once

#include "mapgen.hpp"
#include "mountains.hpp"

namespace om { class Profile; }

namespace om::gen {

/// Where a feature actually ended up (Python: GeneratedMap.feature_spots).
struct FeatureSpot {
    FeatKind kind;
    int x, y;
    int size;   ///< radius (mountain, plateau, clearing) or width (valley, choke)
};

/// A generated map (Python: GeneratedMap).
struct Result {
    Grid<int32_t> tiles;                       ///< mapping index per cell, into the profile's standard mapping table
    Grid<uint8_t> cells;                       ///< Outpost 2 cell type per cell
    std::vector<Point> spawns;
    std::vector<Point> volcanoes;
    std::vector<std::vector<Point>> lavaZones; ///< lava-possible cells, per volcano or lava valley
    double passableShare = 0, plateauShare = 0, detailShare = 0, riverShare = 0;
    bool connected = true;                     ///< all spawns reachable from each other
    std::vector<double> familyShare;           ///< per profile ground
    std::array<int, 5> featureReport{};        ///< per FeatKind: cliff cells (clearings: count)
    std::vector<FeatureSpot> featureSpots;
    MountainReport zones;                      ///< mountains only: walls and building ground per zone
};

/// Generates a map (same profile, params and seed = same map). Throws
/// std::runtime_error for unusable input (bad size, profile without a
/// standard mapping table).
Result generate(const Profile& profile, const Params& p);

}  // namespace om::gen
