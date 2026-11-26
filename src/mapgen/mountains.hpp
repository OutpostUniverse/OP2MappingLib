// Mountains, gorges, plateaus and valleys (Python: mapgen_berge.py).
//
// Four zones, told apart by whether you can build there:
//
//     mountain   no building ground   tight terraces, outer wall closed
//     gorge      no building ground   narrow cut through a mountain, passable
//     plateau    building ground      one terrace, large area, two ramps
//     valley     building ground      wide cut through the mountain chain
//
// planMountains fixes the shapes only (masks), drawMountains covers them with
// walls through BlendMap::drawCliff. The map is a cutout of a larger world:
// planning happens on an area reaching kWorldMargin cells further on every
// side, so mountains and walls run off the map as in the original maps.
//
// Ported from fefi-byte's op2-codegen-editor (MIT, see THIRD-PARTY-NOTICES.md).
#pragma once

#include "mapgen.hpp"

#include "op2maplib/op2maplib.h"

namespace om::gen {

/// Cells the world reaches beyond the map on every side (WELT_RAND).
constexpr int kWorldMargin = 32;

/// A gorge or valley (Python: Einschnitt): centre strip, area between the
/// walls and its two ends.
struct Cut {
    Mask sole;      ///< centre strip (sohle)
    Mask area;      ///< between the walls (flaeche)
    Mask endA, endB;
    Floats along;   ///< position along the cut (laengs)
};

/// Zone shapes, no tiles yet (Python: BergPlan).
struct MountainPlan {
    std::vector<Mask> terraces;   ///< outside -> inside; terraces[0] is the mountain area
    std::vector<Mask> plateaus;
    std::vector<Cut> gorges, valleys;

    bool empty() const { return terraces.empty() && plateaus.empty(); }
    /// Mountains and plateaus (alles), empty mask if none.
    Mask all(int w, int h) const;
    /// Area that has to be one ground so the walls fit (saum).
    Mask seam(int w, int h) const;
};

/// berge_gewuenscht
inline bool mountainsWanted(const Params& p) { return p.mountains > 0 || p.plateaus > 0; }

/// berg_plan: shapes of all zones of the map. `blocked`: map-sized mask of
/// cells that stay free (volcanoes, clearings).
MountainPlan planMountains(const Params& p, const Mask& blocked);

/// Tiles the walls are drawn into: mapping indices into the profile's
/// standard table plus the cell types the generator keeps.
struct WallCanvas {
    const Profile& profile;
    const std::vector<om_mapping>& table;
    const std::vector<uint32_t>& packedOfMapping;   ///< mapping -> tileset * 4096 + image
    const std::vector<int>& cellOfMapping;          ///< mapping -> cell type (-1 unknown)
    Grid<int32_t>& tiles;
    Grid<int32_t>& cells;
};

/// What berge_zeichnen and zonen_bericht found.
struct MountainReport {
    int mountains = 0, leaky = 0, plateaus = 0, unreachable = 0;
    double build = 0;                       ///< share of the map that is building ground
    double buildMountain = -1, buildPlateau = -1, buildValley = -1, buildGorge = -1;
    int gorges = 0, gorgesPassable = 0;
};

/// berge_zeichnen: draws all walls of the plan into the canvas. Returns the
/// wall mask (cells whose new tile is a cliff).
Mask drawMountains(WallCanvas& c, const MountainPlan& plan, const Params& p, MountainReport& report);

/// zonen_bericht: building ground per zone, gorges passable.
void zoneReport(const Grid<uint8_t>& cellTypes, const MountainPlan& plan, MountainReport& report);

/// bauplatz_maske: cells covered by a free, reachable 7x7 square.
Mask buildingGround(const Grid<uint8_t>& cellTypes);

}  // namespace om::gen
