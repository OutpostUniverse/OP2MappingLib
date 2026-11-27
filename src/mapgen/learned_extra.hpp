// Learned map generator, part 3: the additions of op2-codegen-editor after
// v0.11.1 (Python: mapgen_netz_gross.py, mapgen_netz_fels.py,
// mapgen_netz_deko.py and parts of mapgen_netz.py):
//   - large shape ("coherence"): a height plan of the whole map whose contours
//     go into the sketch network as fixed cliffs, mountain terraces, hills;
//   - calmer biomes and terrain, the gradient across the map;
//   - lava rock: lava cores in rock areas and the rock tile patterns;
//   - volcanoes in three sizes, craters, cracks and debris.
// Ported from fefi-byte's op2-codegen-editor (MIT, see THIRD-PARTY-NOTICES.md).
#pragma once

#include "learned.hpp"

#include <initializer_list>
#include <utility>

namespace om::gen::extra {

// --- noise and masks -------------------------------------------------------

/// _weich: triple box filter of radius r (edge padded), almost a bell.
Floats blur3(const Floats& f, int r);
/// rauschen: soft noise, sum of several sizes (radius, weight), spread 1.
Floats softNoise(int w, int h, Rng& rng, std::initializer_list<std::pair<int, double>> radii);
/// numpy.quantile (linear).
double quantileOf(std::vector<double> v, double q);
/// _flaechen: 8-connected areas of at least `minCells` cells.
Mask bigAreas(const Mask& m, int minCells);
/// ohne_schnipsel: wall pieces (1..8) under `shortest` cells become `empty`.
Grid<int32_t> dropShortWalls(const Grid<int32_t>& wall, int shortest, int empty);

// --- large shape (mapgen_netz_gross.py) ------------------------------------

/// grossform: wall preset per sketch cell, 9 = open, else the fall direction.
/// `rough`: the gradient's terrain per cell (nullptr without a gradient).
Grid<int32_t> largeShape(const Grid<int32_t>& raster, double strength, Rng& rng, double density, double calm,
                         const Floats* rough);

// --- grounds -------------------------------------------------------------

/// _biome_beruhigen: merges grounds into large areas, walls are barriers.
/// `pref`: per ground 0..3 a local preference (gradient), may be null.
Grid<int32_t> calmBiomes(const Grid<int32_t>& wall, const Grid<int32_t>& ground, double calm, Rng& rng,
                         const std::vector<Floats>* pref);
/// _kleine_flecken_weg: patches under `size` cells dissolve into the neighbouring ground.
Grid<int32_t> dropSmallPatches(const Grid<int32_t>& ground, const Grid<int32_t>& wall, double size);
/// schmaler_saum: walls and their 4x4 blocks on dirt or sand, nothing more.
Grid<int32_t> narrowRim(const Grid<int32_t>& wall, const Grid<int32_t>& ground);

// --- rock and lava (mapgen_netz_fels.py) ---------------------------------

enum Kind : int8_t { kOther = 0, kRock = 1, kLava = 2, kSeam = 3 };
/// kachelart: per standard mapping index, rock, lava rock, transition or other.
int8_t tileKind(int mapping);
/// The five base patterns (rocbas1..5): first mapping, last mapping, pattern width.
struct Pattern { int first, last, width; bool lava; double share; };
extern const Pattern kPatterns[5];
/// lavaplan: where lava cores lie in rock, on the sketch raster.
Mask lavaPlan(const Grid<int32_t>& ground, const Grid<int32_t>& wall, double amount, Rng& rng);
/// zellplan: per tile rock, lava or seam (either) inside rock areas.
Grid<int8_t> cellPlan(const Mask& lava, const Grid<int32_t>& ground);
/// flaechen: where each base pattern lies (connected areas instead of dice).
std::vector<Mask> patternAreas(int w, int h, Rng& rng);

/// A reserved volcano site on the sketch raster.
struct VolcanoSite {
    int y, x;      ///< sketch cell
    int size;      ///< 0 small, 1 medium, 2 large
    bool lava;     ///< on lava rock (medium and large) or rock (small)
    int r;         ///< radius of its room in sketch cells
};
/// vulkanplaetze: picks the sites and turns the ground there to rock. Only cells
/// inside [x0, x1) x [y0, y1) and, if given, `allowed`.
std::vector<VolcanoSite> volcanoSites(Grid<int32_t>& ground, const Grid<int32_t>& wall, const int counts[3], int x0,
                                      int y0, int x1, int y1, Rng& rng, const Mask* allowed);

}  // namespace om::gen::extra
