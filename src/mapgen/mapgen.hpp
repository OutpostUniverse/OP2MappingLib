// Map generator - internal C++ types and functions behind om_mapgen_*.
//
// Port of fefi-byte's map generator from op2-codegen-editor (mapview/mapgen*.py,
// shapegen.py; MIT, github.com/fefi-byte/op2-codegen-editor). The design is
// fefi-byte's; this is a C++ port of it. The layout layer (biome field, relief, features, volcano
// sites, spawn points) is ported here; the tile layer reuses this library's
// own engines: BlendMap::paint/resolve picks ground and transition tiles, and
// BlendMap::drawCliff draws the cliffs along the highland contours.
//
// Conventions
//   - Grids are row-major, index y * w + x, like the rest of the library.
//   - "Family" means a profile ground index (Profile::grounds); the Outpost 2
//     profile's grounds are lav, mud, roc, san, sno, veg - the same order the
//     Python generator's families had.
//   - All randomness goes through om::Rng (rng.hpp), never <random>, so a seed
//     gives the same map on every platform. Results do NOT reproduce the
//     Python generator's maps for the same seed (different generator and a
//     different tile layer); the shapes and behaviour are what is ported.
//   - Python names are kept in comments next to each function so the original
//     can be found (German identifiers: Tal = valley, Berg = mountain,
//     Frei = clearing, Enge = choke, Hoehenfeld = height field, Rand = edge,
//     Pfad = path, Breite = width).
#pragma once

#include "../rng.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace om {
class Profile;
class BlendMap;
}

namespace om::gen {

// ---------------------------------------------------------------------------
// Grid

/// Dense 2-d grid, row-major.
template <class T>
struct Grid {
    int w = 0, h = 0;
    std::vector<T> v;

    Grid() = default;
    Grid(int width, int height, T init = T{}) : w(width), h(height), v(size_t(width) * height, init) {}

    T& at(int x, int y) { return v[size_t(y) * w + x]; }
    const T& at(int x, int y) const { return v[size_t(y) * w + x]; }
    bool inside(int x, int y) const { return x >= 0 && y >= 0 && x < w && y < h; }
    /// Value at (x, y), or `outside` for coordinates off the grid.
    T get(int x, int y, T outside) const { return inside(x, y) ? at(x, y) : outside; }
    bool any() const {
        for (const T& t : v) if (t != T{}) return true;
        return false;
    }
    size_t count() const {
        size_t n = 0;
        for (const T& t : v) if (t != T{}) ++n;
        return n;
    }
    void fill(T t) { std::fill(v.begin(), v.end(), t); }
};

using Mask = Grid<uint8_t>;     ///< 0 / 1
using Field = Grid<int16_t>;    ///< family (profile ground) per cell
using Floats = Grid<double>;

using Point = std::pair<int, int>;   ///< (x, y)
using Path = std::vector<Point>;

// ---------------------------------------------------------------------------
// Parameters (Python: MapGenParams, Feature, TEMPLATES)

enum class Sym { None, MirrorX, MirrorY, Rot180, Quad };   // SYM_NONE, SYM_MIRROR_X, ...

enum class FeatKind { Valley, Mountain, Plateau, Clearing, Choke };   // tal, berg, plateau, frei, enge

/// A requested terrain feature (Python: Feature).
struct Feature {
    FeatKind kind = FeatKind::Mountain;
    std::string where;   ///< sector "nw","n","no","w","c","o","sw","s","so", or "" = seed decides
    std::string to;      ///< valley/choke target sector, "" = opposite of `where`
    double size = 0.5;   ///< 0..1
    bool lava = false;   ///< valley only: lava-possible floor plus a volcano
};

/// Generator settings (Python: MapGenParams). Defaults match the Python ones.
struct Params {
    int width = 128, height = 128;
    uint64_t seed = 1;
    /// Share per ground id ("mud", "roc", "san", "veg", "sno", "lav"); need not sum to 1.
    std::vector<std::pair<std::string, double>> mix{{"mud", 0.45}, {"roc", 0.25}, {"san", 0.15}, {"veg", 0.15}};
    double roughness = 0.0;     ///< 0 soft large areas .. 1 rugged
    Sym symmetry = Sym::None;
    int spawns = 0;             ///< spawn points (0 = none)
    int spawnClearance = 6;     ///< free radius per spawn point, in tiles
    double cliffs = 0.0;        ///< plateau share (0 = none)
    int ramps = 2;              ///< ramps (gaps in the cliff ring) per map
    int volcanoes = 0;
    int lavaZoneTiles = 300;    ///< lava-possible area per volcano
    int smooth = 2;             ///< majority smoothing passes
    int vegetationSpacing = 1;  ///< bewuchs_abstand
    double minBiome = 0.006;    ///< min_biome: smallest biome patch as share of the map
    double details = 0.5;       ///< scatter density (0.5 ~ original maps)
    std::vector<Feature> features;
    bool landscape = false;     ///< landschaft (New Terra relief mode)
    int rivers = -1;            ///< fluesse: dry riverbeds, -1 = by map size, 0 = none
    // Mountains and building ground (mountains.cpp, Python: mapgen_berge.py)
    double mountains = 0.0;         ///< berge: area share of the mountain chains (0 = none)
    double mountainDensity = 0.8;   ///< berg_dichte: how tight the terraces stand (1 = as tight as the walls allow)
    int gorges = 1;                 ///< schluchten: narrow passable cuts through a mountain
    int valleys = 1;                ///< taeler: wide cuts with building ground
    int plateaus = 0;               ///< hochebenen: one-terrace areas with building ground and ramps
    bool mountainSpread = false;    ///< grossform: more, more varied mountains spread over the whole world
    bool learnedRocks = true;       ///< berg_felsen, learned generator only: rocks and craters where the
                                    ///< walls alone leave building ground in mountains and gorges
    bool worldCutout = false;       ///< classic cliffs: plan a larger world and take the middle,
                                    ///< so cliffs run off the map edge as in the original maps
};

/// Fills `p` with a template's mix, roughness, landscape and volcano settings
/// (Python: MapGenParams.from_template). Unknown names give "plains".
/// Names: plains, canyon, desert, islands, ice, newterra.
void applyTemplate(Params& p, const std::string& name);

/// Concrete geometry of one feature (Python: one `layout` dict entry).
struct Placement {
    Feature feat;
    int x = 0, y = 0;
    int radius = 0;   ///< mountain, plateau, clearing ("radius")
    int width = 0;    ///< valley, choke ("breite")
    Path path;        ///< valley, choke ("pfad")
};

// ---------------------------------------------------------------------------
// field.cpp - biome field (Python: mapgen_felder.py, _bewuchs_abstand)

/// _value_noise: smooth noise in [0, 1) by bilinear interpolation of a random lattice.
Floats valueNoise(Rng& rng, int h, int w, double freq);
/// _family_field: which family each cell gets. `groundIds` are the profile's ground ids.
Field familyField(const Params& p, Rng& rng, const std::vector<std::string>& groundIds);
/// _smooth_families: majority vote over the (2r+1)^2 neighbourhood, `passes` times.
Field smoothFamilies(const Field& f, int passes, int radius = 2);
/// _components: 4-connected components of a mask, each as a list of cell indices (y*w+x).
std::vector<std::vector<int>> components(const Mask& m);
/// _dissolve_small_biomes: patches smaller than minCells take the surrounding family.
void dissolveSmallBiomes(Field& f, int minCells);
/// _apply_symmetry: mirror the base half/quarter onto the rest (works for any grid type).
template <class T> void applySymmetry(Grid<T>& g, Sym s);
/// _mirror_region: copy the tile grid's base half/quarter onto the mirrored part.
void mirrorRegion(Grid<int32_t>& tiles, Sym s);
/// _bewuchs_abstand: vegetation closer than `spacing` to a foreign ground becomes mud.
int vegetationSpacing(Field& f, int veg, int mud, int spacing);
/// _volcano_sites: volcano centres (mirrored per symmetry); sets the surroundings to rock.
std::vector<Point> volcanoSites(Field& f, const Params& p, Rng& rng, int rockFamily);
/// _lava_zone: lava-possible cells around a volcano.
std::vector<Point> lavaZone(Point centre, int w, int h, int tiles, Rng& rng);

// ---------------------------------------------------------------------------
// shape.cpp - height-field shapes (Python: shapegen.py)

Mask smoothMask(const Mask& m, int passes = 2);                     // glaette
Mask snapEdge(const Mask& m, int block = 3);                        // begradige
Mask blob(int w, int h, Point centre, int radius, Rng& rng, double roughness = 0.45);        // blob
Mask corridor(int w, int h, const Path& path, int width, Rng& rng, double roughness = 0.35); // korridor
Mask barrier(int w, int h, const Path& path, int gapAt, int gap, Rng& rng);                 // riegel
Mask removeSmallBits(const Mask& m, int minCells = 24);             // entferne_kleinkram

// ---------------------------------------------------------------------------
// paths.cpp - passability and spawn points (Python: mapgen_wege.py)

/// _passable_mask: cell types vehicles can cross (fast, medium, slow).
Mask passableMask(const Grid<uint8_t>& cellTypes);
/// _clearance: share of passable cells in the (2r+1)^2 square around each cell.
Floats clearance(const Mask& passable, int radius);
/// _groesstes_gebiet: largest 4-connected area of a mask.
Mask largestArea(const Mask& m);
/// _pick_spawns: free, far-apart start positions.
std::vector<Point> pickSpawns(const Grid<uint8_t>& cellTypes, int count, int clearance, Sym s, Rng& rng);
/// _connected: all spawns reachable from each other over passable cells.
bool connected(const Grid<uint8_t>& cellTypes, const std::vector<Point>& spawns);

// ---------------------------------------------------------------------------
// relief.cpp - height field, landscape mode, contours, features (Python: mapgen_relief.py)

Mask heightField(const Params& p, Rng& rng, const std::vector<Placement>& layout);   // _hoehenfeld
Path riverPath(Point start, Point goal, int w, int h, Rng& rng);                     // _fluss_pfad
Mask pathBand(int w, int h, const Path& pts, double radius);                         // _pfad_band
Mask dilate4(const Mask& m, int steps);                                              // _dehne
/// _landschaft_hoehenfeld -> (height field, riverbeds).
std::pair<Mask, Mask> landscapeHeightField(const Params& p, Rng& rng);
/// _landschaft_familien: one family per wall (plateau plus rim), riverbeds sand.
void landscapeFamilies(Field& f, const Mask& H, const Mask& beds, int mud, int sand);
/// _rand_zyklus: ordered boundary cycle of one component (Moore tracing).
Path boundaryCycle(const Mask& comp);
/// _kontur_vereinfachen: Ramer-Douglas-Peucker.
Path simplifyContour(const Path& pts, double eps = 1.0);
/// _kontur_hand: is the highland left (true) or right (false) of the walking direction?
bool contourHighLeft(const Path& pts, const Mask& comp);
/// _kontur_boegen: split a cycle into open arcs (map edge removed, ramp gaps inserted).
std::vector<Path> contourArcs(const Path& cycle, const Mask& comp, int gaps, Rng& rng, int gapLength = 9);
Point sectorPoint(const std::string& where, int w, int h, Rng& rng, Sym s);           // _sektor_punkt
std::vector<Placement> featureLayout(const Params& p, Rng& rng);                     // _feature_layout
Path featurePath(Point start, Point end, Rng& rng);                                  // _merkmal_pfad
void steerFeatureFamilies(Field& f, const std::vector<Placement>& layout, int mud, int sand);   // _steer_feature_families
Mask featureMask(const Placement& e, int w, int h, int margin = 0);                  // _feature_maske

}  // namespace om::gen
