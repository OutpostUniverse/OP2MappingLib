// Map generator pipeline (Python: mapgen.generate) plus the stamp steps
// (Python: _scatter_details, _stamp_volcanoes, _naht_passt in mapgen_felder.py),
// cliffs from contours (Python: _klippen_aus_linien in mapgen_klippen.py) and
// the feature report (Python: _apply_features in mapgen_relief.py).
//
// The layout steps run as in the Python. The tile steps use this library's
// engines instead of the Python tile sampler:
//   - grounds and transitions: a BlendMap over the profile's standard mapping
//     table; every cell is painted with its family and the map is resolved;
//   - cliffs: each highland contour arc goes through BlendMap::drawCliff;
//   - stamps (details, volcanoes) come from the profile's "generator" section.
#include "generate.hpp"

#include "../blend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace om::gen {

namespace {

constexpr int kMediumPassable1 = 4;   // celltypes.MEDIUM_PASSIBLE_1: fallback cell type
constexpr int kNorthCliffs = 8, kCliffsHighSide = 9, kCliffsLowSide = 10;
constexpr double kWallHeight = 3.0;   // HOEHE_MIN/HOEHE_MAX 1.5/2.5 = a 3-cell wall (dicke_aus_hoehe)

bool isCliff(int ct) { return ct == kNorthCliffs || ct == kCliffsHighSide || ct == kCliffsLowSide; }

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
int stampMapping(const Canvas& c, const StampTile& t) {
    auto it = c.mappingOfPacked.find(GeneratorData::pack(t.tileset, t.image));
    return it == c.mappingOfPacked.end() ? -1 : it->second;
}

/// Writes a stamp at (x0, y0) into canvas and blend map. False (and nothing
/// written) if one of its tiles is missing from the mapping table.
bool placeStamp(Canvas& c, BlendMap& bm, const Stamp& s, int x0, int y0) {
    std::vector<int> maps(s.tiles.size());
    for (size_t i = 0; i < s.tiles.size(); ++i)
        if ((maps[i] = stampMapping(c, s.tiles[i])) < 0) return false;
    for (int j = 0; j < s.h; ++j)
        for (int i = 0; i < s.w; ++i) {
            const size_t k = static_cast<size_t>(j) * s.w + i;
            c.put(x0 + i, y0 + j, maps[k], s.tiles[k].cellType);
            bm.setTile(x0 + i, y0 + j, maps[k]);
        }
    return true;
}

/// _naht_passt: do the stamp's borders fit the tiles around it? Pairs whose
/// first tile has no recorded neighbours are not checked.
bool seamFits(const Canvas& c, const Stamp& s, int x, int y) {
    const GeneratorData& g = c.profile.generator;
    auto packedAt = [&](int cx, int cy) { return c.packedOfMapping[c.tiles.at(cx, cy)]; };
    auto packedOf = [&](int i, int j) {
        const StampTile& t = s.tiles[static_cast<size_t>(j) * s.w + i];
        return GeneratorData::pack(t.tileset, t.image);
    };
    auto allowed = [&](bool horizontal, uint32_t a, uint32_t b) {
        return !g.known(horizontal, a) || g.seen(horizontal, a, b);
    };
    for (int i = 0; i < s.w; ++i) {
        if (y > 0 && !allowed(false, packedAt(x + i, y - 1), packedOf(i, 0))) return false;
        if (y + s.h < c.h && !allowed(false, packedOf(i, s.h - 1), packedAt(x + i, y + s.h))) return false;
    }
    for (int j = 0; j < s.h; ++j) {
        if (x > 0 && !allowed(true, packedAt(x - 1, y + j), packedOf(0, j))) return false;
        if (x + s.w < c.w && !allowed(true, packedOf(s.w - 1, j), packedAt(x + s.w, y + j))) return false;
    }
    return true;
}

/// _scatter_details: rocks, cracks and craters in whole stamps on matching
/// ground, until `target` cells are covered. Returns the cells placed.
int scatterDetails(Canvas& c, BlendMap& bm, const Field& fam, Rng& rng, int target, Mask& blocked, Sym sym) {
    const auto& details = c.profile.generator.details;
    if (target <= 0 || details.empty()) return 0;
    // Under symmetry most random positions fall outside the base part; allow
    // more tries so the density stays about the same.
    const int images = sym == Sym::None ? 1 : sym == Sym::Quad ? 4 : 2;
    int placed = 0, tries = 0;
    while (placed < target && tries < (target * 6 + 200) * images) {
        ++tries;
        const Stamp& s = details[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(details.size())))];
        if (s.w + 2 >= c.w || s.h + 2 >= c.h) continue;
        const int x = static_cast<int>(rng.integers(1, c.w - s.w - 1));
        const int y = static_cast<int>(rng.integers(1, c.h - s.h - 1));
        // With symmetry, place in the base half/quarter and repeat at the
        // mirrored positions (the stamp itself cannot be flipped).
        const bool leftHalf = x + s.w <= c.w / 2, topHalf = y + s.h <= c.h / 2;
        if ((sym == Sym::MirrorX && !leftHalf) || ((sym == Sym::MirrorY || sym == Sym::Rot180) && !topHalf) ||
            (sym == Sym::Quad && !(leftHalf && topHalf)))
            continue;
        std::vector<Point> spots{{x, y}};
        const int mx = c.w - x - s.w, my = c.h - y - s.h;
        if (sym == Sym::MirrorX) spots.emplace_back(mx, y);
        if (sym == Sym::MirrorY) spots.emplace_back(x, my);
        if (sym == Sym::Rot180) spots.emplace_back(mx, my);
        if (sym == Sym::Quad) { spots.emplace_back(mx, y); spots.emplace_back(x, my); spots.emplace_back(mx, my); }
        bool ok = true;
        for (const auto& [sx, sy] : spots) {
            for (int j = 0; j < s.h && ok; ++j)
                for (int i = 0; i < s.w && ok; ++i) ok = !blocked.at(sx + i, sy + j);
            if (ok && !seamFits(c, s, sx, sy)) ok = false;
            if (ok && s.ground >= 0) {
                // At least 75 % of the cells under the stamp must be its ground.
                int same = 0;
                for (int j = 0; j < s.h; ++j)
                    for (int i = 0; i < s.w; ++i) same += fam.at(sx + i, sy + j) == s.ground;
                ok = same >= 0.75 * s.w * s.h;
            }
            if (!ok) break;
        }
        if (!ok) continue;
        for (const auto& [sx, sy] : spots) {
            if (!placeStamp(c, bm, s, sx, sy)) break;
            for (int j = 0; j < s.h; ++j)
                for (int i = 0; i < s.w; ++i) blocked.at(sx + i, sy + j) = 1;
            placed += s.w * s.h;
        }
    }
    return placed;
}

/// _stamp_volcanoes: the largest volcano stamp (the smallest if it does not
/// fit) centred on each site. Returns the placed centres.
std::vector<Point> stampVolcanoes(Canvas& c, BlendMap& bm, const std::vector<Point>& sites) {
    std::vector<Point> placed;
    const auto& vs = c.profile.generator.volcanoes;
    if (vs.empty()) return placed;
    auto area = [](const Stamp& s) { return s.w * s.h; };
    const Stamp* largest = &*std::max_element(vs.begin(), vs.end(), [&](const Stamp& a, const Stamp& b) { return area(a) < area(b); });
    const Stamp* smallest = &*std::min_element(vs.begin(), vs.end(), [&](const Stamp& a, const Stamp& b) { return area(a) < area(b); });
    for (const auto& [cx, cy] : sites) {
        const Stamp* s = (largest->w > c.w || largest->h > c.h) ? smallest : largest;
        if (s->w > c.w || s->h > c.h) continue;
        const int x0 = std::clamp(cx - s->w / 2, 0, c.w - s->w);
        const int y0 = std::clamp(cy - s->h / 2, 0, c.h - s->h);
        if (placeStamp(c, bm, *s, x0, y0)) placed.emplace_back(x0 + s->w / 2, y0 + s->h / 2);
    }
    return placed;
}

/// _klippen_aus_linien: walks every highland component's boundary, splits it
/// into arcs with ramp gaps and draws each arc with the cliff solver.
/// Returns the cliff mask.
Mask cliffsFromContours(Canvas& c, BlendMap& bm, const Mask& H, Rng& rng, int ramps) {
    Mask cliffs(c.w, c.h, 0);
    if (!H.any()) return cliffs;
    auto comps = components(H);
    std::stable_sort(comps.begin(), comps.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
    for (size_t nr = 0; nr < comps.size(); ++nr) {
        Mask comp(c.w, c.h, 0);
        for (int i : comps[nr]) comp.v[static_cast<size_t>(i)] = 1;
        const Path cycle = boundaryCycle(comp);
        if (cycle.size() < 12) continue;
        // The largest area gets the configured ramps, every other one, so no mesa is unreachable.
        const int gaps = nr == 0 ? std::max(ramps, 1) : 1;
        for (const Path& arc : contourArcs(cycle, comp, gaps, rng)) {
            Path pts;
            if (arc.size() > 2 && arc.front() == arc.back()) {
                // Closed figure: simplify the two halves separately (a zero-length chord sees no distance).
                const size_t mid = arc.size() / 2;
                Path a = simplifyContour(Path(arc.begin(), arc.begin() + static_cast<long>(mid) + 1));
                Path b = simplifyContour(Path(arc.begin() + static_cast<long>(mid), arc.end()));
                a.pop_back();
                pts = a;
                pts.insert(pts.end(), b.begin(), b.end());
            } else {
                pts = simplifyContour(arc);
            }
            Path dedup;
            for (const Point& p : pts)
                if (dedup.empty() || dedup.back() != p) dedup.push_back(p);
            if (dedup.size() < 2) continue;
            const int side = contourHighLeft(dedup, comp) ? OM_CLIFF_HIGH_LEFT : OM_CLIFF_HIGH_RIGHT;
            std::vector<int32_t> xy;
            for (const auto& [x, y] : dedup) { xy.push_back(x); xy.push_back(y); }
            const uint64_t seed = static_cast<uint64_t>(rng.integers(0, int64_t(1) << 62));
            if (bm.drawCliff(xy.data(), static_cast<int>(dedup.size()), side, kWallHeight, seed) <= 0) continue;
            for (const om_change& ch : bm.changes()) {
                c.put(ch.x, ch.y, ch.new_mapping, ch.cell_type);
                cliffs.at(ch.x, ch.y) = 1;
            }
        }
    }
    return cliffs;
}

/// Applies the changes of the BlendMap's last resolve to the canvas.
void takeChanges(Canvas& c, const BlendMap& bm) {
    for (const om_change& ch : bm.changes()) c.put(ch.x, ch.y, ch.new_mapping, ch.cell_type);
}

/// Seam pass: in reading order, re-picks each fill or transition tile among
/// the tiles of the same kind (same ground; same ground pair and corners) so
/// it fits its left and upper neighbours, i.e. the pair was seen side by side
/// in the original maps. Grounds, corners and fixed tiles (cliffs, stamps)
/// stay as they are; a tile that already fits is kept. This stands in for the
/// Python sampler, which drew every tile from the learned neighbour table.
void seamPass(Canvas& c, BlendMap& bm, Rng& rng) {
    const Profile& prof = c.profile;
    const GeneratorData& g = prof.generator;
    if (g.right.empty() && g.down.empty()) return;
    // Mapping index of every profile tile in the standard table (-1 missing).
    std::vector<int> mappingOfTile(prof.tiles.size(), -1);
    for (size_t t = 0; t < prof.tiles.size(); ++t) {
        auto it = c.mappingOfPacked.find(GeneratorData::pack(prof.tiles[t].tileset, prof.tiles[t].image));
        if (it != c.mappingOfPacked.end()) mappingOfTile[t] = it->second;
    }
    // Candidate groups: fill tiles per ground, transition tiles per (a, b, corners).
    auto groupKey = [&](const ProfileTile& t) -> int64_t {
        if (t.role == TileRole::Fill) return t.ground;
        if (t.role == TileRole::Transition) return 1000 + (int64_t(t.ground) * 64 + t.groundB) * 16 + t.corners;
        return -1;
    };
    std::unordered_map<int64_t, std::vector<int>> groups;
    for (size_t t = 0; t < prof.tiles.size(); ++t)
        if (mappingOfTile[t] >= 0 && groupKey(prof.tiles[t]) >= 0) groups[groupKey(prof.tiles[t])].push_back(static_cast<int>(t));
    std::vector<int> tileOfMapping(prof.standardMapping.size(), -1);
    for (size_t m = 0; m < prof.standardMapping.size(); ++m)
        tileOfMapping[m] = prof.findTile(prof.standardMapping[m].first, prof.standardMapping[m].second);

    std::vector<int> fits;
    for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
            const int m = c.tiles.at(x, y);
            const int t = (m >= 0 && m < static_cast<int>(tileOfMapping.size())) ? tileOfMapping[m] : -1;
            if (t < 0) continue;
            const int64_t key = groupKey(prof.tiles[t]);
            if (key < 0) continue;
            const bool hasLeft = x > 0, hasTop = y > 0;
            const uint32_t left = hasLeft ? c.packedOfMapping[c.tiles.at(x - 1, y)] : 0;
            const uint32_t top = hasTop ? c.packedOfMapping[c.tiles.at(x, y - 1)] : 0;
            auto fitsHere = [&](int cand) {
                const uint32_t k = c.packedOfMapping[mappingOfTile[cand]];
                if (hasLeft && g.known(true, left) && !g.seen(true, left, k)) return false;
                if (hasTop && g.known(false, top) && !g.seen(false, top, k)) return false;
                return true;
            };
            if (fitsHere(t)) continue;
            fits.clear();
            uint64_t total = 0;
            for (int cand : groups[key])
                if (fitsHere(cand)) {
                    fits.push_back(cand);
                    total += prof.tiles[cand].weight;
                }
            if (fits.empty()) continue;
            uint64_t r = static_cast<uint64_t>(rng.integers(0, static_cast<int64_t>(total)));
            int pick = fits.back();
            for (int cand : fits) {
                if (r < prof.tiles[cand].weight) { pick = cand; break; }
                r -= prof.tiles[cand].weight;
            }
            c.put(x, y, mappingOfTile[pick], prof.tiles[pick].cellType);
            bm.setTile(x, y, mappingOfTile[pick]);
        }
}

}  // namespace

Result generate(const Profile& profile, const Params& in) {
    Params p = in;
    if (p.width < 16 || p.height < 16 || p.width > 1024 || p.height > 1024)
        throw std::runtime_error("map size must be 16..1024 cells per side");
    if (profile.standardMapping.empty())
        throw std::runtime_error("the profile has no standard mapping table");
    const int w = p.width, h = p.height;

    std::vector<std::string> ids;
    for (const Ground& g : profile.grounds) ids.push_back(g.id);
    auto ground = [&](const char* id, int fallback) { const int g = profile.findGround(id); return g >= 0 ? g : fallback; };
    const int mud = ground("mud", 0), sand = ground("san", mud), rock = ground("roc", mud), veg = ground("veg", -1);

    Rng rng(p.seed);
    // --- biome field
    Field fam = familyField(p, rng, ids);
    applySymmetry(fam, p.symmetry);
    // Smooth first, place lava after, otherwise smoothing eats the lava area.
    fam = smoothFamilies(fam, p.smooth);
    dissolveSmallBiomes(fam, static_cast<int>(std::lround(p.minBiome * w * h)));
    applySymmetry(fam, p.symmetry);   // dissolving works locally and can break the mirroring
    if (veg >= 0) vegetationSpacing(fam, veg, mud, p.vegetationSpacing);

    // --- landscape relief before the transitions: its rims and riverbeds change biome borders
    Mask landH, landBeds;
    if (p.landscape) {
        std::tie(landH, landBeds) = landscapeHeightField(p, rng);
        if (p.symmetry != Sym::None) {
            applySymmetry(landH, p.symmetry);
            applySymmetry(landBeds, p.symmetry);
        }
        landscapeFamilies(fam, landH, landBeds, mud, sand);
    }

    // --- features first: the volcano search must know the clearings
    const std::vector<Placement> layout = featureLayout(p, rng);
    std::vector<Point> sites = volcanoSites(fam, p, rng, rock);
    {
        Mask clear(w, h, 0);
        for (const Placement& e : layout)
            if (e.feat.kind == FeatKind::Clearing) {
                const Mask m = featureMask(e, w, h);
                for (size_t i = 0; i < m.v.size(); ++i) clear.v[i] |= m.v[i];
            }
        sites.erase(std::remove_if(sites.begin(), sites.end(), [&](const Point& s) { return clear.at(s.first, s.second) != 0; }), sites.end());
    }
    steerFeatureFamilies(fam, layout, mud, sand);

    // --- height field
    Mask volcanoBlock(w, h, 0);
    for (const auto& [vx, vy] : sites)
        for (int y = std::max(0, vy - 10); y <= std::min(h - 1, vy + 10); ++y)
            for (int x = std::max(0, vx - 10); x <= std::min(w - 1, vx + 10); ++x) volcanoBlock.at(x, y) = 1;
    Mask H;
    if (p.landscape) {
        // The cliff slider is already part of the relief; only the requested features add to it.
        Params noCliffs = p;
        noCliffs.cliffs = 0.0;
        H = heightField(noCliffs, rng, layout);
        for (size_t i = 0; i < H.v.size(); ++i) H.v[i] = static_cast<uint8_t>(H.v[i] | landH.v[i]);
    } else {
        H = heightField(p, rng, layout);
    }
    if (p.symmetry != Sym::None) applySymmetry(H, p.symmetry);
    for (size_t i = 0; i < H.v.size(); ++i) if (volcanoBlock.v[i]) H.v[i] = 0;
    if (p.landscape && H.any())
        H = removeSmallBits(H, std::max(24, static_cast<int>(std::lround(0.0015 * w * h))));

    // --- ground and transition tiles (this library's blend engine)
    // Grounds without fill tiles cannot be placed (snow: of its tiles almost
    // all are impassable, none common enough; Python: familie_nutzbar). As in
    // the Python, such areas end up as rock (mud if rock cannot be placed).
    const int standIn = !profile.grounds[static_cast<size_t>(rock)].fill.empty() ? rock : mud;
    for (int16_t& g : fam.v)
        if (g < 0 || g >= static_cast<int>(profile.grounds.size()) || profile.grounds[static_cast<size_t>(g)].fill.empty())
            g = static_cast<int16_t>(standIn);

    Canvas canvas(profile, w, h);
    // Start every cell on a fixed tile (one the profile does not know), so
    // painting makes the engine choose fresh fill tiles everywhere; a cell
    // that already held a fitting fill tile would be kept as it is.
    int fixedMapping = 0;
    for (size_t m = 0; m < profile.standardMapping.size(); ++m)
        if (profile.findTile(profile.standardMapping[m].first, profile.standardMapping[m].second) < 0) {
            fixedMapping = static_cast<int>(m);
            break;
        }
    std::vector<uint16_t> init(static_cast<size_t>(w) * h, static_cast<uint16_t>(fixedMapping));
    std::vector<om_mapping> table(profile.standardMapping.size());
    for (size_t m = 0; m < table.size(); ++m)
        table[m] = {profile.tilesets[profile.standardMapping[m].first].c_str(), profile.standardMapping[m].second};
    BlendMap bm(profile, w, h, table.data(), static_cast<int>(table.size()), init.data());
    {
        std::vector<std::vector<int32_t>> byGround(profile.grounds.size());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const int g = fam.at(x, y);
                byGround[static_cast<size_t>(g)].push_back(x);
                byGround[static_cast<size_t>(g)].push_back(y);
            }
        for (size_t g = 0; g < byGround.size(); ++g)
            if (!byGround[g].empty()) bm.paint(static_cast<int>(g), byGround[g].data(), static_cast<int>(byGround[g].size() / 2));
        om_rect all{0, 0, w, h};
        bm.resolve(&all, static_cast<uint64_t>(rng.integers(0, int64_t(1) << 62)));
        takeChanges(canvas, bm);
    }

    // --- volcanoes and lava zones
    Result r;
    r.volcanoes = stampVolcanoes(canvas, bm, sites);
    for (const Point& v : r.volcanoes) r.lavaZones.push_back(lavaZone(v, w, h, p.lavaZoneTiles, rng));

    // --- cliffs last: they need real ground neighbours
    Mask cliffs = cliffsFromContours(canvas, bm, H, rng, p.ramps);
    // Ground next to the new cliffs: re-pick what no longer fits.
    {
        om_rect all{0, 0, w, h};
        bm.resolve(&all, static_cast<uint64_t>(rng.integers(0, int64_t(1) << 62)));
        takeChanges(canvas, bm);
    }
    seamPass(canvas, bm, rng);

    // --- feature report and blocking (_apply_features)
    Mask blocked = volcanoBlock;
    for (size_t i = 0; i < blocked.v.size(); ++i) blocked.v[i] |= cliffs.v[i];
    for (const Placement& e : layout) {
        const Mask m = featureMask(e, w, h, 4);
        const int k = static_cast<int>(e.feat.kind);
        for (size_t i = 0; i < m.v.size(); ++i) blocked.v[i] |= m.v[i];
        if (e.feat.kind == FeatKind::Clearing) { r.featureReport[k] += 1; continue; }
        for (size_t i = 0; i < m.v.size(); ++i) r.featureReport[k] += (m.v[i] && cliffs.v[i]) ? 1 : 0;
        if (e.feat.kind == FeatKind::Valley && e.feat.lava) {
            // Not lava tiles: the field lava may flow on in-game, the floor between the walls.
            Mask floor(w, h, 0);
            for (const auto& [px, py] : e.path)
                for (int dx = -e.width; dx <= e.width; ++dx)
                    for (int dy = -e.width; dy <= e.width; ++dy) {
                        const int x = px + dx, y = py + dy;
                        if (floor.inside(x, y) && !H.at(x, y)) floor.at(x, y) = 1;
                    }
            std::vector<Point> zone;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) if (floor.at(x, y)) zone.emplace_back(x, y);
            if (!zone.empty()) r.lavaZones.push_back(std::move(zone));
        }
    }

    // --- scatter last, around the cliffs rather than under them
    // With symmetry the grounds and cliffs are already symmetric (each half
    // drawn on its own, so walls face the right way); only the scatter follows
    // the mirroring. The Python copied the whole tile grid here
    // (_mirror_region), which turned the mirrored half's cliffs the wrong way.
    const int detailCells = scatterDetails(canvas, bm, fam, rng, static_cast<int>(std::lround(p.details * 0.06 * w * h)),
                                           blocked, p.symmetry);

    // --- cell types, lava zones without cliffs, spawns, statistics
    Grid<uint8_t> cellTypes(w, h, 0);
    for (size_t i = 0; i < cellTypes.v.size(); ++i) cellTypes.v[i] = static_cast<uint8_t>(canvas.cells.v[i] & 31);
    for (auto& zone : r.lavaZones)
        zone.erase(std::remove_if(zone.begin(), zone.end(), [&](const Point& q) { return isCliff(cellTypes.at(q.first, q.second)); }), zone.end());
    r.lavaZones.erase(std::remove_if(r.lavaZones.begin(), r.lavaZones.end(), [](const auto& z) { return z.empty(); }), r.lavaZones.end());

    r.spawns = pickSpawns(cellTypes, p.spawns, p.spawnClearance, p.symmetry, rng);
    r.tiles = canvas.tiles;
    r.cells = cellTypes;
    const double area = static_cast<double>(w) * h;
    r.passableShare = static_cast<double>(passableMask(cellTypes).count()) / area;
    r.connected = connected(cellTypes, r.spawns);
    size_t plateau = 0;
    for (uint8_t ct : cellTypes.v) plateau += isCliff(ct);
    r.plateauShare = plateau / area;
    r.detailShare = detailCells / area;
    r.riverShare = landBeds.v.empty() ? 0.0 : landBeds.count() / area;
    r.familyShare.assign(profile.grounds.size(), 0.0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int g = bm.groundAt(x, y);
            if (g >= 0) r.familyShare[static_cast<size_t>(g)] += 1.0 / area;
        }
    for (const Placement& e : layout)
        r.featureSpots.push_back({e.feat.kind, e.x, e.y, e.radius ? e.radius : e.width});
    return r;
}

}  // namespace om::gen
