// C API of the map generator (om_mapgen_*): settings handle, generate, result
// accessors. Same rules as api.cpp: NULL handles are safe, no C++ exception
// crosses the C boundary.
#include "../handles.hpp"
#include "generate.hpp"

#include <cstring>
#include <exception>
#include <string>

struct om_mapgen_settings {
    om::gen::Params params;
};

struct om_mapgen_result {
    om::gen::Result result;
};

namespace {

using om::gen::Params;

// Numeric settings by key: a reference to the field and whether it is an integer.
struct Field {
    double* d = nullptr;
    int* i = nullptr;
    bool* b = nullptr;
};

Field field(Params& p, const char* key) {
    if (!key) return {};
    const std::string k = key;
    if (k == "width") return {nullptr, &p.width};
    if (k == "height") return {nullptr, &p.height};
    if (k == "roughness") return {&p.roughness};
    if (k == "spawns") return {nullptr, &p.spawns};
    if (k == "spawn_clearance") return {nullptr, &p.spawnClearance};
    if (k == "cliffs") return {&p.cliffs};
    if (k == "ramps") return {nullptr, &p.ramps};
    if (k == "volcanoes") return {nullptr, &p.volcanoes};
    if (k == "lava_zone_tiles") return {nullptr, &p.lavaZoneTiles};
    if (k == "smooth") return {nullptr, &p.smooth};
    if (k == "vegetation_spacing") return {nullptr, &p.vegetationSpacing};
    if (k == "min_biome") return {&p.minBiome};
    if (k == "details") return {&p.details};
    if (k == "landscape") return {nullptr, nullptr, &p.landscape};
    if (k == "rivers") return {nullptr, &p.rivers};
    return {};
}

const std::vector<om::gen::Point>* points(const om_mapgen_result* r, int list) {
    if (!r) return nullptr;
    if (list == OM_MAPGEN_SPAWNS) return &r->result.spawns;
    if (list == OM_MAPGEN_VOLCANOES) return &r->result.volcanoes;
    return nullptr;
}

int copyPoints(const std::vector<om::gen::Point>& pts, int32_t* xy, int32_t max) {
    if (!xy || max <= 0) return 0;
    const int n = static_cast<int>(std::min<size_t>(pts.size(), static_cast<size_t>(max)));
    for (int i = 0; i < n; ++i) {
        xy[2 * i] = pts[static_cast<size_t>(i)].first;
        xy[2 * i + 1] = pts[static_cast<size_t>(i)].second;
    }
    return n;
}

}  // namespace

extern "C" {

OM_API om_mapgen_settings* OM_CALL om_mapgen_settings_create(const char* template_name) {
    try {
        auto* s = new om_mapgen_settings{};
        om::gen::applyTemplate(s->params, template_name ? template_name : "");
        return s;
    } catch (...) {
        return nullptr;
    }
}

OM_API void OM_CALL om_mapgen_settings_free(om_mapgen_settings* settings) { delete settings; }

// The symmetry is stored as an enum; every other key maps to a field.
OM_API int OM_CALL om_mapgen_set(om_mapgen_settings* settings, const char* key, double value) {
    if (!settings || !key) return -1;
    if (std::strcmp(key, "symmetry") == 0) {
        const int v = static_cast<int>(value);
        if (v < OM_SYMMETRY_NONE || v > OM_SYMMETRY_QUAD) return -1;
        settings->params.symmetry = static_cast<om::gen::Sym>(v);
        return 0;
    }
    const Field f = field(settings->params, key);
    if (f.d) *f.d = value;
    else if (f.i) *f.i = static_cast<int>(value);
    else if (f.b) *f.b = value != 0;
    else return -1;
    return 0;
}

OM_API double OM_CALL om_mapgen_get(const om_mapgen_settings* settings, const char* key) {
    if (!settings || !key) return 0;
    if (std::strcmp(key, "symmetry") == 0) return static_cast<double>(settings->params.symmetry);
    if (std::strcmp(key, "seed") == 0) return static_cast<double>(settings->params.seed);
    const Field f = field(const_cast<Params&>(settings->params), key);
    if (f.d) return *f.d;
    if (f.i) return *f.i;
    if (f.b) return *f.b ? 1 : 0;
    return 0;
}

OM_API void OM_CALL om_mapgen_set_seed(om_mapgen_settings* settings, uint64_t seed) {
    if (settings) settings->params.seed = seed;
}

OM_API void OM_CALL om_mapgen_clear_mix(om_mapgen_settings* settings) {
    if (settings) settings->params.mix.clear();
}

OM_API int OM_CALL om_mapgen_set_mix(om_mapgen_settings* settings, const char* ground_id, double share) {
    if (!settings || !ground_id || !*ground_id || share < 0) return -1;
    for (auto& [id, s] : settings->params.mix)
        if (id == ground_id) {
            s = share;
            return 0;
        }
    settings->params.mix.emplace_back(ground_id, share);
    return 0;
}

OM_API void OM_CALL om_mapgen_clear_features(om_mapgen_settings* settings) {
    if (settings) settings->params.features.clear();
}

OM_API int OM_CALL om_mapgen_add_feature(om_mapgen_settings* settings, int kind, const char* where,
                                         const char* to, double size, int lava) {
    if (!settings || kind < OM_FEATURE_VALLEY || kind > OM_FEATURE_CHOKE) return -1;
    om::gen::Feature f;
    f.kind = static_cast<om::gen::FeatKind>(kind);
    f.where = where ? where : "";
    f.to = to ? to : "";
    f.size = size;
    f.lava = lava != 0;
    settings->params.features.push_back(f);
    return 0;
}

OM_API om_mapgen_result* OM_CALL om_mapgen_generate(const om_profile* profile, const om_mapgen_settings* settings,
                                                    char* err, size_t err_size) {
    om::setError(err, err_size, "");
    if (!profile || !settings) {
        om::setError(err, err_size, "no profile or settings given");
        return nullptr;
    }
    try {
        return new om_mapgen_result{om::gen::generate(profile->profile, settings->params)};
    } catch (const std::exception& e) {
        om::setError(err, err_size, e.what());
    } catch (...) {
        om::setError(err, err_size, "unknown error");
    }
    return nullptr;
}

OM_API void OM_CALL om_mapgen_result_free(om_mapgen_result* result) { delete result; }

OM_API int32_t OM_CALL om_mapgen_width(const om_mapgen_result* result) { return result ? result->result.tiles.w : 0; }
OM_API int32_t OM_CALL om_mapgen_height(const om_mapgen_result* result) { return result ? result->result.tiles.h : 0; }

OM_API int OM_CALL om_mapgen_get_tiles(const om_mapgen_result* result, uint16_t* out, int32_t max) {
    if (!result || !out || max <= 0) return 0;
    const auto& v = result->result.tiles.v;
    const int n = static_cast<int>(std::min<size_t>(v.size(), static_cast<size_t>(max)));
    for (int i = 0; i < n; ++i) out[i] = static_cast<uint16_t>(v[static_cast<size_t>(i)]);
    return n;
}

OM_API int OM_CALL om_mapgen_get_cell_types(const om_mapgen_result* result, uint8_t* out, int32_t max) {
    if (!result || !out || max <= 0) return 0;
    const auto& v = result->result.cells.v;
    const int n = static_cast<int>(std::min<size_t>(v.size(), static_cast<size_t>(max)));
    std::memcpy(out, v.data(), static_cast<size_t>(n));
    return n;
}

OM_API int OM_CALL om_mapgen_point_count(const om_mapgen_result* result, int list) {
    const auto* p = points(result, list);
    return p ? static_cast<int>(p->size()) : 0;
}

OM_API int OM_CALL om_mapgen_get_points(const om_mapgen_result* result, int list, int32_t* xy, int32_t max) {
    const auto* p = points(result, list);
    return p ? copyPoints(*p, xy, max) : 0;
}

OM_API int OM_CALL om_mapgen_lava_zone_count(const om_mapgen_result* result) {
    return result ? static_cast<int>(result->result.lavaZones.size()) : 0;
}

OM_API int OM_CALL om_mapgen_lava_zone_size(const om_mapgen_result* result, int zone) {
    if (!result || zone < 0 || zone >= static_cast<int>(result->result.lavaZones.size())) return 0;
    return static_cast<int>(result->result.lavaZones[static_cast<size_t>(zone)].size());
}

OM_API int OM_CALL om_mapgen_lava_zone_cells(const om_mapgen_result* result, int zone, int32_t* xy, int32_t max) {
    if (!result || zone < 0 || zone >= static_cast<int>(result->result.lavaZones.size())) return 0;
    return copyPoints(result->result.lavaZones[static_cast<size_t>(zone)], xy, max);
}

OM_API double OM_CALL om_mapgen_stat(const om_mapgen_result* result, const char* key) {
    if (!result || !key) return 0;
    const auto& r = result->result;
    const std::string k = key;
    if (k == "passable") return r.passableShare;
    if (k == "plateau") return r.plateauShare;
    if (k == "details") return r.detailShare;
    if (k == "rivers") return r.riverShare;
    if (k == "connected") return r.connected ? 1 : 0;
    return 0;
}

OM_API double OM_CALL om_mapgen_ground_share(const om_mapgen_result* result, int ground) {
    if (!result || ground < 0 || ground >= static_cast<int>(result->result.familyShare.size())) return 0;
    return result->result.familyShare[static_cast<size_t>(ground)];
}

OM_API int OM_CALL om_mapgen_feature_count(const om_mapgen_result* result) {
    return result ? static_cast<int>(result->result.featureSpots.size()) : 0;
}

OM_API int OM_CALL om_mapgen_feature(const om_mapgen_result* result, int index,
                                     int32_t* kind, int32_t* x, int32_t* y, int32_t* size) {
    if (!result || index < 0 || index >= static_cast<int>(result->result.featureSpots.size())) return 0;
    const auto& s = result->result.featureSpots[static_cast<size_t>(index)];
    if (kind) *kind = static_cast<int32_t>(s.kind);
    if (x) *x = s.x;
    if (y) *y = s.y;
    if (size) *size = s.size;
    return 1;
}

OM_API int OM_CALL om_mapgen_feature_report(const om_mapgen_result* result, int kind) {
    if (!result || kind < 0 || kind > OM_FEATURE_CHOKE) return 0;
    return result->result.featureReport[static_cast<size_t>(kind)];
}

}  // extern "C"
