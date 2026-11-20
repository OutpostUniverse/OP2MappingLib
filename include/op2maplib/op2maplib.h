/*
 * op2maplib - reusable map-editing algorithms for Outpost 2 map tools.
 *
 * Plain C API so any language can use the library: C/C++ directly, .NET via
 * P/Invoke, Python via ctypes/cffi, Rust, Delphi, WebAssembly, ...
 *
 * Current module: terrain blending ("om_blend_*"). Paint a ground type
 * (mud, rock, sand, ...) onto cells; the library picks tiles so the borders
 * between grounds use the game's own transition tiles. Which tiles exist and
 * how they fit together comes from a profile (JSON, see docs/profile-format.md),
 * so the library itself knows nothing about a particular tileset.
 *
 * The library never reads or writes .map files. The caller passes the map's
 * tile mapping table and the mapping index of every cell, and gets back a list
 * of changes to apply in its own editor (and its own undo system).
 *
 * Conventions
 *   - Every function is safe to call with NULL handles (it does nothing / fails).
 *   - Strings are UTF-8 (in practice ASCII). Returned strings are owned by the
 *     library and stay valid while the owning object lives.
 *   - Results are deterministic: the same inputs and seed give the same tiles
 *     on every platform and in every language binding.
 *   - Not thread-safe per object; separate objects may be used from separate threads.
 *   - Objects returned by *_load / *_create / *_find are owned by the caller
 *     and must be released with the matching *_free (which accepts NULL).
 *
 * The blending approach and the Outpost 2 tile data come from
 * op2-codegen-editor (MIT licence).
 */
#ifndef OP2MAPLIB_H
#define OP2MAPLIB_H

#include <stddef.h>
#include <stdint.h>

/*
 * OM_API marks exported functions: define OP2MAPLIB_BUILD when building the
 * DLL, OP2MAPLIB_STATIC when linking the static library, neither when using
 * the DLL. OM_CALL fixes the calling convention (cdecl) for 32-bit Windows.
 */
#if defined(_WIN32)
#  if defined(OP2MAPLIB_BUILD)
#    define OM_API __declspec(dllexport)
#  elif defined(OP2MAPLIB_STATIC)
#    define OM_API
#  else
#    define OM_API __declspec(dllimport)
#  endif
#  define OM_CALL __cdecl
#else
#  define OM_API __attribute__((visibility("default")))
#  define OM_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------ */
/* Library                                                                   */

/** Library version as "major.minor.patch". */
OM_API const char* OM_CALL om_version(void);

/* ------------------------------------------------------------------------ */
/* Profiles                                                                  */

/** Opaque loaded profile. Read-only once loaded; one profile can serve many
    blend maps. */
typedef struct om_profile om_profile;

/** One entry of a map's tile mapping table. */
typedef struct om_mapping {
    const char* tileset;   /* tileset file name as stored in the map, e.g. "well0001" (case-insensitive) */
    int32_t     image;     /* image (graphic) index within that tileset */
} om_mapping;

/** What kind of profile tile a cell holds (om_blend_tile_role). */
typedef enum om_tile_role {
    OM_TILE_UNKNOWN    = 0,   /* not in the profile: a fixed cell (cliff, wall, ...) */
    OM_TILE_FILL       = 1,   /* plain ground the library places */
    OM_TILE_GROUND     = 2,   /* belongs to a ground but is never placed (decoration, rare tiles) */
    OM_TILE_TRANSITION = 3    /* blends two grounds */
} om_tile_role;

/**
 * Loads a blend profile from JSON text (not a file name, so callers can embed
 * it or read it however they like). Returns NULL on failure and, if err is not
 * NULL, writes a message into err (at most err_size bytes, always terminated).
 */
OM_API om_profile* OM_CALL om_profile_load(const char* json, size_t json_len,
                                           char* err, size_t err_size);

/** Frees a profile (NULL is fine). Free every blend map created from it first. */
OM_API void OM_CALL om_profile_free(om_profile* profile);

/** Profile display name, e.g. "Outpost 2 standard tilesets (...)". */
OM_API const char* OM_CALL om_profile_name(const om_profile* profile);

/** Number of grounds (terrain types) in the profile. */
OM_API int OM_CALL om_profile_ground_count(const om_profile* profile);

/** Short id of a ground, e.g. "mud". NULL if out of range. */
OM_API const char* OM_CALL om_profile_ground_id(const om_profile* profile, int ground);

/** Display name of a ground, e.g. "Mud". NULL if out of range. */
OM_API const char* OM_CALL om_profile_ground_name(const om_profile* profile, int ground);

/**
 * Number of plain fill tiles for a ground. A ground with 0 cannot be painted
 * usefully (there is nothing to fill it with); UIs should hide or disable it.
 */
OM_API int OM_CALL om_profile_ground_fill_count(const om_profile* profile, int ground);

/** Index of the ground with this id, or -1. */
OM_API int OM_CALL om_profile_find_ground(const om_profile* profile, const char* id);

/** 1 if the profile has transition tiles between the two grounds, else 0. */
OM_API int OM_CALL om_profile_grounds_blend(const om_profile* profile, int ground_a, int ground_b);

/**
 * The standard tile mapping table carried by the profile (for Outpost 2: the
 * table shared by all campaign maps). Useful for creating new maps and for
 * tests. Returns the number of entries; 0 if the profile has none.
 */
OM_API int OM_CALL om_profile_standard_mapping_count(const om_profile* profile);

/** Fills *out with standard mapping entry i. Returns 1 on success, 0 if out of range. */
OM_API int OM_CALL om_profile_standard_mapping(const om_profile* profile, int index, om_mapping* out);

/* ------------------------------------------------------------------------ */
/* Blend maps                                                                */

/** Opaque blend state of one map: a copy of its cells plus a ground per cell. */
typedef struct om_blend_map om_blend_map;

/** A rectangle in cells: x, y inclusive; width/height in cells. */
typedef struct om_rect {
    int32_t x, y, width, height;
} om_rect;

/** One tile the caller should change. */
typedef struct om_change {
    int32_t x, y;
    int32_t old_mapping;   /* mapping index before */
    int32_t new_mapping;   /* mapping index to write */
    int32_t cell_type;     /* suggested cell type for the new tile, or -1 if the profile has none */
} om_change;

/**
 * Creates the blend state for one map.
 *   mappings / mapping_count : the map's tile mapping table.
 *   cells                    : width*height mapping indices, row by row.
 * The profile must stay loaded until the map is freed.
 * Each cell's ground is worked out from its tile. Tiles the profile does not
 * know (cliffs, walls, buildings, ...) are "fixed": never changed and ignored
 * when choosing neighbours' tiles, until a cell is painted over.
 */
OM_API om_blend_map* OM_CALL om_blend_create(const om_profile* profile,
                                             int32_t width, int32_t height,
                                             const om_mapping* mappings, int32_t mapping_count,
                                             const uint16_t* cells,
                                             char* err, size_t err_size);

/** Frees a blend map (NULL is fine). */
OM_API void OM_CALL om_blend_free(om_blend_map* map);

/** Ground of a cell, or -1 for fixed/unknown cells or out-of-range coordinates. */
OM_API int OM_CALL om_blend_ground_at(const om_blend_map* map, int32_t x, int32_t y);

/** Mapping index currently held for a cell, or -1 if out of range. */
OM_API int OM_CALL om_blend_tile_at(const om_blend_map* map, int32_t x, int32_t y);

/** Role (om_tile_role) of the tile a cell holds; OM_TILE_UNKNOWN if out of range. */
OM_API int OM_CALL om_blend_tile_role(const om_blend_map* map, int32_t x, int32_t y);

/**
 * Tells the library a cell's tile was changed outside it (manual tile placement,
 * undo/redo, ...). The cell's ground is re-derived from the new tile.
 */
OM_API void OM_CALL om_blend_set_tile(om_blend_map* map, int32_t x, int32_t y, int32_t mapping);

/**
 * Paints a ground onto cells. xy holds count (x, y) pairs. Out-of-range cells
 * are skipped. Only the ground layer changes; call om_blend_resolve to pick
 * tiles. Returns the number of cells painted, or -1 for an invalid ground.
 */
OM_API int OM_CALL om_blend_paint(om_blend_map* map, int ground, const int32_t* xy, int32_t count);

/**
 * Picks tiles for a region and applies them to the library's copy of the map.
 * region: cells to re-evaluate (one extra cell around it is included, since a
 * tile depends on its neighbours). NULL = everything painted since the last
 * resolve. Returns the number of changes, retrievable with om_blend_get_changes
 * until the next resolve, or -1 on error.
 */
OM_API int OM_CALL om_blend_resolve(om_blend_map* map, const om_rect* region, uint64_t seed);

/** Whole map: re-picks every blendable cell (a "clean up transitions" pass). */
OM_API int OM_CALL om_blend_resolve_all(om_blend_map* map, uint64_t seed);

/** Copies up to max changes from the last resolve or cliff draw. Returns the number copied. */
OM_API int OM_CALL om_blend_get_changes(const om_blend_map* map, om_change* out, int32_t max);

/* ------------------------------------------------------------------------ */
/* Cliffs                                                                    */
/*                                                                           */
/* Draw a cliff along a line of points. The line is turned into a band of    */
/* rock on a 4x4-per-tile grid (a wall where the high side is north, a thin  */
/* edge where it is south, steps on steep slopes), then the profile's cliff  */
/* tiles are fitted to it. Works on an om_blend_map; the changes come back   */
/* through om_blend_get_changes, with the cliff tiles' cell types (cliff     */
/* high side, low side, north cliffs, ...). Cliff cells become fixed for     */
/* terrain blending.                                                         */

/** Which side of a cliff line is high ground. LEFT/RIGHT are relative to the
    drawing direction; NORTH/SOUTH are fixed to the map. */
typedef enum om_cliff_side {
    OM_CLIFF_HIGH_LEFT  = 0,
    OM_CLIFF_HIGH_RIGHT = 1,
    OM_CLIFF_HIGH_NORTH = 2,
    OM_CLIFF_HIGH_SOUTH = 3
} om_cliff_side;

/**
 * 1 if the ground has cliff tiles of its own, 0 if cliffs drawn on it use the
 * profile's fallback cliffs (Outpost 2: mud), -1 if the profile has no usable
 * cliffs at all.
 */
OM_API int OM_CALL om_profile_cliff_ground(const om_profile* profile, int ground);

/**
 * Draws a cliff along a polyline.
 *   xy / point_count : point_count (x, y) cell coordinates, at least 2. Repeat
 *                      the first point at the end to close a loop.
 *   high_side        : an om_cliff_side.
 *   wall_height      : wall height in tiles, 3 to 6 (clamped).
 *   seed             : same seed and input give the same cliff.
 * The cliff family comes from the ground at the first point. Returns the
 * number of changes (om_blend_get_changes), 0 if nothing could be drawn, or
 * -1 for invalid input (points outside the map, bad side).
 */
OM_API int OM_CALL om_cliff_draw(om_blend_map* map, const int32_t* xy, int32_t point_count,
                                 int high_side, double wall_height, uint64_t seed);

/**
 * How the last om_cliff_draw was solved: 0..7 = solver stage (0 is the
 * strictest: only tile pairs seen in the original maps), -1 = greedy fallback
 * (no stage could solve it; expect rougher joins), -2 = nothing drawn.
 */
OM_API int OM_CALL om_cliff_last_stage(const om_blend_map* map);

/**
 * Snaps a point for interactive drawing: moves (bx, by) so the segment from
 * (ax, ay) uses an allowed cliff slope (flat, vertical, 45 degrees, or 2:1,
 * 3:1, 4:1 either way), keeping about the same length.
 */
OM_API void OM_CALL om_cliff_snap_point(int32_t ax, int32_t ay, int32_t bx, int32_t by,
                                        int32_t* out_x, int32_t* out_y);

/* ------------------------------------------------------------------------ */
/* Passable regions (enclosed areas)                                         */
/*                                                                           */
/* Splits a map's passable cells into connected regions, largest first. The */
/* largest region is the main play area; every other region is enclosed:     */
/* units can never reach it from the main area. Works on plain cell types    */
/* (Outpost 2 CellType values 0-31), so it needs no profile or blend map.    */

/** Opaque result of om_regions_find; holds its own copy of the regions. */
typedef struct om_regions om_regions;

/** Options for om_regions_find. */
typedef enum om_region_flags {
    OM_REGIONS_CONNECT8         = 1,   /* diagonal neighbours connect too (else only N/E/S/W) */
    OM_REGIONS_WRAP_X           = 2,   /* the left and right map edges join (Outpost 2 world maps) */
    OM_REGIONS_SKIP_BOTTOM_ROW  = 4    /* ignore the last row (Outpost 2: never shown, always impassable) */
} om_region_flags;

/**
 * The flags matching Outpost 2: 8-connected, bottom row skipped, and the left
 * and right edges joined only on the 512x256 world map size.
 */
OM_API uint32_t OM_CALL om_regions_op2_flags(int32_t width, int32_t height);

/**
 * Bit mask of the cell types that count as passable in Outpost 2 (bit n =
 * cell type n): fast, medium and slow passable, bulldozed, rubble and tubes.
 * Walls, cliffs, vents and the impassable types block.
 */
OM_API uint32_t OM_CALL om_cell_types_passable_mask(void);

/**
 * Finds the connected regions of passable cells.
 *   cell_types     : width*height cell types (0-31), row by row.
 *   passable_mask  : bit n set = cell type n is passable; 0 = om_cell_types_passable_mask().
 *   flags          : om_region_flags, e.g. om_regions_op2_flags(width, height).
 * Regions are sorted largest first (ties: the one whose first cell comes
 * first, row by row). Returns NULL for invalid input. Free with om_regions_free.
 */
OM_API om_regions* OM_CALL om_regions_find(int32_t width, int32_t height, const uint8_t* cell_types,
                                           uint32_t passable_mask, uint32_t flags);

/** Frees a regions result (NULL is fine). */
OM_API void OM_CALL om_regions_free(om_regions* regions);

/** Number of regions (0 if the map has no passable cells). */
OM_API int OM_CALL om_regions_count(const om_regions* regions);

/** Number of cells in a region, or 0 if out of range. */
OM_API int OM_CALL om_regions_size(const om_regions* regions, int region);

/**
 * Copies up to max cells of a region as (x, y) pairs into xy (2*max ints).
 * Returns the number of cells copied.
 */
OM_API int OM_CALL om_regions_cells(const om_regions* regions, int region, int32_t* xy, int32_t max);

/** Region of a cell, or -1 if the cell is not passable, skipped or outside the map. */
OM_API int OM_CALL om_regions_at(const om_regions* regions, int32_t x, int32_t y);

/** Total number of cells in all enclosed regions (every region but the largest). */
OM_API int OM_CALL om_regions_enclosed_size(const om_regions* regions);

/**
 * Copies up to max cells of all enclosed regions (every region but the
 * largest) as (x, y) pairs. Returns the number of cells copied.
 */
OM_API int OM_CALL om_regions_enclosed_cells(const om_regions* regions, int32_t* xy, int32_t max);

#ifdef __cplusplus
}
#endif

#endif /* OP2MAPLIB_H */
