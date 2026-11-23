# OP2MappingLib

Shared functions for Outpost 2 map making.

OP2MappingLib holds **op2maplib**, a library of reusable map-editing
algorithms for Outpost 2 map tools. It is used by
[OP2Mapper](https://github.com/leviathan400/OP2Mapper) for its Terrain Brush,
Draw Cliff and Find & Fix Enclosed Areas tools. The core is C++17
with no dependencies, behind a plain C API, so any editor can use it: C/C++
directly, .NET via P/Invoke, Python via ctypes, Rust, Delphi, WebAssembly.

Modules:

- **Terrain blending.** Paint a ground (mud, rock, sand, ...) onto cells, and
  the library picks tiles so the borders between grounds use the game's own
  transition tiles.
- **Cliffs.** Draw a line of points and choose the high side and wall height.
  The library builds a cliff from the game's cliff tiles: walls where the high
  side is north, thin edges where it is south, side walls, slopes and steps.
  Each cliff tile comes with its cell type (cliff high side, low side or north
  cliffs).
- **Passable regions (enclosed areas).** Splits a map's passable cells into
  connected regions, largest first. The largest is the main play area; the
  rest are enclosed areas units can never reach. It works on plain cell types,
  so it needs no profile. Outpost 2's rules are built in:
  - passable means fast, medium and slow passable, bulldozed, rubble and tubes;
  - 8-way connected;
  - the bottom row is skipped;
  - the left and right edges join only on 512×256 world maps.

  Moved here from OP2Mapper's Find & Fix Enclosed Areas, and checked to give
  identical results on all 63 maps in OPU's `base/maps`.
- **Map generator.** Generates a whole map from a seed: biome layout (six
  templates, terrain mix, roughness, symmetry for 2 or 4 players), relief
  (plateaus with ramps; valley, mountain, plateau, clearing and choke
  features; the New Terra landscape mode with a highland massif, mesas and dry
  riverbeds), volcanoes with lava-possible zones, scattered rocks and craters,
  and spawn points. Grounds come from the blend engine and cliffs from the
  cliff tool, so a generated map follows the same tile rules as painting.
  Ported from op2-codegen-editor's generator.

## Layout

| Path | What |
|---|---|
| `include/op2maplib/op2maplib.h` | The C API, the only public header. |
| `src/` | Implementation: blend engine, profile loader, small JSON reader. |
| `src/mapgen/` | Map generator: biome field, shapes, relief and features, spawn points, the pipeline and its C API. |
| `profiles/op2.json` | Outpost 2 blend profile (standard `well0000`–`well0012` tilesets). |
| `docs/profile-format.md` | Profile format, and how the solver uses it. |
| `tools/export_op2_profile.py` | Generates `profiles/op2.json` from op2-codegen-editor's learned tile model. |
| `bindings/python/` | Python binding (ctypes) and its tests. |
| `bindings/dotnet/` | .NET binding (`Op2MapLib.Net`, P/Invoke, .NET Framework 4.8; usable from C# and VB.NET), prebuilt native DLLs in `native/x86` and `native/x64`, and its tests. |
| `tools/build-native.ps1` | Builds and tests the 32-bit and 64-bit DLLs and refreshes `bindings/dotnet/native/`. |
| `tools/compile-check.bat` | Compiles single source files with MSVC (`/W4 /WX`, no linking), for quick checks while editing. |
| `tests/` | C++ tests, and shared test vectors every binding must reproduce. |
| `THIRD-PARTY-NOTICES.md` | Licence of the op2-codegen-editor code this library is ported from. |

## Building

Requires CMake 3.16+ and a C++17 compiler.

```
cmake -S . -B build -A x64            # Windows, 64-bit (use -A Win32 for 32-bit)
cmake --build build --config Release
build\bin\Release\op2maplib_tests.exe profiles\op2.json
```

On Linux or macOS, leave out `-A` and the `Release` folder. On Windows the DLL
links the C runtime statically, so it needs no VC++ redistributable.

Python tests, run from the repository root after building:

```
python -I bindings/python/tests/test_blend.py
python -I bindings/python/tests/test_cliffs.py
python -I bindings/python/tests/test_mapgen.py
python -I bindings/python/tests/test_regions.py "path/to/Outpost 2/OPU/base/maps"
```

`test_regions.py` checks every `.map` in a folder, which can also be given in
the `OP2_MAPS` environment variable.

.NET binding:
- Reference `bindings/dotnet/Op2MapLib.Net.csproj` from your project. The
  referencing project's output folder then also gets `x86\op2maplib.dll`,
  `x64\op2maplib.dll` and `op2maplib\op2.json`.
- The binding loads the native DLL that matches the process, so AnyCPU hosts
  work as 32-bit or 64-bit. `BlendProfile.LoadDefault()` loads the shipped
  profile.
- The native DLLs in `bindings/dotnet/native/` are committed. After changing the
  C++ code, run `tools/build-native.ps1` to rebuild and test both DLLs and copy
  them there.
- Tests, for each of `x86` and `x64`:

```
msbuild bindings\dotnet\tests\Op2MapLib.Net.Tests.csproj /p:Configuration=Release /p:TestArch=x64
bindings\dotnet\tests\bin\x64\Op2MapLib.Net.Tests.exe
```

OP2Mapper uses the prebuilt binding: `Op2MapLib.Net.dll` as a plain reference,
with `x86\op2maplib.dll`, `x64\op2maplib.dll` and `op2maplib\op2.json` shipped
next to `OP2Mapper.exe`.

## Using it

```c
om_profile* profile = om_profile_load(json, json_len, err, sizeof err);
om_blend_map* map = om_blend_create(profile, width, height,
                                    mappings, mapping_count,   /* the map's tile mapping table */
                                    cells, err, sizeof err);   /* width*height mapping indices */

om_blend_paint(map, om_profile_find_ground(profile, "san"), xy, count);
int n = om_blend_resolve(map, NULL, seed);       /* NULL = everything painted since last time */
om_change changes[n];
om_blend_get_changes(map, changes, n);           /* apply these to your map, as one undo step */
```

- The library never reads or writes `.map` files. You pass the map's tile
  mapping table and cells, and get back a list of changes, each with a
  suggested cell type.
- Results are deterministic: the same inputs and seed give the same tiles on
  every platform and in every binding. `tests/vectors/` pins this.
- If your editor changes a tile itself (manual placement, undo), call
  `om_blend_set_tile` so the library's copy stays in step.
- Tiles the profile doesn't know (cliffs, walls, buildings) are never changed
  unless painted over.

Cliffs use the same blend map:

```c
const int32_t pts[] = {20, 6, 20, 20, 36, 20};          /* x, y pairs */
int n = om_cliff_draw(map, pts, 3, OM_CLIFF_HIGH_LEFT, 3.0, seed);
om_blend_get_changes(map, changes, n);                   /* tiles + cell types */
```

For interactive drawing, `om_cliff_snap_point` snaps each new point to an
allowed slope. Cliff cells become fixed for blending, so painting next to a
cliff never replaces it.

Regions need only the cell types:

```c
om_regions* r = om_regions_find(width, height, cell_types, 0, om_regions_op2_flags(width, height));
int enclosed = om_regions_enclosed_size(r);               /* cells outside the main area */
om_regions_enclosed_cells(r, xy, enclosed);               /* their (x, y) pairs */
om_regions_free(r);
```

The map generator returns a whole map:

```c
om_mapgen_settings* s = om_mapgen_settings_create("islands");    /* plains, canyon, desert, islands, ice, newterra */
om_mapgen_set(s, "width", 128);
om_mapgen_set(s, "height", 128);
om_mapgen_set(s, "spawns", 2);
om_mapgen_set(s, "cliffs", 0.2);
om_mapgen_set_seed(s, 1234);
om_mapgen_add_feature(s, OM_FEATURE_MOUNTAIN, "c", "", 0.6, 0);
om_mapgen_result* r = om_mapgen_generate(profile, s, err, sizeof err);
om_mapgen_get_tiles(r, tiles, 128 * 128);          /* mapping indices into the standard mapping table */
om_mapgen_get_cell_types(r, cell_types, 128 * 128);
int spawns = om_mapgen_get_points(r, OM_MAPGEN_SPAWNS, xy, 8);
om_mapgen_result_free(r);
om_mapgen_settings_free(s);
```

The tiles index the profile's standard mapping table
(`om_profile_standard_mapping`), which the new map uses as its own. The result
also lists volcanoes, lava-possible zones (for the mission), placed features
and statistics. In Python: `MapGenerator("islands", seed=1234, spawns=2).generate(profile)`;
in .NET: `new MapGenSettings("islands").Generate(profile)`.

## How cliffs work

A port of op2-codegen-editor's cliff tool (`src/cliff.cpp`):

1. **Band.** Each tile is split into 4x4 sub-cells, and a band of rock is drawn
   along the line on that grid.
   - Where the high side is north, the band is a wall. Its thickness comes from
     the wall height and wanders a little.
   - Where the high side is south, it's a thin "north cliff" line. Steep north
     lines become steps.
   - Slopes are split into flat runs and 45° pieces, and the ends taper.
2. **Rules.** The band is adjusted until real tiles can fit it:
   - no rock on an edge facing open ground;
   - neighbouring cliff cells share their edge rock;
   - a wall gets at most two filler tiles of height.
3. **Solver.** A backtracking search fits tiles to the band, in stages from
   strict (only tile pairs seen side by side in the original maps) to looser.
   Among near-equal choices it prefers tiles whose rock image matches the band,
   seams that look right, and less repetition. A greedy pass is the last resort.

Differences from the original: the library uses its own seeded random numbers,
and the solver is limited by search steps instead of time, so results are the
same on every machine.

## How blending works (version 1)

Each cell has a ground. A cell of ground A next to ground B gets a transition
tile when the profile has tiles for that pair. Each corner of the cell counts as
B if any cell meeting at that corner is B, and the four corners select the tile.
Neighbouring cells share the corners of their common edge, so their tiles agree
there without further matching. Tiles that already fit are kept, so
re-resolving doesn't reshuffle the map. Details are in `docs/profile-format.md`.

The approach and the Outpost 2 tile data come from op2-codegen-editor's terrain
brush, which learned them from the campaign maps.

## How the map generator works

A port of op2-codegen-editor's generator (`mapview/mapgen*.py`,
`shapegen.py`), in `src/mapgen/`:

1. **Layout.** A value-noise field picks a ground per cell from the terrain
   mix; roughness sets the noise frequency. Then symmetry, majority smoothing,
   small patches dissolved into their surroundings, and vegetation kept away
   from rock and sand (there are no transitions between them).
2. **Relief.** Features are placed first (sectors of a 3x3 grid, jittered),
   then the height field: mountains and plateaus as irregular blobs, valleys
   as corridors, chokes as a barrier with one gap, extra blobs from the cliffs
   setting, clearings punched out, edges snapped to a 3-cell grid. New Terra
   builds one massif at a map side, mesas and meandering dry riverbeds.
3. **Tiles.** The biome field is painted into a blend map and resolved, so
   borders get the game's transition tiles. A ground with no fill tiles in the
   profile (snow) becomes rock, as in the editor.
4. **Volcanoes**, then **cliffs**: each highland area's boundary is traced,
   split into arcs with ramp gaps (the largest area gets the ramps setting,
   every mesa one), simplified and drawn with the cliff tool.
5. **Seam pass.** In reading order, each fill and transition tile is re-picked
   among tiles of the same kind so it fits its left and upper neighbour (a
   pair seen in the original maps). This stands in for the editor's tile
   sampler, which drew every tile from the learned neighbour table.
6. **Scatter** (rocks, cracks, craters) as whole stamps on matching ground,
   only where their borders fit; then **spawn points** on free, far-apart,
   connected ground.

Differences from the editor's generator:
- Same seed, different map: the library has its own random numbers and its
  own tile layer.
- Symmetry: the editor copies the finished tile grid onto the mirrored half,
  which turns that half's cliff walls the wrong way. Here the biome and height
  fields are symmetric and both halves get their own tiles and cliffs; only the
  scatter is placed mirrored.
- Measured on the six templates at 128x128 (seed 1234), neighbour pairs never
  seen in the original maps: 0.2–4.3% here against 2.9–19.4% for the editor
  (`bindings/python/tests/test_mapgen.py` prints them). Ground shares and
  passability are close to the editor's.
- Speed: 20–30 ms for a map without cliffs (editor: about 2 s). Cliffs cost
  about 50 ms per wall, so a canyon with many plateaus takes a few seconds
  (editor: about 8 s).

## Status

- Version 0.3.2. Blending works for mud, rock and sand (and lava/rock).
  Cliffs work for mud and sand, and every other ground uses mud cliffs.
  Passable regions (enclosed areas) work for any map. The map generator
  covers everything the editor's generator does.
- The DLLs carry version details (company Outpost Universe). The .NET binding
  looks for the profile at `op2maplib\op2.json` next to it, and still accepts
  the older `op2maplib\profiles\op2.json`.
- The C++ tests (32-bit and 64-bit), the Python tests and the .NET tests (32-bit
  and 64-bit) all pass and reproduce the same golden hashes
  (`tests/vectors/blend_disc.json`, `tests/vectors/cliff_l.json`,
  `tests/vectors/mapgen_islands.json`).
- Cliff quality, from `bindings/python/tests/test_cliffs.py`: 240 test walls
  (straight, sloped, L/U shapes, loops; all sides; heights 3–6).
  - Every wall was solved without the greedy fallback, with no holes and no
    invalid tile pairs.
  - Pairs never seen in the original maps: 8.2% on mud (op2-codegen-editor:
    25–28%) and 66.5% on sand, whose edge data is automatic.
  - About 55 ms per wall on mud.
  - Not yet checked in the game itself.
- Measured seam quality: on a 64×64 test scene, 4.8% of corner segments and
  11.8% of middle segments don't match their neighbour. The main causes are
  tiles whose corner data is ambiguous, and fallback picks where no exact tile
  exists.
- Next steps:
  - blending: prefer tiles whose edges match already-chosen neighbours (the
    generator's seam pass does this; the blend engine itself does not yet);
  - generator: snow and lava cannot be generated (no fill tiles in the
    profile); quad-symmetry spawns can land close to the centre lines, as in
    the editor;
  - cliffs: close cuts cleanly when a cliff is painted over (ramps);
  - cliffs: a mud rim around cliffs on grounds that use the fallback;
  - cliffs: measure the wall height from nearby cliffs;
  - cliffs, side walls ignore the high side:
    - Vertical or steep lines come out looking the same with the high side
      left or right. The setting only moves the rock band to the other side of
      the line, and the high-side cliff tiles land on the west edge either way.
    - This comes from the original algorithm: a side wall's band is the same
      on both sides, so the solver can't tell high from low.
    - Fix: a soft solver preference for high-side tiles (cell type 9) next to
      the high ground and low-side tiles (10) next to the low ground. First
      check that convention in the campaign maps' vertical cliffs.
    - OP2Mapper: clearer high-side choices. Offer North / South / East / West,
      fixed to the map, and keep "Left / Right of the drawing direction" for
      bent shapes such as L and U. Drawing downward, "left" is currently east.
  - profile, rock types: split "rock" into light rock and dark (lava) rock,
    and check for black rock too:
    - op2-codegen-editor's model lumps them into one "rock" ground, so
      rock-filled areas come out as a light/dark checkerboard.
    - Analyse first:
      - decode the map's own terrain-type table (tile ranges, lava tile);
      - make contact sheets of each ground's fill tiles with colour, cell
        type and tileset;
      - check which rock each transition set blends with (roctomud,
        santoroc, roctola3);
      - see which ground surrounds lava in volcano maps.
    - Then export a profile with separate grounds. Mud cliffs on rock may need
      revisiting too.

## Credits

The terrain blending rules, the cliff tool and the Outpost 2 tile model are
ported from [op2-codegen-editor](https://github.com/fefi-byte/op2-codegen-editor)
by fefi-byte, used under the MIT License. See `THIRD-PARTY-NOTICES.md`.
