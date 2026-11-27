# The learned map generator

`om_mapgen_generate_learned` makes maps in the style of the original Outpost 2
maps. It is fefi-byte's learned generator from op2-codegen-editor
(`mapview/mapgen_netz.py` and its parts `mapgen_netz_waende.py`,
`mapgen_netz_gross.py`, `mapgen_netz_fels.py`, `mapgen_netz_deko.py`, version
0.12.0), ported to C++ with a small inference engine of its own, so the library
still has no dependencies.

## What it does

Two small neural networks, trained by fefi-byte on 61 maps (the campaign and
multiplayer maps plus OPU's maps):

1. **Sketch network** (0.7 million weights). Lays out walls and grounds on a
   half-resolution grid: per sketch cell a wall (none, or the direction it
   falls towards: E, SE, S, SW, W, NW, N, NE) and a ground (mud, rock, sand,
   vegetation). It is guided by zones:
   - *tight*: mountains and gorges, where no building should fit;
   - *building ground*: plateaus, valleys, and large free areas whose share
     follows the `cliffs` setting;
   - *any*: the rest.
2. **Tile network** (1.3 million weights). Paints the map tile by tile from
   the sketch, choosing among 1442 tiles. Only tile pairs that occur side by
   side in the training maps are placed.

The zones come from the mountain planner (`mountains`, `mountain_density`,
`gorges`, `valleys`, `plateaus`; see `src/mapgen/mountains.cpp`), planned on a
world 32 cells larger than the map on every side, so walls run off the map edge
as in the originals.

Steps (`src/mapgen/learned_sketch.cpp`, `learned_tiles.cpp`, `learned_extra.cpp`):

1. **Zones** on the larger world, then the zone guide per sketch cell, and one
   ground per mountain (mud, sand in 6 % of cases as in the originals).
   **Large shape** (`coherence`): the sketch network only sees its
   surroundings, so a height plan of the whole map is made first — uplands and
   basins from soft noise, rastered coarsely so their rims run in straight
   steps as in the originals. The rims become fixed wall cells the network has
   to draw around; small uplands stay closed, large ones get ramps. Mountains
   get nested terraces, and small closed hills are added. `calm_terrain` makes
   the plan larger-scale, drops short cliffs and leaves more building ground
   away from the large lines. A **gradient** biases the network's ground choice
   per place and moves mountains, uplands and basins to the rugged end; on the
   flat end large-shape lines and the network's own cliffs are thinned out.
2. **Sketch**: 16 rounds, cells drawn in random order (most-certain-first would
   leave the map empty: "no wall" is the safest guess almost everywhere). Zones
   are strengthened by comparing with a run without zones; the ground shares
   are steered towards the terrain mix. On maps up to 128x128 two sketches are
   drawn and the one closer to the wanted wall share is kept. With
   `biome_calm` a draft comes first: its grounds are merged into large areas by
   a diffusion that walls block (so borders snap to cliffs), weights corrected
   until the shares match again, then bent with soft noise away from walls; the
   real sketch is drawn on these grounds.
3. **Wall clean-up**: wall cells are joined into chains, short gaps are closed,
   snippets dropped, chains extended inside tight zones; parallel north edges
   (one cell thick) are thinned; every wall and its rim get one ground (with
   `biome_calm` only the wall's own 4x4 blocks are put on dirt or sand, since
   in the originals almost a third of the biome borders run along a cliff).
4. **Symmetry** mirrors the sketch, turning wall directions with it. Then the
   volcano sites are reserved (rock for small volcanoes, a lava core for medium
   and large ones; in the base half and mirrored) and the **lava cores** are
   planned in the rock areas (`lava_rock`: chance and size per area as in the
   originals, following the area's shape). Then the sketch is cut out of the
   world.
5. **Painting**: open cells are filled over several rounds, certain ones first,
   each from the tile network's probabilities, restricted to tiles that fit the
   neighbours already set; dead ends are reopened. With lava rock the tile
   probabilities follow the lava plan (rock, lava rock, the transition tiles at
   the core's rim) and the five base patterns' areas, and rock never touches
   lava rock without a transition tile. Unknown tile pairs are repainted spot
   by spot; with the large shape, sketch walls left without a wall tile (the
   network likes to skip thin north edges) are repainted with a growing pull
   towards wall tiles. Tiny wall bits are painted over, and building ground
   left inside mountains gets an extra terrace (with the large shape only from
   `mountain_density` 0.8). Rock and lava rock are finally laid in the brush
   patterns of the original editors.
6. **Finishing**: specks of a foreign ground dissolve, scatter objects are
   thinned to the `details` setting, rocks and craters close the last building
   spots in mountains and gorges (`learned_rocks`). The volcanoes are stamped
   on their sites (single foreign tiles in the frame become the most common
   base tile), then craters, cracks and debris from the game's tile groups
   (profile section `generator.decorations`), each only on its ground and where
   its seam fits; debris comes as crash sites. Lava-possible zones, spawn
   points.

Quality levels: *sketch* stops after step 4 (for a quick preview), *fast*
paints in 6 rounds, *good* in 24 (fewer odd spots, about 1.5x slower).

## Using it

```c
om_mapgen_model* model = om_mapgen_model_load(net_bytes, net_len, err, sizeof err);   /* profiles/op2-net.bin */
om_mapgen_settings* s = om_mapgen_settings_create("learned");
om_mapgen_set(s, "width", 128);
om_mapgen_set(s, "height", 128);
om_mapgen_set(s, "spawns", 2);
om_mapgen_set_seed(s, 7);
om_mapgen_result* r = om_mapgen_generate_learned(profile, model, s, OM_LEARNED_FAST, 0, progress, user, err, sizeof err);
/* the usual result accessors: om_mapgen_get_tiles, om_mapgen_get_cell_types, om_mapgen_stat, ... */
om_mapgen_result_free(r);
```

```python
model = LearnedModel.from_file("profiles/op2-net.bin")
gen = MapGenerator("learned", width=128, height=128, seed=7, spawns=2)
preview = gen.generate_learned(profile, model, LearnedQuality.SKETCH)   # preview.sketch_walls / sketch_grounds
result = gen.generate_learned(profile, model, LearnedQuality.FAST, progress=lambda f: print(f"{f:.0%}"))
```

```csharp
using (var model = LearnedModel.LoadDefault())          // op2maplib\op2-net.bin next to the binding
using (var gen = new MapGenSettings("learned"))
{
    gen.Set("width", 128); gen.Set("height", 128); gen.Seed = 7;
    GeneratedMap map = gen.GenerateLearned(profile, model, LearnedQuality.Fast, 0, f => true);
}
```

The model can be shared between calls and threads. The progress callback gets
the painted share (0 while sketching) and can cancel. Width and height must be
multiples of 8. The settings that matter: `width`, `height`, the terrain mix
(mud, rock, sand, vegetation; snow and lava are not part of the training
style), `symmetry`, `spawns`, `spawn_clearance`, `cliffs`, `details`,
`mountains`, `mountain_density`, `gorges`, `valleys`, `plateaus`,
`learned_rocks`, `lava_zone_tiles`, and the learned generator's own:
`coherence`, `calm_terrain`, `biome_calm`, `lava_rock`, `volcanoes_small`,
`volcanoes_medium`, `volcanoes_large`, `craters`, `cracks`, `debris`,
`gradient`, `gradient_ground_start`/`_end`, `gradient_terrain_start`/`_end`
(meanings in `op2maplib.h`, `om_mapgen_set`). With all of these at 0 the
generator is the one of library versions before them. Features, `volcanoes`,
the landscape mode and roughness are not used. The template `learned` sets the
suggestions of the editor's template (mix of the training maps, mountains
0.10, density 0.5, two plateaus, cliffs 0.12, details 0.10, coherence 0.6,
calm terrain 0.5, calm biomes 0.5, lava rock 0.5, one small and one medium
volcano, craters 0.3, cracks 0.3, debris 0.15).

```python
gen = MapGenerator("learned", width=256, height=256, seed=7,
                   gradient=1, gradient_ground_start=2,          # north -> south, sand in the north
                   gradient_terrain_start=0.05, gradient_terrain_end=0.95)
```

The result has the zone statistics of `om_mapgen_stat` (`build`,
`build_mountain`, `build_plateau`, `build_valley`, `build_gorge`,
`gorges_passable`, ...), and the sketch at half resolution
(`om_mapgen_sketch_size`, `om_mapgen_get_sketch`).

## Measurements

128x128, template `learned`, six seeds, against op2-codegen-editor 0.11.1 with
the same settings (measured with fefi-byte's map metrics):

| | library | editor | original maps (median) |
|---|---|---|---|
| wall share | 6.3 % | 6.4 % | 8.8 % |
| building ground | 85 % | 86 % | 74 % |
| wall runs per 10k cells | 12.8 | 10.1 | 8.9 |
| median wall run | 33 | 38 | 72 |

Building ground in mountains is 0–20 % of the mountain area, valleys and
plateaus are 95–100 % building ground, and gorges are passable.

With all additions (template `learned`, 128x128, seeds 101–108, the editor
0.12.0 with the same settings):

| | library | editor |
|---|---|---|
| wall share | 7.6 % | 7.4 % |
| wall pieces under 30 tiles | 4.4 | 3.6 |
| biome border within 4 tiles of a cliff | 33.6 % | 34.6 % |
| closed areas (20–2500 tiles) | 3.9 | 2.4 |
| lava rock patches | 2.6 | 2.0 |
| volcanoes | 1.9 | 1.9 |

Gradient north to south, sand and flat in the north, rugged in the south
(256x256, six seeds), per quarter from north to south:

| | library | editor |
|---|---|---|
| sand | 100, 74, 25, 11 % | 100, 88, 39, 7 % |
| cliffs | 0.4, 2.2, 7.4, 9.7 % | 0.1, 1.8, 6.7, 11.0 % |

The additions cost no noticeable time: 256x256 with all of them takes about as
long as without (34 s against 35 s, fast, 8 threads).

Time on one desktop CPU (8 threads): 64x64 about 2 s, 128x128 6
to 9 s (fast) and 10 s (good), 256x256 about 27 s (fast) and 44 s (good). The editor
with ONNX Runtime is about 1.5–2x faster: it uses fused multiply-add and wider
vector units, which this library avoids so every platform gets the same map.

## Same map everywhere

The map depends only on profile, model, settings and seed — not on the
platform, compiler or thread count (`tests/vectors/mapgen_learned.json`,
`mapgen_learned_full.json`). The additions in `learned_extra.cpp` are plain
scalar code in one fixed order, with the library's own `exp` and `log`.
What keeps it so (`src/mapgen/net.cpp`):

- every output value is summed in one fixed order; threads only split
  independent outputs;
- no fused multiply-add: the build turns floating-point contraction off
  (`-ffp-contract=off`; MSVC does not contract by default);
- the vector paths (GCC/Clang vector types, with an AVX2 variant chosen at run
  time on Linux; SSE2 intrinsics for MSVC) do exactly the operations of the
  scalar code, which is also kept;
- `exp`, `log` and `erf` are computed by the library, not taken from the C
  library (which differ in the last bit between platforms);
- sums over many values (GroupNorm) use eight fixed lanes, then the lanes in
  order.

## The network file

`profiles/op2-net.bin` (4.5 MB) holds both networks (weights as float16) and
the tables the generator needs: the tile vocabulary, the tile pairs seen in the
training maps, and per tile of the standard mapping table its cell type,
ground and role. `tools/export_learned_model.py` writes it from the files
op2-codegen-editor ships (`mapview/netz_skizze.onnx`, `netz_kacheln.onnx`,
`netz_tabellen.npz` and its tile model); the format is described there. Like
`profiles/op2.json`, it is derived from the Outpost 2 maps the networks were
trained on.
