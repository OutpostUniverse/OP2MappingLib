# The learned map generator

`om_mapgen_generate_learned` makes maps in the style of the original Outpost 2
maps. It is fefi-byte's learned generator from op2-codegen-editor
(`mapview/mapgen_netz.py`, `mapgen_netz_waende.py`, version 0.11.1), ported to
C++ with a small inference engine of its own, so the library still has no
dependencies.

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

Steps (`src/mapgen/learned_sketch.cpp`, `learned_tiles.cpp`):

1. **Zones** on the larger world, then the zone guide per sketch cell, and one
   ground per mountain (mud, sand in 6 % of cases as in the originals).
2. **Sketch**: 16 rounds, cells drawn in random order (most-certain-first would
   leave the map empty: "no wall" is the safest guess almost everywhere). Zones
   are strengthened by comparing with a run without zones; the ground shares
   are steered towards the terrain mix. On maps up to 128x128 two sketches are
   drawn and the one closer to the wanted wall share is kept.
3. **Wall clean-up**: wall cells are joined into chains, short gaps are closed,
   snippets dropped, chains extended inside tight zones; parallel north edges
   (one cell thick) are thinned; every wall and its rim get one ground.
4. **Symmetry** mirrors the sketch, turning wall directions with it; then the
   sketch is cut out of the world.
5. **Painting**: open cells are filled over several rounds, certain ones first,
   each from the tile network's probabilities, restricted to tiles that fit the
   neighbours already set; dead ends are reopened. Unknown tile pairs are
   repainted spot by spot, tiny wall bits painted over, and building ground
   left inside mountains gets an extra terrace.
6. **Finishing**: specks of a foreign ground dissolve, scatter objects are
   thinned to the `details` setting, rocks and craters close the last building
   spots in mountains and gorges (`learned_rocks`), spawn points.

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
`learned_rocks`. Features, volcanoes, the landscape mode and roughness are not
used. The template `learned` sets the suggestions (mix of the training maps,
mountains 0.10, density 0.5, two plateaus, cliffs 0.12, details 0.10).

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

Time on one desktop CPU (8 threads): 64x64 about 2 s, 128x128 6
to 9 s (fast) and 10 s (good), 256x256 about 27 s (fast) and 44 s (good). The editor
with ONNX Runtime is about 1.5–2x faster: it uses fused multiply-add and wider
vector units, which this library avoids so every platform gets the same map.

## Same map everywhere

The map depends only on profile, model, settings and seed — not on the
platform, compiler or thread count (`tests/vectors/mapgen_learned.json`).
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
