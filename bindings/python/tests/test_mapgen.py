"""Map generator tests for op2maplib (via the Python binding).

Run from the repository root after building the native library (see the
README; or set OP2MAPLIB_PATH to the built library):

    python -I bindings/python/tests/test_mapgen.py

Uses profiles/op2.json and tests/vectors/mapgen_islands.json from the
repository; no game files are needed. Exit code: 0 if all checks pass, 1 if
any fail.

Checks:
  1. The shared golden map (tests/vectors/mapgen_islands.json) reproduces the
     same hash as the C++ tests.
  2. Every template generates a plausible map at 128x128 (passable share,
     spawns connected, ground shares near the mix), and prints its timing.
  3. Features, volcanoes and lava zones, symmetry and settings errors.
  4. Seam quality: the share of neighbouring tile pairs (right and down) that
     were never seen side by side in the original maps, measured on the
     profile's own pair tables, per template.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bindings" / "python"))

from op2maplib import FeatureKind, MapGenerator, Profile, Symmetry, version  # noqa: E402

failures = 0


def check(ok: bool, what: str) -> None:
    """Prints a failed check and counts it."""
    global failures
    if not ok:
        failures += 1
        print("FAIL", what)


def fnv1a64(values: list[int]) -> str:
    """FNV-1a 64 over each value's 4 little-endian bytes (as the C++ tests)."""
    h = 1469598103934665603
    for v in values:
        for b in range(4):
            h ^= (v >> (8 * b)) & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def unseen_pairs(profile_json: dict, tiles: list[int], w: int, h: int) -> float:
    """Share of right/down neighbour pairs never seen in the original maps.

    Pairs whose first tile has no recorded neighbours at all are skipped, as in
    the generator's own seam check.
    """
    sm = profile_json["standardMapping"]
    packed = [ts * 4096 + img for ts, img in sm]
    gen = profile_json["generator"]
    tables = {}
    for key in ("right", "down"):
        flat = gen[key]
        pairs = {(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)}
        tables[key] = (pairs, {a for a, _ in pairs})
    bad = total = 0
    for y in range(h):
        for x in range(w):
            a = packed[tiles[y * w + x]]
            for key, (dx, dy) in (("right", (1, 0)), ("down", (0, 1))):
                if x + dx >= w or y + dy >= h:
                    continue
                pairs, known = tables[key]
                if a not in known:
                    continue
                total += 1
                bad += (a, packed[tiles[(y + dy) * w + x + dx]]) not in pairs
    return bad / total if total else 0.0


def main() -> int:
    print(f"op2maplib {version()} (Python binding)")
    profile = Profile.from_file(REPO / "profiles" / "op2.json")
    pj = json.loads((REPO / "profiles" / "op2.json").read_text(encoding="utf-8"))
    ids = [profile.ground_id(g) for g in range(profile.ground_count)]

    # 1. Golden map, shared with the C++ tests.
    vec = json.loads((REPO / "tests" / "vectors" / "mapgen_islands.json").read_text(encoding="utf-8"))
    gen = MapGenerator(vec["template"], seed=vec["seed"], **vec["settings"])
    for f in vec["features"]:
        gen.add_feature(f["kind"], f["where"], f["to"], f["size"], f["lava"])
    r = gen.generate(profile)
    got = fnv1a64(r.tiles)
    print(f"golden map hash: {got}")
    check(got == vec["fnv1a64"], f"golden map hash {got} == {vec['fnv1a64']}")
    check(gen.generate(profile).tiles == r.tiles, "same seed gives the same map")
    check(MapGenerator(vec["template"], seed=vec["seed"] + 1, **vec["settings"]).generate(profile).tiles != r.tiles,
          "another seed gives another map")

    # 2. Templates at 128x128.
    print(f"\n{'template':9s} {'ms':>6s} {'passable':>8s} {'plateau':>7s} {'details':>7s} {'unseen':>6s}  shares")
    for tpl, extra in (("plains", {}), ("canyon", {"cliffs": 0.3}), ("desert", {}), ("islands", {}),
                       ("ice", {}), ("newterra", {})):
        g = MapGenerator(tpl, width=128, height=128, seed=1234, spawns=2, **extra)
        t = time.perf_counter()
        m = g.generate(profile)
        ms = (time.perf_counter() - t) * 1000
        shares = {ids[i]: round(s, 2) for i, s in enumerate(m.ground_shares) if s > 0.005}
        unseen = unseen_pairs(pj, m.tiles, m.width, m.height)
        print(f"{tpl:9s} {ms:6.0f} {m.stats['passable']:8.3f} {m.stats['plateau']:7.3f} "
              f"{m.stats['details']:7.3f} {100 * unseen:5.1f}%  {shares}")
        check(m.stats["passable"] > 0.85, f"{tpl}: mostly passable")
        check(len(m.spawns) == 2 and m.stats["connected"] == 1.0, f"{tpl}: 2 connected spawns")
        check(len(m.tiles) == 128 * 128 and len(m.cell_types) == 128 * 128, f"{tpl}: full grids")
        if tpl == "canyon":
            check(m.stats["plateau"] > 0.01, "canyon with cliffs has cliff cells")
        if tpl == "newterra":
            check(len(m.volcanoes) == 1 and m.lava_zones and m.stats["rivers"] > 0,
                  "newterra: a volcano with a lava zone and riverbeds")

    # 3. Features, symmetry, errors.
    g = MapGenerator("plains", seed=7)
    g.add_feature(FeatureKind.VALLEY, "w", "o", 0.6, lava=True)
    g.add_feature(FeatureKind.MOUNTAIN, "c", size=0.7)
    g.add_feature(FeatureKind.PLATEAU, "nw")
    g.add_feature(FeatureKind.CLEARING, "so")
    g.add_feature(FeatureKind.CHOKE, "n", "s")
    m = g.generate(profile)
    print(f"\nfeatures {m.features}\nreport {m.feature_report}, lava zones {[len(z) for z in m.lava_zones]}")
    check(len(m.features) == 5, "five features placed")
    check(m.feature_report[FeatureKind.MOUNTAIN] > 0 and m.feature_report[FeatureKind.CHOKE] > 0,
          "mountain and choke produce cliffs")
    check(m.feature_report[FeatureKind.CLEARING] == 1, "clearing counted")
    check(len(m.lava_zones) >= 1, "lava valley gives a lava zone")

    for sym in (Symmetry.MIRROR_X, Symmetry.MIRROR_Y, Symmetry.ROT180, Symmetry.QUAD):
        m = MapGenerator("islands", seed=3, symmetry=sym, spawns=4 if sym == Symmetry.QUAD else 2,
                         cliffs=0.2).generate(profile)
        w, h = m.width, m.height
        # The cell types (passability) are symmetric; tiles are not mirrored pictures.
        def mirror(x, y):
            return {Symmetry.MIRROR_X: (w - 1 - x, y), Symmetry.MIRROR_Y: (x, h - 1 - y),
                    Symmetry.ROT180: (w - 1 - x, h - 1 - y), Symmetry.QUAD: (w - 1 - x, y)}[sym]
        passable = {0, 2, 3, 4, 5, 7, 21, 22, 26}
        same = sum((m.cell_types[y * w + x] in passable) ==
                   (m.cell_types[mirror(x, y)[1] * w + mirror(x, y)[0]] in passable)
                   for y in range(h) for x in range(w))
        print(f"symmetry {sym}: passability mirrored for {100 * same / (w * h):.1f} % of cells, spawns {m.spawns}")
        check(same / (w * h) > 0.95, f"symmetry {sym}: passability mostly mirrored")

    try:
        MapGenerator("plains", width=8).generate(profile)
        check(False, "a too small map is refused")
    except Exception as e:   # Op2MapLibError
        check("size" in str(e), f"error names the size ({e})")
    try:
        MapGenerator("plains", no_such_setting=1)
        check(False, "an unknown setting is refused")
    except KeyError:
        pass

    print("\nall checks passed" if not failures else f"\n{failures} check(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
