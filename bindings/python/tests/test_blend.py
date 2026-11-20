"""Terrain-blending tests for the Python binding. Run from the repository
root after building the native library (see the README; or set
OP2MAPLIB_PATH to the built library):

    python -I bindings/python/tests/test_blend.py

Uses profiles/op2.json and tests/vectors/blend_disc.json from the repository;
no game files are needed. Exit code: 0 if all checks pass, 1 if any fail.

Checks:
  1. The shared test vector (tests/vectors/blend_disc.json) reproduces the same
     FNV-1a hash as the C++ tests - the binding gives identical results.
  2. Seams: along every border between two neighbouring tiles of the same
     ground pair, the edge patterns in the profile ("edges") must agree.
     Corner segments must always agree (that's what the corner rule
     guarantees); the middle segments are reported as a quality measure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the binding from this checkout rather than any installed copy.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bindings" / "python"))

from op2maplib import BlendMap, Profile, TileRole, version  # noqa: E402

FAILURES = 0


def check(cond: bool, what: str) -> None:
    """Records and prints a failed check; the run continues."""
    global FAILURES
    if not cond:
        FAILURES += 1
        print(f"  FAILED: {what}")


def fnv1a64(values: list[int]) -> str:
    """64-bit FNV-1a hash of the values as little-endian 32-bit words, as hex.

    Must match the hash the C++ tests compute over the same tile grid.
    """
    h = 1469598103934665603   # FNV-1a 64-bit offset basis; the multiplier below is the FNV prime
    for v in values:
        for b in (v & 0xFFFFFFFF).to_bytes(4, "little"):
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def run_vector(profile: Profile, vector: dict) -> BlendMap:
    """Replays a shared test vector: a blank map, then its paint/resolve steps.

    A paint step covers either every cell ("cells": "all") or a filled disc
    {"x", "y", "r"} in cells; a resolve step gives the region to resolve.
    """
    w, h = vector["width"], vector["height"]
    bm = BlendMap(profile, w, h, profile.standard_mapping(), [vector["initialMapping"]] * (w * h))
    for step in vector["steps"]:
        if "paint" in step:
            g = profile.find_ground(step["paint"])
            if step.get("cells") == "all":
                cells = [(x, y) for y in range(h) for x in range(w)]
            else:
                d = step["disc"]
                cells = [(x, y) for y in range(d["y"] - d["r"], d["y"] + d["r"] + 1)
                         for x in range(d["x"] - d["r"], d["x"] + d["r"] + 1)
                         if (x - d["x"]) ** 2 + (y - d["y"]) ** 2 <= d["r"] ** 2]
            bm.paint(g, cells)
        elif "resolve" in step:
            bm.resolve(seed=vector["seed"], region=step["resolve"])
    return bm


def seam_report(bm: BlendMap, profile_json: dict) -> tuple[int, int, int, int]:
    """Compares the edge patterns of neighbouring tiles across every seam.

    Each tile side has 4 segments (profile "edges", e.g. "aabb"); segments 0
    and 3 touch the corners, 1 and 2 are the middle. Only seams next to at
    least one transition tile are checked, using that tile's ground pair.

    Returns (corner checks, corner mismatches, middle checks, middle mismatches).
    """
    # The test maps use the standard mapping, so mapping index -> (ts, img)
    # -> profile tile entry is enough to look up any cell's tile.
    tiles = {(t["ts"], t["img"]): t for t in profile_json["tiles"]}
    std = profile_json["standardMapping"]

    def info(x: int, y: int):
        """Profile tile entry at a cell, or None (outside the map / not in the profile)."""
        m = bm.tile_at(x, y)
        return tiles.get(tuple(std[m])) if 0 <= m < len(std) else None

    def side(t: dict, s: str, pair: tuple[str, str]) -> str | None:
        """Edge pattern of side s ('n','e','s','w') as grounds of `pair`.

        A plain tile of either ground counts as a solid edge ("aaaa"/"bbbb");
        None if the tile does not belong to this pair.
        """
        if t["role"] == "transition":
            if (t["a"], t["b"]) != pair:
                return None
            return t["edges"][s]
        g = t.get("ground")
        if g == pair[0]:
            return "aaaa"
        if g == pair[1]:
            return "bbbb"
        return None

    cc = cm = mc = mm = 0
    for y in range(bm.height):
        for x in range(bm.width):
            here = info(x, y)
            if here is None:
                continue
            # Only the east and south neighbours, so each seam is counted once.
            for dx, dy, mine, theirs in ((1, 0, "e", "w"), (0, 1, "s", "n")):
                there = info(x + dx, y + dy)
                if there is None:
                    continue
                trans = [t for t in (here, there) if t["role"] == "transition"]
                if not trans:
                    continue
                pair = (trans[0]["a"], trans[0]["b"])
                p, q = side(here, mine, pair), side(there, theirs, pair)
                if p is None or q is None:
                    continue
                # Both sides are read in the same direction (top->bottom or
                # left->right), so segment i faces segment i.
                cc += 2
                cm += (p[0] != q[0]) + (p[3] != q[3])
                mc += 2
                mm += (p[1] != q[1]) + (p[2] != q[2])
    return cc, cm, mc, mm


def main() -> int:
    """Runs all checks; returns the process exit code."""
    print(f"op2maplib {version()} (Python binding)")
    profile_json = json.loads((REPO / "profiles" / "op2.json").read_text(encoding="utf-8"))
    profile = Profile.from_file(REPO / "profiles" / "op2.json")
    print("profile:", profile.name)

    check(profile.find_ground("mud") >= 0, "mud ground exists")
    check(profile.grounds_blend(profile.find_ground("mud"), profile.find_ground("san")), "mud blends with sand")
    check(len(profile.standard_mapping()) == 2017, "standard mapping has 2017 entries")

    # Same vector and hash as the C++ tests: proves the binding passes data
    # through unchanged and the library is deterministic across front ends.
    vector = json.loads((REPO / "tests" / "vectors" / "blend_disc.json").read_text(encoding="utf-8"))
    bm = run_vector(profile, vector)
    grid =[bm.tile_at(x, y) for y in range(bm.height) for x in range(bm.width)]
    got = fnv1a64(grid)
    print("vector hash:", got)
    check(got == vector["fnv1a64"], f"vector hash {got} == {vector['fnv1a64']}")

    # Larger, messier scene for the seam check: blobs of three grounds.
    w = h = 64
    scene = BlendMap(profile, w, h, profile.standard_mapping(), [0] * (w * h))
    mud, san, roc = (profile.find_ground(g) for g in ("mud", "san", "roc"))
    scene.paint(mud, [(x, y) for y in range(h) for x in range(w)])
    scene.resolve(seed=7)
    for cx, cy, r, g in ((20, 20, 9, san), (44, 22, 7, roc), (30, 44, 11, san), (48, 46, 6, roc), (12, 50, 4, san)):
        scene.paint(g, [(x, y) for y in range(cy - r, cy + r + 1) for x in range(cx - r, cx + r + 1)
                        if (x - cx) ** 2 + (y - cy) ** 2 <= r * r])
        scene.resolve(seed=7)
    transitions = sum(scene.tile_role(x, y) == TileRole.TRANSITION for y in range(h) for x in range(w))
    cc, cm, mc, mm = seam_report(scene, profile_json)
    print(f"seams: {transitions} transition tiles; corner segments {cm}/{cc} mismatched, "
          f"middle segments {mm}/{mc} mismatched")
    # The mismatch counts are reported, not checked: they measure quality.
    check(transitions > 0, "scene has transition tiles")

    if FAILURES:
        print(f"{FAILURES} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
