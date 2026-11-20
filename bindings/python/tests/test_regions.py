"""Regions (enclosed areas) tests, run from the repository root after building:

    python -I bindings/python/tests/test_regions.py <maps folder>

The maps folder is the first argument, or else the OP2_MAPS environment
variable; with neither, the script prints a usage message and exits with
code 2. It is searched recursively for .map files (e.g. the game's maps
folder).

1. Small hand-made cases (a walled pocket, diagonal gaps with 8- and
   4-connectivity, the passable cell-type mask).
2. Equivalence with OP2Mapper's original "Find & Fix Enclosed Areas" (the
   VB.NET flood fill this module replaces): both are run on every .map file
   in the maps folder and must find the same main area and enclosed cells.

Exit code: 0 if all checks pass, 1 if any fail, 2 for a missing maps folder.
"""
from __future__ import annotations

import os
import struct
import sys
from collections import deque
from pathlib import Path

# Import the binding from this checkout rather than any installed copy.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bindings" / "python"))

from op2maplib import Regions, version  # noqa: E402

FAILURES = 0
# Passable Outpost 2 cell types, copied from OP2Mapper's VB IsPassableCellType.
PASSABLE = {0, 2, 3, 4, 5, 7, 21, 22, 26, 27, 28, 29, 30, 31}


def check(cond: bool, what: str) -> None:
    """Records and prints a failed check; the run continues."""
    global FAILURES
    if not cond:
        FAILURES += 1
        print(f"  FAILED: {what}")


def read_cell_types(path: Path) -> tuple[int, int, list[int]]:
    """Reads (width, height, cell types row by row) from an Outpost 2 .map file."""
    buf = path.read_bytes()
    # 20-byte header: version, saved-game flag, log2(width), height, tileset count.
    _ver, _saved, lg_w, h, _ts = struct.unpack_from("<IiIII", buf, 0)
    w = 1 << lg_w
    # Then one 32-bit word per cell; the cell type is its low 5 bits.
    tiles = struct.unpack_from(f"<{w * h}I", buf, 20)
    return w, h, [t & 0x1F for t in tiles]


def vb_components(w: int, h: int, cells: list[int]) -> list[list[tuple[int, int]]]:
    """Line-for-line port of fMain.FindConnectedPassableComponents.

    The reference implementation: an 8-connected BFS flood fill that skips the
    bottom row and wraps east-west on 512x256 world maps. Returns the
    components (lists of (x, y) cells) in discovery order, unsorted.
    """
    comps = []
    max_y = h - 2                  # the bottom row is never part of a region
    wrap = w == 512 and h == 256   # world maps join their left and right edges
    visited = [[False] * h for _ in range(w)]   # indexed [x][y], as in the VB code
    for y in range(max_y + 1):
        for x in range(w):
            if visited[x][y]:
                continue
            visited[x][y] = True
            if cells[y * w + x] not in PASSABLE:
                continue
            comp = []
            q = deque([(x, y)])
            while q:
                px, py = q.popleft()
                comp.append((px, py))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = px + dx, py + dy
                        if wrap:
                            nx %= w
                        elif nx < 0 or nx >= w:
                            continue
                        if ny < 0 or ny > max_y or visited[nx][ny]:
                            continue
                        visited[nx][ny] = True
                        if cells[ny * w + nx] in PASSABLE:
                            q.append((nx, ny))
            comps.append(comp)
    return comps


def main() -> int:
    """Runs all checks; returns the process exit code."""
    # Resolve the maps folder first, so a missing argument fails fast.
    folder_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OP2_MAPS")
    if not folder_arg:
        print("usage: python -I bindings/python/tests/test_regions.py <maps folder>\n"
              "       (or set the OP2_MAPS environment variable to the maps folder)")
        return 2
    folder = Path(folder_arg)

    print(f"op2maplib {version()} - regions")

    # 1. Hand-made cases. Cell type 0 is passable, 23 is a wall.
    w, h = 20, 12
    cells = [0] * (w * h)
    for y in range(2, 7):
        for x in range(2, 7):
            if x in (2, 6) or y in (2, 6):
                cells[y * w + x] = 23
    r = Regions.find(w, h, cells)
    check(len(r) == 2 and r.size(1) == 9, "wall ring makes a 9-cell pocket")
    check(sorted(r.enclosed_cells()) == [(x, y) for x in range(3, 6) for y in range(3, 6)], "pocket cells")
    check(Regions.passable_mask() == sum(1 << c for c in PASSABLE), "passable mask matches OP2Mapper")
    # Two passable cells touching only at a corner, in an all-wall map.
    diag = [23] * (w * h)
    diag[2 * w + 2] = 0
    diag[3 * w + 3] = 0
    check(len(Regions.find(w, h, diag, flags=Regions.CONNECT8)) == 1, "8-connected joins diagonals")
    check(len(Regions.find(w, h, diag, flags=0)) == 2, "4-connected does not")

    # 2. Same results as the original on real maps
    maps = sorted(folder.rglob("*.map"))
    print(f"comparing with OP2Mapper's original on {len(maps)} maps in {folder}")
    with_enclosed = same = ties = 0
    for p in maps:
        try:
            w, h, cells = read_cell_types(p)
        except Exception as e:   # noqa: BLE001 - skip files that aren't plain maps
            print(f"  skipped {p.name}: {e}")
            continue
        old = sorted(vb_components(w, h, cells), key=len, reverse=True)
        new = Regions.find(w, h, cells)
        if len(old) != len(new) or [len(c) for c in old] != [new.size(i) for i in range(len(new))]:
            check(False, f"{p.name}: region sizes differ")
            continue
        if len(old) >= 2 and len(old[0]) == len(old[1]):
            ties += 1      # main area ambiguous in the original (unstable sort): compare sizes only
            continue
        old_enclosed = sorted(c for comp in old[1:] for c in comp)
        new_enclosed = sorted(new.enclosed_cells())
        check(old_enclosed == new_enclosed, f"{p.name}: enclosed cells differ")
        if old_enclosed == new_enclosed:
            same += 1
        if len(old) > 1:
            with_enclosed += 1
    print(f"  {same} maps identical ({with_enclosed} of them have enclosed areas), {ties} with a tie for largest")

    if FAILURES:
        print(f"{FAILURES} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
