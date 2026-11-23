"""Export the Outpost 2 terrain-blend profile for op2maplib.

Reads the tile model learned by op2-codegen-editor (mapview/tiles_model.npz,
mapview/celltypes.json) plus one standard campaign map for its tile mapping
table, and writes profiles/op2.json in the op2maplib profile format
(see docs/profile-format.md).

Standard library only: an .npz file is a zip of .npy arrays, which are read
here directly, so numpy is not needed.

Only needed to regenerate profiles/op2.json (which is checked in); users of
the library do not need to run it.

Usage:
    python -I tools/export_op2_profile.py <op2-codegen-editor/mapview> <campaign .map> <out.json>

Example (Windows cmd; ^ continues the line):
    python -I tools/export_op2_profile.py ^
        path/to/op2-codegen-editor/mapview ^
        "path/to/Outpost 2/.../cm02.map" ^
        profiles/op2.json

Inputs read from the mapview folder: tiles_model.npz, celltypes.json,
cliff_kanten.json, cliff_kanten_auto.json, cliff_tiles_model.npz and
cliff_pixels_model.npz. Many keys and constants in those files are German
(e.g. "kanten" = edges, "kacheln" = tiles, "muster" = pattern); the comments
below name the source of each rule. Prints a summary when done; exits with
code 2 (after printing this text) for wrong arguments.
"""
from __future__ import annotations

import ast
import json
import struct
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# .npy / .npz reading (no numpy)

# numpy dtype descr -> (struct format code, item size in bytes)
_DTYPES = {"<i2": ("h", 2), "<i4": ("i", 4), "<i8": ("q", 8), "<f4": ("f", 4), "<f8": ("d", 8),
           "|u1": ("B", 1), "|i1": ("b", 1), "|b1": ("?", 1), "<u2": ("H", 2), "<u4": ("I", 4)}


class Array:
    """A minimal N-d array: flat values plus a shape."""

    def __init__(self, values: list, shape: tuple):
        self.values = values
        self.shape = shape

    def rows(self) -> list:
        """2-d array as a list of row lists (or the flat list for 1-d)."""
        if len(self.shape) == 1:
            return list(self.values)
        width = 1
        for d in self.shape[1:]:
            width *= d
        return [self.values[i:i + width] for i in range(0, len(self.values), width)]


def read_npy(raw: bytes) -> Array:
    """Parses one .npy file (C order; numeric dtypes from _DTYPES or fixed-width
    unicode strings). Raises ValueError for unsupported files."""
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy file")
    # Header length is 2 bytes in format version 1.x and 4 bytes in 2.x/3.x;
    # the header itself is a Python dict literal (descr, fortran_order, shape).
    major = raw[6]
    if major == 1:
        hlen = struct.unpack_from("<H", raw, 8)[0]
        start = 10
    else:
        hlen = struct.unpack_from("<I", raw, 8)[0]
        start = 12
    header = ast.literal_eval(raw[start:start + hlen].decode("latin1"))
    if header["fortran_order"]:
        raise ValueError("fortran-order arrays are not supported")
    shape = tuple(header["shape"])
    count = 1
    for d in shape:
        count *= d
    body = raw[start + hlen:]
    descr = header["descr"]
    if descr.startswith("<U"):
        # Fixed-width UTF-32 strings
        n = int(descr[2:])
        values = []
        for i in range(count):
            chunk = body[i * n * 4:(i + 1) * n * 4]
            values.append(chunk.decode("utf-32-le").rstrip("\x00"))
        return Array(values, shape)
    code, size = _DTYPES[descr]
    values = list(struct.unpack_from(f"<{count}{code}", body, 0))
    return Array(values, shape)


def read_npz(path: Path) -> dict[str, Array]:
    """Reads every array of an .npz file, keyed by name without ".npy"."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            out[name[:-4] if name.endswith(".npy") else name] = read_npy(z.read(name))
    return out


# ---------------------------------------------------------------------------
# .map tile mapping table (layout as documented in op2-codegen-editor/mapview/op2map.py)

def read_map_mappings(path: Path) -> tuple[list[str], list[tuple[int, int]]]:
    """Returns (tileset filenames, [(tilesetIndex, graphicIndex)] per mapping index)."""
    buf = path.read_bytes()
    _ver, _saved, lg_width, height, tileset_count = struct.unpack_from("<IiIII", buf, 0)
    # Skip the 20-byte header, the cell array (4 bytes per cell) and the
    # 16-byte clip rectangle that follows it.
    pos = 20 + (1 << lg_width) * height * 4 + 16
    names = []
    for _ in range(tileset_count):
        ln = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        names.append(buf[pos:pos + ln].decode("ascii", "replace"))
        pos += ln
        if ln > 0:
            pos += 4  # numTiles
    if buf[pos:pos + 10] != b"TILE SET\x1a\x00":
        raise ValueError("'TILE SET' marker not found")
    pos += 10
    count = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    mappings = []
    for _ in range(count):
        # tileset index, graphic index, then two animation fields not needed here
        ts, gfx, _ac, _ad = struct.unpack_from("<4H", buf, pos)
        pos += 8
        mappings.append((ts, gfx))
    # The tileset table has fixed slots; unused ones at the end are empty.
    while names and not names[-1]:
        names.pop()
    return names, mappings


# ---------------------------------------------------------------------------
# Rules mirrored from op2-codegen-editor/mapview/mapgen_modell.py (TileModel)

MIN_USAGE = 20                 # TileModel.MINDEST_NUTZUNG
FAST_PASSIBLE_1 = 0            # cellFastPassible1 (Outpost 2 CellType enum)
GROUND_NAMES = {"mud": "Mud", "roc": "Rock", "san": "Sand", "sno": "Snow",
                "lav": "Lava", "veg": "Vegetation"}

# muster side order in the model: east, west, south, north. East/west read
# top->bottom, south/north read left->right (learn_tiles.kantenmuster).
SIDE_E, SIDE_W, SIDE_S, SIDE_N = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Cliffs (from op2-codegen-editor's cliff tool data)
#
#   cliff_kanten.json       hand-made edge codes (only ids listed in "bearbeitet" count)
#   cliff_kanten_auto.json  generated edge codes (sand); the hand list wins
#   cliff_tiles_model.npz   which cliff tile stood next to which in the original maps
#   cliff_pixels_model.npz  tile images and edge pixel rows
#
# Per cliff tile the profile stores: its family (ground), 4x4 edge codes, the
# share of rock among those codes, a 4x4 inner rock image, the default cell
# type, the neighbours seen in the original maps per direction, and the edge
# pixel rows (for seam scoring). See docs/profile-format.md.

CLIFF_MIN_TILES = 40          # klippe_algo.MINDEST_KACHELN: fewer tiles can't build walls
CLIFF_FALLBACK = "mud"        # paint_layers.KLIPPEN_ERSATZ: cliffs for grounds without their own
ROCK_BRIGHTNESS = 80          # klippe_algo.FELS_HELLIGKEIT: mud inner images
SIDES = ("N", "O", "S", "W")  # O = east, as in the source data


def _percentile(sorted_vals: list[float], q: float) -> float:
    """numpy.percentile with linear interpolation, on an already sorted list."""
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _box5(img: list) -> list:
    """5x5 box blur with edge padding (cliff_kanten_auto._glatt). img: 32x32 of (r,g,b)."""
    n = len(img)
    out = []
    for y in range(n):
        row = []
        for x in range(n):
            s = [0.0, 0.0, 0.0]
            for dy in range(-2, 3):
                yy = min(n - 1, max(0, y + dy))
                for dx in range(-2, 3):
                    xx = min(n - 1, max(0, x + dx))
                    p = img[yy][xx]
                    s[0] += p[0]; s[1] += p[1]; s[2] += p[2]
            row.append((s[0] / 25, s[1] / 25, s[2] / 25))
        out.append(row)
    return out


def _kmeans_centres(blurred: list) -> tuple:
    """Two colour centres (ground, rock) for a family (cliff_kanten_auto.zentren)."""
    px = [p for img in blurred for row in img for p in row]
    bright = sorted(sum(p) / 3 for p in px)
    hi, lo = _percentile(bright, 80), _percentile(bright, 20)

    def mean(ps):
        n = len(ps)
        return (sum(p[0] for p in ps) / n, sum(p[1] for p in ps) / n, sum(p[2] for p in ps) / n)

    c = [mean([p for p in px if sum(p) / 3 > hi]), mean([p for p in px if sum(p) / 3 < lo])]
    for _ in range(20):
        groups = ([], [])
        for p in px:
            d0 = sum((p[i] - c[0][i]) ** 2 for i in range(3))
            d1 = sum((p[i] - c[1][i]) ** 2 for i in range(3))
            groups[0 if d0 <= d1 else 1].append(p)
        c = [mean(groups[i]) if groups[i] else c[i] for i in range(2)]
    if sum(c[0]) < sum(c[1]):
        c = c[::-1]
    return c[0], c[1]      # ground (lighter), rock (darker)


def _blocks4(mask32: list) -> str:
    """32x32 bool -> 16 chars '0'/'1': each 8x8 block is rock if more than half is."""
    out = []
    for by in range(4):
        for bx in range(4):
            n = sum(mask32[y][x] for y in range(by * 8, by * 8 + 8) for x in range(bx * 8, bx * 8 + 8))
            out.append("1" if n / 64 > 0.5 else "0")
    return "".join(out)


def export_cliffs(mapview: Path, mappings: list, family_of: dict, ground_ids: list,
                  default_cell: dict) -> dict:
    """Builds the profile's "cliffs" section from the cliff tool data.

    Args:
        mapview:      the op2-codegen-editor mapview folder.
        mappings:     the standard mapping table, (tileset index, image) per tile.
        family_of:    tile (mapping index) -> index into ground_ids, or -1.
        ground_ids:   ground ids, e.g. ["mud", "roc", ...].
        default_cell: tile -> default Outpost 2 cell type.

    Only tiles with all four edges coded are included.
    """
    hand =json.loads((mapview / "cliff_kanten.json").read_text(encoding="utf-8"))
    done = {int(t) for t in hand.get("bearbeitet", [])}
    edges = {int(k): v for k, v in hand["kanten"].items() if int(k) in done}
    auto = json.loads((mapview / "cliff_kanten_auto.json").read_text(encoding="utf-8"))
    for k, v in auto.get("kanten", {}).items():
        edges.setdefault(int(k), v)

    tiles_m = read_npz(mapview / "cliff_tiles_model.npz")
    px_m = read_npz(mapview / "cliff_pixels_model.npz")
    px_ids = px_m["kacheln"].values
    px_index = {t: i for i, t in enumerate(px_ids)}
    side_names = px_m["kanten"].values                       # order of the edge rows
    rims = px_m["raender"].values                            # (n, 4, 32, 3) uint8
    imgs = px_m["bilder"].values                             # (n, 32, 32, 3) uint8

    def image(t: int) -> list:
        """32x32 image of cliff tile t as rows of (r, g, b)."""
        i = px_index[t]
        base = i * 32 * 32 * 3
        return [[tuple(imgs[base + (y * 32 + x) * 3 + c] for c in range(3)) for x in range(32)]
                for y in range(32)]

    def rim_hex(t: int, side: str) -> str:
        """Edge pixel row of tile t on one side: 32 RGB pixels as 192 hex digits."""
        i = px_index[t]
        s = side_names.index(side)
        base = (i * 4 + s) * 32 * 3
        return bytes(rims[base:base + 96]).hex()

    # Pools: tiles with all four edges known, grouped by ground family.
    pools: dict[str, list] = {}
    for t in sorted(edges):
        e = edges[t]
        if any(s not in e or len(e[s]) != 4 for s in SIDES):
            continue
        f = family_of.get(t, -1)
        if f < 0:
            continue
        pools.setdefault(ground_ids[f], []).append(t)

    # Inner 4x4 rock images: mud by brightness; other families by a 2-colour
    # k-means over all of that family's cliff images.
    inner: dict[int, str] = {}
    for fam, ts in pools.items():
        with_px = [t for t in ts if t in px_index]
        if fam == "mud":
            for t in with_px:
                img = image(t)
                inner[t] = _blocks4([[sum(p) / 3 < ROCK_BRIGHTNESS for p in row] for row in img])
        else:
            # Colour centres from every cliff image of this family, as the source does.
            all_fam = [t for t in px_ids if family_of.get(t, -1) >= 0 and ground_ids[family_of[t]] == fam]
            blurred_all = {t: _box5(image(t)) for t in all_fam}
            ground_c, rock_c = _kmeans_centres(list(blurred_all.values()))
            for t in with_px:
                g = blurred_all.get(t) or _box5(image(t))
                mask = [[sum((p[i] - rock_c[i]) ** 2 for i in range(3)) <
                         sum((p[i] - ground_c[i]) ** 2 for i in range(3)) for p in row] for row in g]
                inner[t] = _blocks4(mask)

    # Neighbours seen in the original maps (straight directions only).
    # nb_a/nb_b index into "kacheln" (tile ids); nb_richtung is the direction,
    # where 0-3 are the straight ones in SIDES order.
    kach = tiles_m["kacheln"].values
    seen: dict[tuple, set] = {}
    for r, a, b in zip(tiles_m["nb_richtung"].values, tiles_m["nb_a"].values, tiles_m["nb_b"].values):
        if r < 4:
            seen.setdefault((kach[a], r), set()).add(kach[b])

    # Output order (family, then tile id) defines each tile's index in the
    # profile; "seen" lists refer to those indices, and only within a family.
    out_tiles = []
    index_of: dict[int, int] = {}
    order = [(fam, t) for fam in sorted(pools) for t in pools[fam]]
    for fam, t in order:
        index_of[t] = len(index_of)
    for fam, t in order:
        e = edges[t]
        ts, img = mappings[t]
        entry = {
            "ts": ts, "img": img, "family": fam,
            "edges": " ".join("".join(str(int(v)) for v in e[s]) for s in SIDES),
            "inner": inner.get(t),
            "cell": default_cell.get(t, 4),
            "seen": {s.lower() if s != "O" else "e": sorted(index_of[u] for u in seen.get((t, ri), ())
                                                         if u in index_of and order[index_of[u]][0] == fam)
                     for ri, s in enumerate(SIDES)},
            "rims": {(s.lower() if s != "O" else "e"): rim_hex(t, s) for s in SIDES} if t in px_index else None,
        }
        out_tiles.append(entry)

    families = {fam: len(ts) for fam, ts in pools.items()}
    return {
        "minTiles": CLIFF_MIN_TILES,
        "fallback": CLIFF_FALLBACK,
        "families": families,
        "tiles": out_tiles,
    }


def export_generator(m: dict, mappings: list, default_cell: dict) -> dict:
    """The map generator's data (profile section "generator", see docs/profile-format.md).

    - details:   scatter stamps (rocks, cracks, craters) larger than 1x1, with
                 the ground they belong on (TileModel.details).
    - volcanoes: volcano stamps cut from the original maps (TileModel.volcanoes).
    - right/down: tile pairs seen side by side in the original maps, used to
                 check a stamp's seam against the ground around it. Each tile
                 is packed as tileset * 4096 + image.

    Stamp tiles are written as [tileset, image, cell type]; most of them are
    structure or decoration tiles that the "tiles" section leaves out.
    """
    def ref(t: int) -> list[int]:
        ts, img = mappings[t]
        return [ts, img, default_cell.get(t, -1)]

    def packed(t: int) -> int:
        ts, img = mappings[t]
        return ts * 4096 + img

    def stamps(sizes, offsets, flat):
        """(w, h, tile refs) per stamp of a size/offset/tile triple."""
        out = []
        for i, (w, h) in enumerate(sizes.rows()):
            block = flat.values[offsets.values[i]:offsets.values[i + 1]]
            out.append((int(w), int(h), [ref(t) for t in block] if len(block) == w * h else None))
        return out

    details = []
    fams = m["detail_families"].values
    for i, (w, h, tiles) in enumerate(stamps(m["detail_sizes"], m["detail_offsets"], m["detail_tiles"])):
        # 1x1 stamps are skipped, as in TileModel: a single tile looks like noise.
        if tiles is not None and w * h > 1:
            details.append({"ground": fams[i] if i < len(fams) else "", "w": w, "h": h, "tiles": tiles})

    volcanoes = []
    names = m["volcano_names"].values
    for i, (w, h, tiles) in enumerate(stamps(m["volcano_sizes"], m["volcano_offsets"], m["volcano_tiles"])):
        if names[i] and tiles is not None and w * h:
            volcanoes.append({"name": names[i], "w": w, "h": h, "tiles": tiles})

    def seen(key: str) -> list[int]:
        """Distinct (left/top, right/bottom) pairs of one adjacency table, packed,
        as one flat array a0, b0, a1, b1, ... (main() writes it on one line)."""
        out = set()
        for a, b, _count in m[key].rows():
            if a < len(mappings) and b < len(mappings):
                out.add((packed(a), packed(b)))
        return [v for p in sorted(out) for v in p]

    return {"details": details, "volcanoes": volcanoes, "right": seen("right"), "down": seen("down")}


def main(argv: list[str]) -> int:
    """Exports the profile; argv as sys.argv. Returns the process exit code."""
    if len(argv) != 4:
        print(__doc__)
        return 2
    mapview = Path(argv[1])
    map_path = Path(argv[2])
    out_path = Path(argv[3])

    m = read_npz(mapview / "tiles_model.npz")
    celltypes = json.loads((mapview / "celltypes.json").read_text(encoding="utf-8"))
    default_cell = {int(k): int(v) for k, v in celltypes["default"].items()}
    tilesets, mappings = read_map_mappings(map_path)

    families = list(m["families"].values)          # e.g. mud roc san sno lav
    ground_ids = families + ["veg"]                 # veg appended as in TileModel
    veg = len(families)

    usage = {t: n for t, n in m["usage"].rows()}          # tile -> times used in the original maps
    family_of = {t: f for t, f in m["family"].rows()}     # tile -> ground index, -1 = none
    # Vegetation is not a family in the model: tiles whose default cell type
    # is fast-passable-1 form the extra "veg" ground.
    for t, ct in default_cell.items():
        if ct == FAST_PASSIBLE_1:
            family_of[t] = veg

    structure = set(m["structure"].values)
    # Decoration (detail stamps larger than 1x1, plus deko groups) is never free fill.
    deco = set(m["deko"].values)
    sizes = m["detail_sizes"].rows()
    offs = m["detail_offsets"].values
    dtiles = m["detail_tiles"].values
    for i, (w, h) in enumerate(sizes):
        if w * h > 1:
            deco.update(dtiles[offs[i]:offs[i + 1]])

    trans_tiles = m["trans_tiles"].values
    trans_a = m["trans_a"].values
    trans_b = m["trans_b"].values
    trans_set = set(trans_tiles)
    muster = m["muster"].values                     # (2017, 4, 4) family index per segment

    def seg(tile: int, side: int) -> list[int]:
        """The 4 segment family indices of one side (SIDE_*) of a tile."""
        base = (tile * 4 + side) * 4
        return muster[base:base + 4]

    def tile_ref(t: int) -> tuple[int, int]:
        """(tileset index, image index) of mapping index t."""
        return mappings[t]

    tiles_out = []
    corner_conflicts = 0
    pairs: dict[tuple[str, str], int] = {}

    # Transition tiles
    for t, a, b in zip(trans_tiles, trans_a, trans_b):
        a_i, b_i = ground_ids.index(a), ground_ids.index(b)
        n, e, s, w = seg(t, SIDE_N), seg(t, SIDE_E), seg(t, SIDE_S), seg(t, SIDE_W)

        def letter(v: int) -> str:
            """Segment family index -> 'a'/'b' of this tile's ground pair."""
            return "b" if v == b_i else "a"

        edges = {"n": "".join(letter(v) for v in n), "e": "".join(letter(v) for v in e),
                 "s": "".join(letter(v) for v in s), "w": "".join(letter(v) for v in w)}
        # Corners from the two segments that meet there:
        #   NW = N[0] & W[0]   NE = N[3] & E[0]   SE = S[3] & E[3]   SW = S[0] & W[3]
        corner_segs = ((n[0], w[0]), (n[3], e[0]), (s[3], e[3]), (s[0], w[3]))
        corners = ""
        for p, q in corner_segs:
            if p != q:
                corner_conflicts += 1
                # The segments disagree; the tile's border changes right at
                # the corner. Count it as 'b' if either side says b, matching
                # the placement rule (a corner is b if any adjoining cell is b).
                corners += "b"
            else:
                corners += letter(p)
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
        ts, img = tile_ref(t)
        tiles_out.append({
            "ts": ts, "img": img, "role": "transition", "a": a, "b": b,
            "corners": corners, "edges": edges,
            "w": int(usage.get(t, 0)), "cell": default_cell.get(t),
        })

    # Plain ground (fill pool) and classify-only ground tiles
    plain_counts = {g: 0 for g in ground_ids}
    skipped_structure = 0
    for t in range(len(mappings)):
        if t in trans_set or t not in family_of or family_of[t] < 0:
            continue
        # Structure tiles (cliffs, walls, tubes, volcanoes, lava, bulldozed,
        # wreckage) are left out of the profile entirely, so the library treats
        # them as fixed: never replaced by a transition or fill tile unless the
        # user paints over them.
        if t in structure:
            skipped_structure += 1
            continue
        g = ground_ids[family_of[t]]
        placeable = (usage.get(t, 0) >= MIN_USAGE and t not in deco)
        role = "plain" if placeable else "ground"
        if placeable:
            plain_counts[g] += 1
        ts, img = tile_ref(t)
        tiles_out.append({
            "ts": ts, "img": img, "role": role, "ground": g,
            "w": int(usage.get(t, 0)), "cell": default_cell.get(t),
        })

    # Deterministic output order: by tileset, then image.
    tiles_out.sort(key=lambda d: (d["ts"], d["img"]))

    profile = {
        "format": "op2maplib.blend-profile",
        "version": 1,
        "name": "Outpost 2 standard tilesets (well0000-well0012)",
        "source": "Learned from the Outpost 2 campaign maps by op2-codegen-editor "
                  "(tiles_model.npz, celltypes.json); exported by tools/export_op2_profile.py",
        "tilesets": tilesets,
        "grounds": [{"id": g, "name": GROUND_NAMES.get(g, g)} for g in ground_ids],
        "pairs": [{"a": a, "b": b} for (a, b) in sorted(pairs)],
        "tiles": tiles_out,
        "standardMapping": [[ts, img] for ts, img in mappings],
        "cliffs": export_cliffs(mapview, mappings, family_of, ground_ids, default_cell),
        "generator": export_generator(m, mappings, default_cell),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The two pair tables hold ~60,000 numbers; indented they would put each on
    # its own line and quadruple the file. Write them compact, on one line each.
    gen = profile["generator"]
    compact = {k: gen[k] for k in ("right", "down")}
    for k in compact:
        gen[k] = f"@@{k}@@"
    text = json.dumps(profile, indent=1, separators=(",", ":"))
    for k, values in compact.items():
        text = text.replace(f'"@@{k}@@"', json.dumps(values, separators=(",", ":")))
        gen[k] = values
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.write("\n")

    # Summary
    print(f"tilesets: {len(tilesets)}, standard mappings: {len(mappings)}")
    print(f"grounds: {ground_ids}")
    print("transition pairs:", {f"{a}|{b}": n for (a, b), n in sorted(pairs.items())})
    print("plain fill tiles per ground:", plain_counts)
    print(f"tiles written: {len(tiles_out)}, structure tiles left out (fixed): {skipped_structure}, "
          f"corner conflicts: {corner_conflicts} of {4 * len(trans_tiles)}")
    # Coverage of the 16 corner keys per pair (each key is one combination of
    # a/b at the four corners; the two solid ones are plain tiles, not transitions)
    for (a, b) in sorted(pairs):
        keys = {}
        for d in tiles_out:
            if d["role"] == "transition" and d["a"] == a and d["b"] == b:
                keys[d["corners"]] = keys.get(d["corners"], 0) + 1
        missing = [k for k in (f"{i:04b}".replace("0", "a").replace("1", "b") for i in range(16))
                   if k not in keys and k not in ("aaaa", "bbbb")]
        print(f"  {a}|{b}: {len(keys)} corner keys used, missing: {missing}")
    cl = profile["cliffs"]
    no_inner = sum(1 for t in cl["tiles"] if t["inner"] is None)
    print(f"cliffs: {cl['families']} (min {cl['minTiles']} to draw), "
          f"{len(cl['tiles'])} tiles, {no_inner} without inner image")
    gen = profile["generator"]
    print(f"generator: {len(gen['details'])} detail stamps, {len(gen['volcanoes'])} volcanoes, "
          f"{len(gen['right']) // 2} right pairs, {len(gen['down']) // 2} down pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
