"""Export the learned map generator's networks for op2maplib.

Reads the two networks and the tile tables that op2-codegen-editor ships for
its learned map generator (mapview/netz_skizze.onnx, netz_kacheln.onnx,
netz_tabellen.npz) plus its tile model, and writes profiles/op2-net.bin, the
network file of om_mapgen_model_load (see docs/learned-generator.md).

Needs numpy and onnx (pip install numpy onnx) and imports the editor's mapview
modules for the per-tile tables. Only needed to regenerate profiles/op2-net.bin
(which is checked in); users of the library do not need to run it.

Usage:
    python tools/export_learned_model.py <op2-codegen-editor/mapview> <profiles/op2.json> <out.bin>

File format (little endian): "OMNN", u32 version = 1, u32 tensor count, then
per tensor: u16 name length, name (UTF-8), u8 type (0 float32, 1 float16,
2 uint8, 3 uint16), u8 rank, u32 dims[rank], the values. Tensors:
    sketch.*, tiles.*        network weights (float16), named as the PyTorch
                             modules (see src/mapgen/net.cpp)
    vocab                    tile network vocabulary -> standard mapping index
    vocab_cell_types         cell type per vocabulary tile
    pairs_right, pairs_down  tile pairs seen side by side in the training maps
                             (numpy.packbits rows, big-endian bit order)
    mapping_cell_types       per standard mapping index: default cell type,
    mapping_grounds          ground (profile ground index, 255 = none),
    mapping_flags            1 placeable fill, 2 transition, 4 part of a scatter stamp

German names in the editor's files: netz = network, skizze = sketch, kacheln =
tiles, tabellen = tables, rechts/unten = right/down, vok = vocabulary.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def onnx_weights(path: Path) -> dict[str, np.ndarray]:
    """Named weights of an exported network, with the GroupNorm weights the
    exporter left unnamed (onnx::Mul_N / onnx::Add_N after each
    InstanceNormalization) named after the block that follows them."""
    m = onnx.load(str(path))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    out = {}
    pending = []   # unnamed (Mul, Add) constants waiting for their block
    for node in m.graph.node:
        consts = [i for i in node.input if i in init]
        for name in consts:
            if name.startswith("m."):
                out[name[2:]] = init[name]
        if node.op_type in ("Mul", "Add") and consts and consts[0].startswith("onnx::"):
            pending.append(init[consts[0]].reshape(-1))
        if node.op_type == "Conv" and consts and consts[0].endswith(".a.weight") and pending:
            block = consts[0][2:-len(".a.weight")]
            out[block + ".n.weight"], out[block + ".n.bias"] = pending
            pending = []
    if len(pending) == 2:   # the final GroupNorm
        out["n.weight"], out["n.bias"] = pending
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    mapview, profile_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    sys.path.insert(0, str(mapview))
    import celltypes   # noqa: E402  (editor module)
    import mapgen      # noqa: E402  (editor module)

    sketch = onnx_weights(mapview / "netz_skizze.onnx")
    tiles = onnx_weights(mapview / "netz_kacheln.onnx")
    d = np.load(mapview / "netz_tabellen.npz")
    vok = d["vok"].astype(np.int64)

    tm = mapgen.model()
    n_map = len(json.loads(profile_path.read_text(encoding="utf-8"))["standardMapping"])
    detail = np.zeros(n_map, bool)
    for block, _ground in tm.details:
        u = np.unique(block)
        detail[u[u < n_map]] = True
    fam = np.array([tm.family_of[t] if t < len(tm.family_of) else -1 for t in range(n_map)])
    flags = np.zeros(n_map, np.uint8)
    for t in range(n_map):
        if t < len(tm.placeable) and tm.placeable[t]:
            flags[t] |= 1
        if t < len(tm.ist_uebergang) and tm.ist_uebergang[t]:
            flags[t] |= 2
        if detail[t]:
            flags[t] |= 4

    tensors = [("sketch." + k, v.astype(np.float16)) for k, v in sketch.items()]
    tensors += [("tiles." + k, v.astype(np.float16)) for k, v in tiles.items()]
    tensors += [
        ("vocab", vok.astype(np.uint16)),
        ("vocab_cell_types", np.array([celltypes.default_cell_type(int(t)) for t in vok], np.uint8)),
        ("mapping_cell_types", np.array([celltypes.default_cell_type(t) for t in range(n_map)], np.uint8)),
        ("mapping_grounds", np.where(fam < 0, 255, fam).astype(np.uint8)),
        ("mapping_flags", flags),
        ("pairs_right", d["rechts"].astype(np.uint8)),
        ("pairs_down", d["unten"].astype(np.uint8)),
    ]
    codes = {np.dtype(np.float32): 0, np.dtype(np.float16): 1, np.dtype(np.uint8): 2, np.dtype(np.uint16): 3}
    with open(out, "wb") as f:
        f.write(b"OMNN" + struct.pack("<II", 1, len(tensors)))
        for name, a in tensors:
            a = np.ascontiguousarray(a)
            raw = name.encode()
            f.write(struct.pack("<H", len(raw)) + raw + struct.pack("<BB", codes[a.dtype], a.ndim))
            f.write(struct.pack("<%dI" % a.ndim, *a.shape))
            f.write(a.astype(a.dtype.newbyteorder("<")).tobytes())
    print(f"{out}: {out.stat().st_size} bytes, {len(tensors)} tensors "
          f"({len(sketch)} sketch, {len(tiles)} tile network weights, vocabulary {len(vok)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
