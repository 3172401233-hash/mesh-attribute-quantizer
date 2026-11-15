"""Mesh attribute quantizer.

Quantizes mesh vertex attributes (position, normal, UV, blend weights) into
configurable bit widths, welds identical quantized vertices so shared edges
stay crack-free, and packs the result into an interleaved vertex buffer with
component offsets and alignment information.

Pure Python, no third-party dependencies.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

POSITION = "position"
NORMAL = "normal"
UV = "uv"
WEIGHT = "weight"


@dataclass(frozen=True)
class AttributeSpec:
    """Configuration for one quantized attribute.

    bits:      bits per stored component.
    max_error: error budget in decoded units (positions/UVs: absolute L-inf,
               normals: L2 on the unit sphere, weights: absolute L-inf).
    """

    kind: str
    bits: int
    max_error: float

    @property
    def components(self) -> int:
        return {POSITION: 3, NORMAL: 2, UV: 2, WEIGHT: 4}[self.kind]

    @property
    def levels(self) -> int:
        if not 2 <= self.bits <= 32:
            raise ValueError(f"bits must be in [2, 32], got {self.bits}")
        return (1 << self.bits) - 1


@dataclass
class Mesh:
    """Indexed triangle mesh with float attributes."""

    positions: list  # [(x, y, z), ...]
    indices: list  # flat triangle indices
    normals: list = field(default_factory=list)  # [(x, y, z), ...] or []
    uvs: list = field(default_factory=list)  # [(u, v), ...] or []
    weights: list = field(default_factory=list)  # [(w0, w1, w2, w3), ...] or []

    @property
    def vertex_count(self) -> int:
        return len(self.positions)


@dataclass
class QuantizedMesh:
    """Result of quantization."""

    buffer: bytes
    stride: int
    offsets: dict  # kind -> byte offset within a vertex
    component_format: dict  # kind -> (components, bits)
    alignment: int
    indices: list
    vertex_count: int
    max_error: dict  # kind -> achieved max error in decoded units
    within_budget: dict  # kind -> bool
    sanitized: dict  # kind -> number of NaN/inf components replaced
    welded_away: int  # vertices removed by welding

    def vertex_bytes(self, i: int) -> bytes:
        start = i * self.stride
        return self.buffer[start : start + self.stride]

    def decode_attribute(self, i: int, kind: str, ranges) -> tuple:
        """Decode attribute `kind` of vertex `i` back to floats.

        `ranges` is the dict returned by MeshQuantizer.quantize (per-kind
        decode ranges); positions and UVs need it, normals/weights ignore it.
        """
        comps, bits = self.component_format[kind]
        base = i * self.stride + self.offsets[kind]
        nbytes = (bits + 7) // 8
        raw = []
        for c in range(comps):
            b = self.buffer[base + c * nbytes : base + (c + 1) * nbytes]
            raw.append(int.from_bytes(b, "little"))
        levels = (1 << bits) - 1
        if kind == POSITION:
            mn, mx = ranges[POSITION]
            return tuple(
                mn[a] + raw[a] / levels * (mx[a] - mn[a]) for a in range(3)
            )
        if kind == UV:
            island = ranges["island_of_vertex"][i]
            mn, mx = ranges[UV][island]
            return tuple(mn[a] + raw[a] / levels * (mx[a] - mn[a]) for a in range(2))
        if kind == NORMAL:
            enc = tuple(r / levels * 2.0 - 1.0 for r in raw)
            return _oct_decode(enc)
        if kind == WEIGHT:
            return tuple(r / levels for r in raw)
        raise KeyError(kind)


# ---------------------------------------------------------------------------
# sanitization
# ---------------------------------------------------------------------------


def _clean_float(value: float, default: float = 0.0) -> tuple:
    """Replace NaN/inf with a default. Returns (value, was_replaced)."""
    if isinstance(value, float) and not math.isfinite(value):
        return default, True
    return value, False


def _sanitize(mesh: Mesh, kinds: set) -> tuple:
    """Return a sanitized copy of the mesh plus per-kind replacement counts."""
    counts = {k: 0 for k in kinds}

    def fix(vec, kind, default):
        out = []
        for v in vec:
            nv, bad = _clean_float(v, default)
            if bad:
                counts[kind] += 1
            out.append(nv)
        return tuple(out)

    positions = [fix(p, POSITION, 0.0) for p in mesh.positions]
    normals = []
    for n in mesh.normals:
        had_nan = any(isinstance(v, float) and not math.isfinite(v) for v in n)
        fn = fix(n, NORMAL, 0.0)
        if had_nan and fn == (0.0, 0.0, 0.0):
            fn = (0.0, 0.0, 1.0)  # NaN normal -> up vector
        normals.append(fn)
    uvs = [fix(t, UV, 0.0) for t in mesh.uvs]
    weights = [fix(w, WEIGHT, 0.0) for w in mesh.weights]
    return (
        Mesh(positions, list(mesh.indices), normals, uvs, weights),
        counts,
    )


# ---------------------------------------------------------------------------
# octahedral normal encoding
# ---------------------------------------------------------------------------


def _sign(v: float) -> float:
    return 1.0 if v >= 0.0 else -1.0


def _oct_encode(n) -> tuple:
    x, y, z = n
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        x, y, z = 0.0, 0.0, 1.0
        length = 1.0
    x, y, z = x / length, y / length, z / length
    inv_l1 = 1.0 / (abs(x) + abs(y) + abs(z))
    x, y = x * inv_l1, y * inv_l1
    if z < 0.0:
        x, y = (1.0 - abs(y)) * _sign(x), (1.0 - abs(x)) * _sign(y)
    return x, y


def _oct_decode(enc) -> tuple:
    x, y = enc
    z = 1.0 - abs(x) - abs(y)
    if z < 0.0:
        x, y = (1.0 - abs(y)) * _sign(x), (1.0 - abs(x)) * _sign(y)
    length = math.sqrt(x * x + y * y + z * z)
    return x / length, y / length, z / length


# ---------------------------------------------------------------------------
# UV islands
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _uv_islands(mesh: Mesh) -> tuple:
    """Assign each vertex a UV-island id.

    UV vertices with identical coordinates are welded; triangles then union
    the welded UV vertices they touch. Mirrored islands (negative or flipped
    coordinates) are simply separate islands with their own ranges.
    """
    weld = {}
    uv_vid = []
    for t in mesh.uvs:
        key = (t[0], t[1])
        if key not in weld:
            weld[key] = len(weld)
        uv_vid.append(weld[key])

    uf = _UnionFind(len(weld))
    tris = mesh.indices
    for i in range(0, len(tris) - 2, 3):
        a, b, c = tris[i], tris[i + 1], tris[i + 2]
        uf.union(uv_vid[a], uv_vid[b])
        uf.union(uv_vid[b], uv_vid[c])

    island_of_uv = {}
    islands = []
    for root in (uf.find(v) for v in range(len(weld))):
        if root not in island_of_uv:
            island_of_uv[root] = len(islands)
            islands.append([float("inf"), float("inf"), float("-inf"), float("-inf")])
    for t, vid in zip(mesh.uvs, uv_vid):
        isl = islands[island_of_uv[uf.find(vid)]]
        isl[0] = min(isl[0], t[0])
        isl[1] = min(isl[1], t[1])
        isl[2] = max(isl[2], t[0])
        isl[3] = max(isl[3], t[1])
    island_ranges = [((a[0], a[1]), (a[2], a[3])) for a in islands]
    return [island_of_uv[uf.find(v)] for v in uv_vid], island_ranges


# ---------------------------------------------------------------------------
# scalar quantizers
# ---------------------------------------------------------------------------


def _quantize_scalar(value: float, mn: float, mx: float, levels: int) -> int:
    span = mx - mn
    if span <= 0.0:
        return 0  # degenerate (e.g. very narrow / flat bbox axis)
    t = (value - mn) / span
    t = min(1.0, max(0.0, t))
    return int(round(t * levels))


def _quantize_weights(w, levels: int) -> tuple:
    """Quantize weights so the quantized values still sum to `levels`.

    Uses largest-remainder rounding so the decoded weights renormalize to
    exactly 1.0 and the per-weight error stays within 1/levels.
    """
    total = sum(w)
    if total <= 0.0:
        return (0, 0, 0, 0)[: len(w)]  # zero weights stay zero
    norm = [max(0.0, x) / total for x in w]
    exact = [x * levels for x in norm]
    floor = [int(math.floor(x)) for x in exact]
    remainder = levels - sum(floor)
    order = sorted(range(len(w)), key=lambda i: exact[i] - floor[i], reverse=True)
    out = list(floor)
    for i in order[:remainder]:
        out[i] += 1
    return tuple(out)


# ---------------------------------------------------------------------------
# main quantizer
# ---------------------------------------------------------------------------


class MeshQuantizer:
    """Quantizes a Mesh according to a list of AttributeSpec."""

    def __init__(self, specs, alignment: int = 4, strict: bool = False):
        self.specs = {s.kind: s for s in specs}
        self.alignment = alignment
        self.strict = strict
        kinds = [s.kind for s in specs]
        if len(set(kinds)) != len(kinds):
            raise ValueError("duplicate attribute kinds in specs")
        if POSITION not in self.specs:
            raise ValueError("a position spec is required")

    # -- layout ----------------------------------------------------------

    def layout(self) -> tuple:
        """Return (offsets, stride) for the interleaved vertex layout."""
        offsets = {}
        cursor = 0
        for kind, spec in self.specs.items():
            nbytes = (spec.bits + 7) // 8
            size = nbytes * spec.components
            align = min(self.alignment, nbytes)
            if cursor % align:
                cursor += align - cursor % align
            offsets[kind] = cursor
            cursor += size
        if cursor % self.alignment:
            cursor += self.alignment - cursor % self.alignment
        return offsets, cursor

    # -- quantization ----------------------------------------------------

    def quantize(self, mesh: Mesh) -> tuple:
        """Quantize `mesh`. Returns (QuantizedMesh, decode_ranges)."""
        kinds = set(self.specs)
        mesh, sanitized = _sanitize(mesh, kinds)
        n = mesh.vertex_count
        levels = {k: s.levels for k, s in self.specs.items()}

        # ranges
        pos_min = tuple(min(p[a] for p in mesh.positions) for a in range(3))
        pos_max = tuple(max(p[a] for p in mesh.positions) for a in range(3))
        ranges = {POSITION: (pos_min, pos_max)}

        island_of_vertex = []
        if UV in kinds:
            island_of_vertex, island_ranges = _uv_islands(mesh)
            ranges[UV] = island_ranges

        # per-vertex quantized keys
        keys = []
        for i in range(n):
            key = []
            if POSITION in kinds:
                key += [
                    _quantize_scalar(mesh.positions[i][a], pos_min[a], pos_max[a],
                                     levels[POSITION])
                    for a in range(3)
                ]
            if NORMAL in kinds:
                enc = _oct_encode(mesh.normals[i])
                key += [
                    int(round((c * 0.5 + 0.5) * levels[NORMAL])) for c in enc
                ]
            if UV in kinds:
                mn, mx = ranges[UV][island_of_vertex[i]]
                key += [
                    _quantize_scalar(mesh.uvs[i][a], mn[a], mx[a], levels[UV])
                    for a in range(2)
                ]
            if WEIGHT in kinds:
                key += list(_quantize_weights(mesh.weights[i], levels[WEIGHT]))
            keys.append(tuple(key))

        # weld: identical quantized keys -> one vertex (stable, first wins)
        remap = {}
        unique_keys = []
        old_to_new = []
        first_old = []  # first source vertex producing each unique key
        for old_id, key in enumerate(keys):
            new_id = remap.get(key)
            if new_id is None:
                new_id = len(unique_keys)
                remap[key] = new_id
                unique_keys.append(key)
                first_old.append(old_id)
            old_to_new.append(new_id)

        if UV in kinds:
            # welded copies share a quantized UV key, hence the same island
            ranges["island_of_vertex"] = [island_of_vertex[i] for i in first_old]

        new_indices = [old_to_new[i] for i in mesh.indices]

        # pack interleaved buffer
        offsets, stride = self.layout()
        buffer = bytearray(stride * len(unique_keys))
        component_format = {}
        for kind, spec in self.specs.items():
            component_format[kind] = (spec.components, spec.bits)
        for vid, key in enumerate(unique_keys):
            base = vid * stride
            cursor = 0
            for kind, spec in self.specs.items():
                nbytes = (spec.bits + 7) // 8
                for c in range(spec.components):
                    value = key[cursor]
                    cursor += 1
                    start = base + offsets[kind] + c * nbytes
                    buffer[start : start + nbytes] = value.to_bytes(nbytes, "little")

        qmesh = QuantizedMesh(
            buffer=bytes(buffer),
            stride=stride,
            offsets=dict(offsets),
            component_format=component_format,
            alignment=self.alignment,
            indices=new_indices,
            vertex_count=len(unique_keys),
            max_error={},
            within_budget={},
            sanitized=sanitized,
            welded_away=n - len(unique_keys),
        )

        # error measurement against the sanitized source
        max_error = self._measure_errors(mesh, qmesh, ranges, old_to_new)
        qmesh.max_error = max_error
        for kind, spec in self.specs.items():
            ok = max_error[kind] <= spec.max_error
            qmesh.within_budget[kind] = ok
            if self.strict and not ok:
                raise ValueError(
                    f"{kind}: error {max_error[kind]:.6g} exceeds budget "
                    f"{spec.max_error:.6g}"
                )
        return qmesh, ranges

    def _measure_errors(self, mesh, qmesh, ranges, old_to_new) -> dict:
        errors = {k: 0.0 for k in self.specs}
        for i in range(mesh.vertex_count):
            vid = old_to_new[i]
            for kind in self.specs:
                dec = qmesh.decode_attribute(vid, kind, ranges)
                if kind == POSITION:
                    err = max(
                        abs(dec[a] - mesh.positions[i][a]) for a in range(3)
                    )
                elif kind == NORMAL:
                    err = math.sqrt(
                        sum((dec[a] - mesh.normals[i][a]) ** 2 for a in range(3))
                    )
                elif kind == UV:
                    err = max(abs(dec[a] - mesh.uvs[i][a]) for a in range(2))
                else:  # WEIGHT
                    err = max(
                        abs(dec[a] - mesh.weights[i][a]) for a in range(4)
                    )
                errors[kind] = max(errors[kind], err)
        return errors


def pack_indices(indices, bits: int = 32) -> bytes:
    """Pack an index buffer as little-endian u16/u32."""
    fmt = "<H" if bits == 16 else "<I"
    return b"".join(struct.pack(fmt, i) for i in indices)
