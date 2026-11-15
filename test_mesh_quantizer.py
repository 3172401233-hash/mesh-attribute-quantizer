import math
import random
import unittest

from mesh_quantizer import (
    NORMAL,
    POSITION,
    UV,
    WEIGHT,
    AttributeSpec,
    Mesh,
    MeshQuantizer,
)


def make_specs(pos_bits=16, nrm_bits=10, uv_bits=12, w_bits=8,
               pos_err=1e-3, nrm_err=5e-2, uv_err=1e-3, w_err=1e-2):
    return [
        AttributeSpec(POSITION, pos_bits, pos_err),
        AttributeSpec(NORMAL, nrm_bits, nrm_err),
        AttributeSpec(UV, uv_bits, uv_err),
        AttributeSpec(WEIGHT, w_bits, w_err),
    ]


def quad_mesh():
    """Two triangles sharing an edge, with duplicated corner vertices."""
    positions = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),  # tri 0
        (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),  # tri 1
    ]
    normals = [(0.0, 0.0, 1.0)] * 6
    uvs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
           (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    weights = [(1.0, 0.0, 0.0, 0.0)] * 6
    indices = [0, 1, 2, 3, 4, 5]
    return Mesh(positions, indices, normals, uvs, weights)


class TestSharedEdgeWelding(unittest.TestCase):
    """Same original vertex must map to the same quantized vertex."""

    def test_shared_edge_welded(self):
        q = MeshQuantizer(make_specs())
        qm, ranges = q.quantize(quad_mesh())
        # 6 input corners -> 4 unique quantized vertices
        self.assertEqual(qm.vertex_count, 4)
        self.assertEqual(qm.welded_away, 2)

    def test_shared_edge_no_crack(self):
        q = MeshQuantizer(make_specs())
        qm, ranges = q.quantize(quad_mesh())
        # the shared edge is (1,0,0)-(0,1,0); both triangles must decode
        # to bit-identical positions for those corners
        tri0 = qm.indices[0:3]
        tri1 = qm.indices[3:6]
        p = lambda v: qm.decode_attribute(v, POSITION, ranges)
        self.assertEqual(p(tri0[1]), p(tri1[0]))  # (1,0,0)
        self.assertEqual(p(tri0[2]), p(tri1[2]))  # (0,1,0)

    def test_welding_is_stable(self):
        # vertex order in the output is deterministic: first occurrence wins
        q = MeshQuantizer(make_specs())
        qm1, _ = q.quantize(quad_mesh())
        qm2, _ = q.quantize(quad_mesh())
        self.assertEqual(qm1.buffer, qm2.buffer)
        self.assertEqual(qm1.indices, qm2.indices)


class TestNarrowBBox(unittest.TestCase):
    def test_very_narrow_bbox(self):
        span = 1e-4
        positions = [
            (0.0, 0.0, 0.0),
            (span, 0.0, 0.0),
            (0.0, span, 0.0),
            (span, span, 0.0),  # flat z: degenerate axis
        ]
        mesh = Mesh(positions, [0, 1, 2, 1, 3, 2],
                    normals=[(0, 0, 1)] * 4,
                    uvs=[(0, 0), (1, 0), (0, 1), (1, 1)],
                    weights=[(1, 0, 0, 0)] * 4)
        # theoretical bound: half a quantization step = span/2 / (2^bits-1)
        bound = span / 2 / (2**16 - 1)
        q = MeshQuantizer(make_specs(pos_err=bound * 1.0001))
        qm, ranges = q.quantize(mesh)
        self.assertLessEqual(qm.max_error[POSITION], bound * 1.0001)
        self.assertTrue(qm.within_budget[POSITION])
        # decode corners
        for i, src in enumerate(positions):
            dec = qm.decode_attribute(i, POSITION, ranges)
            for a in range(3):
                self.assertAlmostEqual(dec[a], src[a], delta=bound * 1.0001)

    def test_single_point_mesh(self):
        mesh = Mesh([(3.0, 4.0, 5.0)], [0, 0, 0],
                    normals=[(0, 0, 1)], uvs=[(0.5, 0.5)],
                    weights=[(1, 0, 0, 0)])
        q = MeshQuantizer(make_specs())
        qm, ranges = q.quantize(mesh)
        dec = qm.decode_attribute(0, POSITION, ranges)
        self.assertEqual(dec, (3.0, 4.0, 5.0))


class TestMirroredUV(unittest.TestCase):
    def test_mirrored_islands(self):
        # island A in [0,1]^2, island B mirrored into negative u
        positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                     (2, 0, 0), (3, 0, 0), (2, 1, 0)]
        uvs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
               (-2.0, 0.0), (-3.0, 0.0), (-2.0, 1.0)]  # mirrored, disjoint
        mesh = Mesh(positions, [0, 1, 2, 3, 4, 5],
                    normals=[(0, 0, 1)] * 6,
                    uvs=uvs,
                    weights=[(1, 0, 0, 0)] * 6)
        q = MeshQuantizer(make_specs(uv_err=1e-3))
        qm, ranges = q.quantize(mesh)
        self.assertEqual(len(ranges[UV]), 2)  # two separate islands
        for i, src in enumerate(uvs):
            dec = qm.decode_attribute(i, UV, ranges)
            self.assertAlmostEqual(dec[0], src[0], delta=1e-3)
            self.assertAlmostEqual(dec[1], src[1], delta=1e-3)
        self.assertTrue(qm.within_budget[UV])


class TestWeights(unittest.TestCase):
    def test_zero_weights_stay_zero(self):
        mesh = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 2],
                    normals=[(0, 0, 1)] * 3,
                    uvs=[(0, 0), (1, 0), (0, 1)],
                    weights=[(0.0, 0.0, 0.0, 0.0)] * 3)
        q = MeshQuantizer(make_specs())
        qm, ranges = q.quantize(mesh)
        for v in range(qm.vertex_count):
            self.assertEqual(qm.decode_attribute(v, WEIGHT, ranges),
                             (0.0, 0.0, 0.0, 0.0))

    def test_weights_renormalize_to_one(self):
        weights = [(0.25, 0.25, 0.25, 0.25),
                   (0.7, 0.2, 0.07, 0.03),
                   (0.333, 0.333, 0.334, 0.0)]
        mesh = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 2],
                    normals=[(0, 0, 1)] * 3,
                    uvs=[(0, 0), (1, 0), (0, 1)],
                    weights=weights)
        q = MeshQuantizer(make_specs(w_err=1.0 / 255))
        qm, ranges = q.quantize(mesh)
        for i, src in enumerate(weights):
            dec = qm.decode_attribute(i, WEIGHT, ranges)
            self.assertAlmostEqual(sum(dec), 1.0, places=12)
            for a in range(4):
                self.assertAlmostEqual(dec[a], src[a], delta=1.0 / 255 + 1e-12)
        self.assertTrue(qm.within_budget[WEIGHT])


class TestNaN(unittest.TestCase):
    def test_nan_sanitized(self):
        nan = float("nan")
        inf = float("inf")
        mesh = Mesh(
            [(0, 0, 0), (nan, 0, 0), (0, 1, inf)],
            [0, 1, 2],
            normals=[(0, 0, 1), (nan, nan, nan), (0, 0, 1)],
            uvs=[(0, 0), (nan, 0), (0, 1)],
            weights=[(1, 0, 0, 0), (nan, 0, 0, 0), (1, 0, 0, 0)],
        )
        q = MeshQuantizer(make_specs())
        qm, ranges = q.quantize(mesh)
        self.assertGreater(qm.sanitized[POSITION], 0)
        self.assertGreater(qm.sanitized[NORMAL], 0)
        self.assertGreater(qm.sanitized[UV], 0)
        self.assertGreater(qm.sanitized[WEIGHT], 0)
        # nothing non-finite survives decoding
        for v in range(qm.vertex_count):
            for kind in (POSITION, NORMAL, UV, WEIGHT):
                for x in qm.decode_attribute(v, kind, ranges):
                    self.assertTrue(math.isfinite(x))
        # NaN normal falls back to a unit vector
        dec = qm.decode_attribute(1, NORMAL, ranges)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in dec)), 1.0)


class TestNormals(unittest.TestCase):
    def test_octahedral_roundtrip(self):
        rng = random.Random(7)
        normals = []
        for _ in range(64):
            v = tuple(rng.gauss(0, 1) for _ in range(3))
            length = math.sqrt(sum(c * c for c in v))
            normals.append(tuple(c / length for c in v))
        n = len(normals)
        mesh = Mesh([(float(i), 0.0, 0.0) for i in range(n)],
                    [i for i in range(n - 2) for i in (i, i + 1, i + 2)]
                    if n > 2 else [0, 0, 0],
                    normals=normals,
                    uvs=[(0.0, 0.0)] * n,
                    weights=[(1, 0, 0, 0)] * n)
        budget = 2.0 / (2**10 - 1) * 2  # ~2 oct-cell diagonals at 10 bits
        q = MeshQuantizer(make_specs(nrm_err=budget))
        qm, ranges = q.quantize(mesh)
        self.assertTrue(qm.within_budget[NORMAL])
        for i, src in enumerate(normals):
            dec = qm.decode_attribute(i, NORMAL, ranges)
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in dec)), 1.0,
                                   places=6)
            err = math.sqrt(sum((dec[a] - src[a]) ** 2 for a in range(3)))
            self.assertLessEqual(err, budget)


class TestLayout(unittest.TestCase):
    def test_interleaved_layout(self):
        q = MeshQuantizer(make_specs(), alignment=4)
        offsets, stride = q.layout()
        # position: 3 x u16 = 6 bytes at 0; normal: 2 x u16 = 4 bytes at 6
        # uv: 2 x u16 = 4 bytes at 10; weight: 4 x u8 = 4 bytes at 14
        # stride padded to 4-byte alignment -> 20
        self.assertEqual(offsets[POSITION], 0)
        self.assertEqual(offsets[NORMAL], 6)
        self.assertEqual(offsets[UV], 10)
        self.assertEqual(offsets[WEIGHT], 14)
        self.assertEqual(stride % 4, 0)
        self.assertEqual(stride, 20)

    def test_buffer_size_and_alignment(self):
        q = MeshQuantizer(make_specs(), alignment=8)
        qm, _ = q.quantize(quad_mesh())
        self.assertEqual(len(qm.buffer), qm.stride * qm.vertex_count)
        self.assertEqual(qm.stride % 8, 0)
        self.assertEqual(qm.alignment, 8)
        for kind, off in qm.offsets.items():
            nbytes = (qm.component_format[kind][1] + 7) // 8
            self.assertEqual(off % min(8, nbytes), 0)


class TestErrorBudget(unittest.TestCase):
    def test_roundtrip_within_budget(self):
        rng = random.Random(42)
        n = 50
        positions = [(rng.uniform(-2, 2), rng.uniform(-1, 3), rng.uniform(0, 1))
                     for _ in range(n)]
        normals = []
        for _ in range(n):
            v = tuple(rng.gauss(0, 1) for _ in range(3))
            length = math.sqrt(sum(c * c for c in v))
            normals.append(tuple(c / length for c in v))
        uvs = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(n)]
        weights = []
        for _ in range(n):
            w = [rng.random() for _ in range(4)]
            s = sum(w)
            weights.append(tuple(x / s for x in w))
        indices = []
        for i in range(0, n - 2, 3):
            indices += [i, i + 1, i + 2]
        mesh = Mesh(positions, indices, normals, uvs, weights)
        specs = make_specs(pos_err=1e-3, nrm_err=5e-2, uv_err=1e-3, w_err=1e-2)
        q = MeshQuantizer(specs, strict=True)  # raises if budget exceeded
        qm, _ = q.quantize(mesh)
        for kind in (POSITION, NORMAL, UV, WEIGHT):
            self.assertTrue(qm.within_budget[kind])
            self.assertLessEqual(qm.max_error[kind],
                                 {POSITION: 1e-3, NORMAL: 5e-2,
                                  UV: 1e-3, WEIGHT: 1e-2}[kind])

    def test_strict_mode_raises(self):
        # 0.5 is off-grid at 4 bits (levels=15): error ~1/30 >> 1e-9
        mesh = Mesh([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)],
                    [0, 1, 2],
                    normals=[(0, 0, 1)] * 3,
                    uvs=[(0, 0), (1, 0), (0, 1)],
                    weights=[(1, 0, 0, 0)] * 3)
        q = MeshQuantizer(make_specs(pos_bits=4, pos_err=1e-9), strict=True)
        with self.assertRaises(ValueError):
            q.quantize(mesh)


if __name__ == "__main__":
    unittest.main()
