# Code style: AI-code. AI-generated regression test.
"""Physical geometry and selection checks for offline event-mixing pairs."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from shift_mixing import (
    MUON_MASS_GEV, common_line_vertex, pair_observables, resolved_config,
    select_pairs, straight_line_approach,
)


def muon(point=(0.0, 0.0, 14800.0), momentum=(1.0, 0.0, -100.0), charge=1, topology=2):
    return dict(zip(("vx", "vy", "vz", "px", "py", "pz"), (*point, *momentum)),
                charge=charge, topology=topology)


class MixingFitTest(unittest.TestCase):
    def test_float_semantics_against_standalone_cmssw_header_reference(self):
        # Captured by compiling the producer's two analytic functions against
        # CMSSW_17_0_0_pre4 GlobalPoint/GlobalVector headers (GNU ExtVec)
        # and production -O3 -march=x86-64-v3 compiler flags.
        first = muon((20.25, -30.125, 15500.5), (1.23456, -2.34567, -543.21))
        second = muon((-15.375, 17.25, 14900.25), (-3.45678, 4.56789, -456.78), charge=-1)
        pair = pair_observables(first, second)
        expected = {
            "vx": 7.7760062217712402, "vy": -11.268238067626953, "vz": 18127.6875,
            "chi2": 0.030189286132018975, "dca": 2.1830811500549316,
            "dcaX": 12.586523056030273, "dcaY": -17.892391204833984,
            "dcaZ": 18476.046875, "originCompatibilityChi2": 0.22071407031250001,
            "mass": 8.6569338679185872,
        }
        for name, value in expected.items():
            # Residual cross-product contraction changes chi2 at sub-1e-8;
            # positions, DCA and mass have the stricter reference bound.
            tolerance = 1.0e-8 if name == "chi2" else max(abs(value) * 1.0e-11, 1.0e-11)
            self.assertAlmostEqual(pair[name], value, delta=tolerance, msg=name)

    def test_known_intersecting_lines_preserve_origin_and_mass(self):
        # Two tracks through a remote beam-line vertex. The fit must not have
        # the detector-volume bounds of a TransientTrack extrapolation.
        first = muon(momentum=(1.0, 0.0, -100.0))
        second = muon(momentum=(-1.0, 0.0, -100.0), charge=-1)
        pair = pair_observables(first, second)
        self.assertIsNotNone(pair)
        self.assertAlmostEqual(pair["vx"], 0.0, places=9)
        self.assertAlmostEqual(pair["vy"], 0.0, places=9)
        self.assertAlmostEqual(pair["vz"], 14800.0, places=6)
        self.assertAlmostEqual(pair["mass"], 2.0 * math.hypot(1.0, MUON_MASS_GEV), places=9)
        self.assertAlmostEqual(pair["pz"], -200.0)
        self.assertAlmostEqual(pair["openingAngle"], 2.0 * math.atan(0.01), places=12)
        self.assertAlmostEqual(pair["chi2"], 0.0, places=20)
        self.assertAlmostEqual(pair["probability"], 1.0)
        self.assertNotIn("refittedMass", pair)
        self.assertNotIn("constrainedMass", pair)

    def test_skew_geometry_and_parallel_branch_match_producer_convention(self):
        approach = straight_line_approach(np.array((1.0, 0.0, 10.0)), np.array((0.0, 0.0, 1.0)),
                                          np.array((0.0, 2.0, 20.0)), np.array((1.0, 0.0, 0.0)))
        self.assertAlmostEqual(approach["distance"], 2.0)
        np.testing.assert_allclose(approach["midpoint"], (1.0, 1.0, 20.0))
        parallel = straight_line_approach(np.array((0.0, 0.0, 10.0)), np.array((0.0, 0.0, 1.0)),
                                          np.array((3.0, 4.0, 22.0)), np.array((0.0, 0.0, -1.0)))
        self.assertAlmostEqual(parallel["distance"], 13.0)
        np.testing.assert_allclose(parallel["midpoint"], (1.5, 2.0, 16.0))

    def test_common_fit_minimizes_track_distances_and_beam_prior(self):
        p1, p2 = np.array((2.0, 3.0, 12000.0)), np.array((-4.0, 5.0, 13000.0))
        u1, u2 = np.array((0.02, 0.01, -1.0)), np.array((-0.03, 0.01, -1.0))
        u1 /= np.linalg.norm(u1)
        u2 /= np.linalg.norm(u2)
        fit = common_line_vertex(p1, u1, p2, u2)
        self.assertIsNotNone(fit)

        def objective(vertex):
            return (np.linalg.norm(np.cross(vertex - p1, u1)) ** 2 +
                    np.linalg.norm(np.cross(vertex - p2, u2)) ** 2 +
                    vertex[0] ** 2 + vertex[1] ** 2) / 100.0 ** 2

        self.assertAlmostEqual(objective(fit["position"]), fit["chi2"], places=12)
        for direction in np.eye(3):
            for sign in (-1, 1):
                self.assertGreater(objective(fit["position"] + sign * direction), fit["chi2"])
        self.assertTrue(np.all(np.linalg.eigvalsh(fit["covariance"]) > 0.0))
        # Exact z-parallel lines leave the origin z unconstrained.
        self.assertIsNone(common_line_vertex(p1, (0.0, 0.0, 1.0), p2, (0.0, 0.0, 1.0)))

    def test_same_side_charge_dca_and_origin_cuts_have_separate_effects(self):
        first = muon()
        second = muon(charge=-1)
        self.assertIsNotNone(pair_observables(first, second))
        self.assertIsNone(pair_observables(first, muon(charge=1)))
        self.assertIsNotNone(pair_observables(first, muon(charge=1), {"requireOppositeSign": False}))
        for z in (0.0, -14800.0):
            self.assertIsNone(pair_observables(first, muon(point=(0.0, 0.0, z), charge=-1)))
        # Parallel branch means this separation also directly tests DCA.
        displaced = muon(point=(0.0, 10.0, 14800.0), charge=-1)
        self.assertIsNone(pair_observables(first, displaced, {"maxPairDca": 9.99}))
        self.assertIsNotNone(pair_observables(first, displaced, {"maxPairDca": 10.0}))
        remote = muon(point=(0.0, 0.0, 30000.0), charge=-1)
        self.assertIsNone(pair_observables(first, remote, {"maxPairDca": 50000.0}))
        self.assertIsNotNone(pair_observables(first, remote, {
            "maxPairDca": 50000.0, "maxPairOriginNormalizedChi2": 100.0}))

    def test_stable_ranking_greedy_non_reuse_and_vertex_cap(self):
        muons = [muon(charge=charge) for charge in (1, -1, 1, -1)]
        pair = select_pairs(muons)
        self.assertEqual([(p["muonIdx1"], p["muonIdx2"]) for p in pair], [(0, 1)])
        unlimited = select_pairs(muons, {"maxDimuonVertices": 0})
        self.assertEqual([(p["muonIdx1"], p["muonIdx2"]) for p in unlimited], [(0, 1), (2, 3)])
        # Lower score, not the first loop combination, wins.
        muons[1]["vy"] = 10.0
        selected = select_pairs(muons)
        self.assertEqual((selected[0]["muonIdx1"], selected[0]["muonIdx2"]), (0, 3))
        ss = select_pairs([muon(charge=1), muon(charge=1)], require_os=False)
        self.assertEqual(ss[0]["isOS"], 0)

    def test_charge_momentum_orientation_is_not_reapplied(self):
        first = muon(momentum=(3.0, 0.0, -100.0))
        second = muon(momentum=(0.0, 4.0, 100.0), charge=-1)
        pair = pair_observables(first, second)
        self.assertGreater(pair["mass"], 200.0)
        self.assertAlmostEqual(pair["pt"], 5.0)
        self.assertAlmostEqual(pair["pz"], 0.0)
        self.assertAlmostEqual(pair["deltaPhi"], math.pi / 2)
        self.assertGreater(pair["openingAngle"], 3.0)

    def test_float_producer_mass_and_double_diagnostic_remain_distinct(self):
        # The source computes GlobalVector::mag2 in float before energy.
        # At large boost, adding 100 to 1e8 rounds to 96, changing pair mass.
        pair = pair_observables(muon(momentum=(10.0, 0.0, -10000.0)),
                                muon(momentum=(-10.0, 0.0, -10000.0), charge=-1))
        self.assertIsNotNone(pair)
        self.assertAlmostEqual(pair["mass"], 2.0 * math.sqrt(96.0 + MUON_MASS_GEV ** 2), places=7)
        self.assertAlmostEqual(pair["massDouble"], 2.0 * math.hypot(10.0, MUON_MASS_GEV), places=7)
        self.assertGreater(pair["massDouble"] - pair["mass"], 0.4)

    def test_scram_fma_accumulation_closes_boosted_pairs(self):
        # SCRAM -O3 -march=x86-64-v3 contracts ExtVec dot accumulation to
        # FMA. Separate float multiply/add changed these masses by 12 MeV
        # and 2.5 MeV respectively; the bound below is output-float precision.
        cases = (
            ((49.14045715332031, 27.174890518188477, 13640.34765625),
             (4.115904808044434, -7.443426609039307, -768.699951171875),
             (-35.83309555053711, 20.36592674255371, 7226.10498046875),
             (0.11989067494869232, 0.21092839539051056, -18.92064666748047),
             2.603573799133301),
            ((-76.12130737304688, 5.17785120010376, 5992.9052734375),
             (0.17090930044651031, 2.512584924697876, -101.57963562011719),
             (-30.457443237304688, 2.3865456581115723, 9993.4775390625),
             (0.06358278542757034, 0.8107864856719971, -31.226503372192383),
             0.2592242956161499),
        )
        for point1, momentum1, point2, momentum2, expected in cases:
            pair = pair_observables(muon(point1, momentum1), muon(point2, momentum2, charge=-1))
            self.assertIsNotNone(pair)
            self.assertAlmostEqual(pair["mass"], expected, delta=2.0e-7)

    def test_standalone_dca_simd_reductions_and_fused_closest_points(self):
        # The separately compiled DCA function retains SIMD products before
        # scalar dot reduction, whereas the inlined fit/mass uses scalar FMA.
        # Using the fit's direction arithmetic here changed DCA by up to 1cm.
        cases = (
            ((-34.45600891113281, 127.4537353515625, 13577.609375),
             (0.26782092452049255, 0.07239875942468643, -18.343151092529297),
             (57.229408264160156, -35.85237121582031, 17768.412109375),
             (0.17448008060455322, 0.2784959077835083, -24.864940643310547),
             0.40555328130722046),
            ((21.718769073486328, 6.536252021789551, 10550.1611328125),
             (-0.19777342677116394, 0.6569605469703674, -20.722444534301758),
             (15.557518005371094, 4.389936923980713, 10399.9404296875),
             (-0.24445757269859314, 0.8659347891807556, -26.71560287475586),
             1.0358033180236816),
            ((135.29798889160156, -83.91673278808594, 11076.5966796875),
             (0.5136422514915466, 0.8281624913215637, -43.55664825439453),
             (1.0094547271728516, -2.4944019317626953, 15143.7890625),
             (1.7290022373199463, 0.6982266902923584, -95.53089904785156),
             0.6997718811035156),
            ((-76.12130737304688, 5.17785120010376, 5992.9052734375),
             (0.17090930044651031, 2.512584924697876, -101.57963562011719),
             (-30.457443237304688, 2.3865456581115723, 9993.4775390625),
             (0.06358278542757034, 0.8107864856719971, -31.226503372192383),
             25.625959396362305),
        )
        for point1, momentum1, point2, momentum2, expected in cases:
            pair = pair_observables(muon(point1, momentum1), muon(point2, momentum2, charge=-1))
            self.assertIsNotNone(pair)
            self.assertEqual(pair["dca"], expected)

    def test_phi_wrapping_probability_and_topology_are_finite(self):
        first = muon(momentum=(-1.0, 0.01, -100.0), topology=3)
        second = muon(momentum=(-1.0, -0.01, -100.0), charge=-1, topology=0)
        pair = pair_observables(first, second)
        self.assertAlmostEqual(pair["deltaPhi"], 2.0 * math.atan(0.01), places=9)
        self.assertEqual((pair["topologyMin"], pair["topologyMax"]), (0, 3))
        self.assertGreaterEqual(pair["probability"], 0.0)
        self.assertLessEqual(pair["probability"], 1.0)

    def test_bad_schema_and_unavailable_hypotheses_fail_closed(self):
        for name in ("vx", "px", "charge", "topology"):
            first = muon()
            first.pop(name)
            with self.assertRaisesRegex(ValueError, "Missing persisted"):
                pair_observables(first, muon(charge=-1))
        for value in (math.nan, math.inf):
            first = muon()
            first["px"] = value
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                pair_observables(first, muon(charge=-1))
        with self.assertRaisesRegex(ValueError, "Zero"):
            pair_observables(muon(momentum=(0.0, 0.0, 0.0)), muon(charge=-1))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            resolved_config({"usePairVertexPropagation": True})
        with self.assertRaisesRegex(ValueError, "positive"):
            resolved_config({"commonVertexLineResolution": 0.0})


if __name__ == "__main__":
    unittest.main()
