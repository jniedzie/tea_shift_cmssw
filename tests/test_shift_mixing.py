# Code style: AI-code. AI-generated regression test.
"""Independent controls for event counting, weights, folds and reused legs."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from shift_mixing import (activity_bin, coverage, event_bootstrap, event_replacement,
                          feature_table, label_pair, like_sign_transfer, pools_for, row_augmentation,
                          same_event_closure)
from shift_mixing import common_parent, validate_inventory, like_sign_closure_bootstrap
from shift_mixing import event_partition
from shift_mixing import DEFAULT_CONFIG, select_pairs
from shift_mixing import MASS_EDGES, run_event_mixing, write_mixing_histograms


def muon(index, charge, px):
    values = dict(index=index, charge=charge, topology=2, vx=0., vy=0., vz=14800.,
                  px=px, py=0., pz=-100., pt=abs(px), phi=0. if px > 0 else np.pi,
                  eta=float(np.arcsinh(-100 / abs(px))), normalizedChi2=1.,
                  nValidHits=24, nValidMuonHits=24, nMuonStations=4, nLostHits=0,
                  recoAlgorithm=1, ptErr=.1, etaErr=.01, phiErr=.01,
                  hitTruthPurity=1., hitTruthMatchedLayers=6, hitSimTrackId=index)
    truth = dict(index=index, pdgId=-13 * charge, status=1, vx=0., vy=0., vz=14800.)
    values["truth"] = {"hit_match": [truth]}
    return values


def event(index, charge=None, partition="train", weight=10.):
    muons = [] if charge is None else [muon(0, charge, 1. if charge > 0 else -1.)]
    return dict(event_key=f"campaign:qcd_0to1:1:1:{index}", event_id=[1, 1, index],
                stratum="qcd_0to1", partition=partition, activity={"nCSCSegments": 3},
                detector_weight=weight, physical_weight_pb=2. * weight,
                muons=muons, nShiftMuon=len(muons), stored_pairs=[])


class MixingTest(unittest.TestCase):
    def test_main_stage_rejects_missing_or_ambiguous_inputs_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with self.assertRaisesRegex(ValueError, "pool and inventory"):
                run_event_mixing(dict(output=output))
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "config_json or fit"):
                run_event_mixing(dict(output=output, config_json="unused", fit={}))
            self.assertFalse(output.exists())

    def test_root_publication_keeps_model_weights_raw_trial_counts_and_source_errors(self):
        try:
            import ROOT
        except ImportError:
            self.skipTest("ROOT runtime unavailable")
        events = [event(0, 1), event(1, -1)]
        row = dict(pair={"mass": 3.1}, weight=6., both_endcaps=True,
                   source_events=[e["event_key"] for e in events],
                   anchor_event=0, donor_event=1, stratum="qcd_0to1",
                   partition="train", activity_bin="4")
        mixed = [dict(row, kind="event_os")] * 2
        with tempfile.TemporaryDirectory() as directory:
            diagnostics, artifacts = write_mixing_histograms(
                directory, [dict(row, kind="same_os")], mixed,
                [dict(row, kind="row_os")], events, 20, np.random.default_rng(3))
            self.assertEqual(len(artifacts), 5)
            model = next(item for item in artifacts if item["name"] == "event_replacement")
            expected = event_bootstrap(mixed, 2, 20, np.random.default_rng(3), events=events)
            with ROOT.TFile.Open(model["histogram_path"], "READ") as file:
                hist = file.Get("mixing/ShiftDimuonVertex_mass_model")
                self.assertEqual(hist.Integral(), 12.)
                self.assertEqual(hist.GetEntries(), 2.)
                self.assertAlmostEqual(hist.GetBinError(hist.FindBin(3.1)), expected.max())
                self.assertEqual(hist.GetNbinsX(), len(MASS_EDGES) - 1)
            trials = next(item for item in artifacts if item["name"] == "row_os")
            with ROOT.TFile.Open(trials["histogram_path"], "READ") as file:
                hist = file.Get("mixing/ShiftDimuonVertexBoth-Both_mass_trials")
                self.assertEqual(hist.Integral(), 1.)
                self.assertEqual(hist.GetBinError(hist.FindBin(3.1)), 0.)
            self.assertEqual(diagnostics["inclusive"]["event_replacement"][1]["distinct_source_events"], 2)

    def test_partition_and_occupancy_are_data_only_and_duplicate_keys_fail(self):
        a, b = event(0, 1), event(1, -1, partition="test")
        a["muons"] *= 20
        self.assertEqual(activity_bin(a), activity_bin(b))
        self.assertEqual(len(pools_for([a, b])), 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pools_for([a, dict(a)])

    def test_empty_events_remain_in_donor_event_accounting(self):
        events = [event(0), event(1)]
        rows, counts = event_replacement(events, DEFAULT_CONFIG, 7, np.random.default_rng(1))
        self.assertEqual(rows, [])
        self.assertEqual(counts["mock_events"], 14)
        self.assertEqual(counts["mock_events_with_no_os_candidate"], 14)

    def test_campaigns_never_share_a_pool(self):
        a, b = event(0, 1), event(1, -1)
        b["event_key"] = b["event_key"].replace("campaign:", "other_campaign:")
        self.assertEqual(len(pools_for([a, b])), 2)

    def test_extraction_inventory_rejects_changed_bytes_and_fold(self):
        source = event(0, 1)
        source["partition"] = event_partition(source["event_key"])
        receipt = dict(schema="shift-event-mixing-inventory-v1", campaign="campaign",
                       statistics={"qcd_0to1": {"events": 1, "muons": 1}},
                       source_input_complete=False, missing_jobs=[], sampling_plan_sha256="plan",
                       sampling_ledger_sha256="ledger", full_gen_sumw={}, cross_sections_pb={})
        with tempfile.TemporaryDirectory() as directory:
            pool, inventory = Path(directory) / "pool", Path(directory) / "inventory.json"
            pool.write_bytes(b"original")
            receipt["pool_sha256"] = hashlib.sha256(pool.read_bytes()).hexdigest()
            inventory.write_text(json.dumps(receipt))
            self.assertFalse(validate_inventory(pool, inventory, [source])["source_input_complete"])
            source["partition"] = "test" if source["partition"] != "test" else "train"
            with self.assertRaisesRegex(ValueError, "partition"):
                validate_inventory(pool, inventory, [source])
            pool.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                validate_inventory(pool, inventory, [source])

    def test_one_parent_cross_section_and_partner_average_not_squared_rate(self):
        events = [event(0, 1, weight=3.), event(1, -1, weight=7.)]
        rows, counts = event_replacement(events, DEFAULT_CONFIG, 5, np.random.default_rng(2))
        self.assertEqual(counts["mock_events"], 10)
        self.assertEqual(len(rows), 5)
        # With one possible donor W_j cancels its donor mean; retain anchor pb.
        self.assertAlmostEqual(sum(row["weight"] for row in rows), 6.)
        self.assertTrue(all(row["anchor_event"] != row["donor_event"] for row in rows))

    def test_row_augmentation_never_crosses_source_event_or_fold(self):
        events = [event(0, 1), event(1, -1), event(2, 1, partition="test")]
        rows, counts = row_augmentation(events, DEFAULT_CONFIG, 4, np.random.default_rng(3))
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row["partition"], "train")
            self.assertNotEqual(*row["source_events"])
            self.assertEqual(row["kind"], "row_os")

    def test_cross_event_coincident_prompt_tracks_are_not_labelled_negative(self):
        a, b = muon(0, 1, 1.), muon(1, -1, -1.)
        self.assertEqual(label_pair(a, b, True), (1, "common_vertex"))
        self.assertEqual(label_pair(a, b, False), (-1, "mixed_coincident_vertices"))
        b["truth"]["hit_match"][0]["vz"] += 10.
        self.assertEqual(label_pair(a, b, False), (0, "different_vertices"))

    def test_source_event_bootstrap_couples_all_pairs_reusing_a_leg(self):
        a = dict(anchor_event=0, donor_event=1, pair={"mass": 3.1}, weight=1.)
        # Duplicating a pair doubles its error, rather than increasing it sqrt(2).
        error = event_bootstrap([a], 2, 300, np.random.default_rng(4))
        doubled = event_bootstrap([a, a], 2, 300, np.random.default_rng(4))
        np.testing.assert_allclose(doubled, 2 * error)

    def test_mass_coverage_counts_original_events_not_trials(self):
        row = dict(pair={"mass": 90.}, source_events=["a", "b"], weight=1.)
        band = coverage([row] * 100)[-1]
        self.assertEqual(band["pairs"], 100)
        self.assertEqual(band["distinct_source_events"], 2)
        self.assertEqual(band["largest_event_fraction"], 1.)

    def test_one_true_muon_twice_is_not_a_resonance_pair(self):
        a, b = muon(0, 1, 1.), muon(1, -1, -1.)
        parent = dict(index=9, pdgId=223)
        a["truth"]["hit_match"].append(parent)
        b["truth"]["hit_match"].append(parent)
        self.assertEqual(common_parent(a, b), 223)
        b["truth"]["hit_match"][0]["index"] = 0
        self.assertIsNone(common_parent(a, b))

    def test_empty_reference_does_not_establish_fit_closure(self):
        self.assertFalse(same_event_closure([event(0)], DEFAULT_CONFIG)["passed"])

    def test_event_bootstrap_resamples_weighted_donor_normalization(self):
        events = [event(0, 1, weight=3.), event(1, -1, weight=7.)]
        row = dict(anchor_event=0, donor_event=1, pair={"mass": 3.1}, weight=6.)
        # With a single eligible donor, donor multiplicity cancels its weight
        # normalization whenever it is represented in the bootstrap sample.
        class FixedCounts:
            def __init__(self):
                self.counts = iter((np.array([1, 1]), np.array([1, 3])))
            def poisson(self, mean, count):
                return next(self.counts)
        result = event_bootstrap([row], 2, 2, FixedCounts(), events=events)
        np.testing.assert_array_equal(result, np.zeros(len(result)))

    def test_same_event_membership_mismatch_blocks_closure(self):
        e = event(0)
        e["muons"] = [muon(0, 1, 1.), muon(1, -1, -1.)]
        report = same_event_closure([e], DEFAULT_CONFIG)
        self.assertFalse(report["passed"])
        self.assertEqual(report["membership_mismatches"], 1)

    def test_feature_export_uses_only_existing_data_contract(self):
        a, b = muon(0, 1, 1.), muon(1, -1, -1.)
        pair = select_pairs([a, b], DEFAULT_CONFIG)[0]
        rows = [dict(pair=pair, muons=[a, b])]
        initial = feature_table(rows)
        a["truth"] = {"arbitrary": 999}
        b["truth"] = {"arbitrary": -100}
        a["truth_label"] = 0
        changed = feature_table(rows)
        np.testing.assert_array_equal(initial["X"], changed["X"])
        self.assertFalse(any("truth" in name or "gen" in name.lower() for name in initial["feature_names"]))

    def test_like_sign_transfer_uses_one_control_yield_without_os_fit(self):
        row = dict(stratum="qcd_0to1", partition="train", activity_bin="4",
                   both_endcaps=True, pair={"mass": 3.})
        same = [dict(row, kind="same_ss", weight=10.), dict(row, kind="same_os", weight=999.)]
        mixed = [dict(row, kind="row_ss", weight=2.), dict(row, kind="row_os", weight=6.)]
        report, predicted = like_sign_transfer(same, mixed)
        self.assertEqual(report["qcd_0to1/train/4"]["predicted_os_pb"], 30.)
        self.assertEqual(predicted[0]["weight"], 30.)
        same[1]["weight"] = 1.
        self.assertEqual(like_sign_transfer(same, mixed)[0]["qcd_0to1/train/4"]["predicted_os_pb"], 30.)

    def test_missing_like_sign_control_is_not_reported_as_zero_background(self):
        row = dict(stratum="qcd_0to1", partition="train", activity_bin="4",
                   both_endcaps=True, pair={"mass": 90.})
        same = [dict(row, kind="same_os", weight=1.)]
        mixed = [dict(row, kind="row_os", weight=1.)]
        report, predicted = like_sign_transfer(same, mixed)
        self.assertIsNone(report["qcd_0to1/train/4"]["predicted_os_pb"])
        self.assertEqual(predicted, [])

    def test_like_sign_transfer_preserves_unequal_pool_exposure_and_sparse_controls(self):
        same, mixed = [], []
        # For identical conditional OS/SS physics, real yields grow linearly
        # with pool size and mixed ordered-pair weights grow quadratically.
        # A single stratum scale would change the relative mass populations.
        for activity, mass, exposure in (("4", 3., 10.), ("19", 90., 100.)):
            row = dict(stratum="qcd_0to1", partition="train", activity_bin=activity,
                       both_endcaps=True, pair={"mass": mass})
            same.extend(dict(row, kind=kind, weight=exposure) for kind in ("same_os", "same_ss"))
            mixed.extend(dict(row, kind=kind, weight=exposure ** 2) for kind in ("row_os", "row_ss"))
        sparse = dict(stratum="qcd_0to1", partition="test", activity_bin="4",
                      both_endcaps=True, pair={"mass": 3.})
        same.append(dict(sparse, kind="same_os", weight=7.))
        mixed.extend(dict(sparse, kind=kind, weight=100.) for kind in ("row_os", "row_ss"))
        report, predicted = like_sign_transfer(same, mixed)
        self.assertEqual({row["pair"]["mass"]: row["weight"] for row in predicted}, {3.: 10., 90.: 100.})
        self.assertIsNone(report["qcd_0to1/test/4"]["predicted_os_pb"])
        self.assertEqual(report["qcd_0to1/test/4"]["uncovered_same_os_pb"], 7.)
        self.assertEqual(report["qcd_0to1/test/4"]["control_missing_reason"], "no_same_event_ss_yield")
        self.assertTrue(all(row["partition"] == "train" for row in predicted))

    def test_closure_bootstrap_preserves_shared_control_and_target_covariance(self):
        row = dict(stratum="qcd_0to1", partition="test", activity_bin="4", both_endcaps=True,
                   pair={"mass": 3.}, anchor_event=0, donor_event=0, weight=10.)
        same = [dict(row, kind=kind) for kind in ("same_ss", "same_os")]
        mixed = [dict(row, kind=kind, donor_event=1, weight=2.) for kind in ("row_ss", "row_os")]
        class FixedCounts:
            def __init__(self):
                self.counts = iter((np.array([1, 1]), np.array([2, 3]), np.array([3, 1])))
            def poisson(self, mean, count):
                return next(self.counts)
        result = like_sign_closure_bootstrap(same, mixed, 2, 3, FixedCounts(), edges=[2., 4.])
        self.assertGreater(result["prediction_std"][0], 0)
        self.assertEqual(result["difference_std"], [0.])
        self.assertEqual(result["nominal_prediction"], result["nominal_target"])
        self.assertGreater(result["prediction_target_covariance"][0][0], 0)

    def test_more_copies_of_mixed_pairs_do_not_create_source_statistics(self):
        row = dict(stratum="qcd_0to1", partition="train", activity_bin="4", both_endcaps=True,
                   pair={"mass": 3.}, anchor_event=0, donor_event=0, weight=10.)
        same = [dict(row, kind="same_ss"), dict(row, kind="same_os", anchor_event=1, donor_event=1)]
        mixed = [dict(row, kind=kind, donor_event=1, weight=2.) for kind in ("row_ss", "row_os")]
        first = like_sign_closure_bootstrap(same, mixed, 2, 50, np.random.default_rng(8), edges=[2., 4.])
        copied = like_sign_closure_bootstrap(same, mixed * 20, 2, 50, np.random.default_rng(8), edges=[2., 4.])
        np.testing.assert_allclose(copied["difference_std"], first["difference_std"])
        self.assertEqual(copied["invalid_control_repetitions"], first["invalid_control_repetitions"])

    def test_sparse_bootstrap_control_is_reported_instead_of_zero_error(self):
        row = dict(stratum="qcd_0to1", partition="train", activity_bin="4", both_endcaps=True,
                   pair={"mass": 3.}, anchor_event=0, donor_event=0, weight=10.)
        same = [dict(row, kind="same_ss"), dict(row, kind="same_os")]
        mixed = [dict(row, kind=kind, donor_event=1, weight=2.) for kind in ("row_ss", "row_os")]
        class FixedCounts:
            def __init__(self):
                self.counts = iter((np.array([1, 0]), np.array([1, 1])))
            def poisson(self, mean, count):
                return next(self.counts)
        result = like_sign_closure_bootstrap(same, mixed, 2, 2, FixedCounts(), edges=[2., 4.])
        self.assertEqual(result["invalid_control_repetitions"], 1)
        self.assertTrue(result["intervals_conditional_on_retained_control_support"])
        self.assertNotIn("difference_std", result)

    def test_uncovered_and_training_pools_do_not_enter_held_out_closure(self):
        row = dict(stratum="qcd_0to1", partition="test", activity_bin="4", both_endcaps=True,
                   pair={"mass": 3.}, anchor_event=0, donor_event=0, weight=10.)
        same = [dict(row, kind="same_os"), dict(row, kind="same_ss", partition="train")]
        mixed = [dict(row, kind=kind, donor_event=1, weight=2.) for kind in ("row_ss", "row_os")]
        result = like_sign_closure_bootstrap(same, mixed, 2, 20, np.random.default_rng(9), partition="test", edges=[2., 4.])
        self.assertFalse(result["prediction_available"])
        self.assertEqual(result["same_os_control_coverage"], 0.)

    def test_joint_closure_does_not_use_os_to_choose_normalization(self):
        row = dict(stratum="qcd_0to1", partition="test", activity_bin="4", both_endcaps=True,
                   pair={"mass": 3.}, anchor_event=0, donor_event=0, weight=10.)
        same = [dict(row, kind="same_ss"), dict(row, kind="same_os")]
        mixed = [dict(row, kind=kind, donor_event=1, weight=2.) for kind in ("row_ss", "row_os")]
        before = like_sign_closure_bootstrap(same, mixed, 2, 50, np.random.default_rng(8), edges=[2., 4.])
        same[1]["weight"] = 1000.
        after = like_sign_closure_bootstrap(same, mixed, 2, 50, np.random.default_rng(8), edges=[2., 4.])
        self.assertEqual(after["nominal_prediction"], before["nominal_prediction"])
        self.assertEqual(after["prediction_std"], before["prediction_std"])


if __name__ == "__main__":
    unittest.main()
