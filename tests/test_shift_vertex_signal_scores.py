# Code style: AI-code. AI-generated locked-scorer corruption and inference tests.
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location(
    "signal_scores", Path(__file__).parents[1] / "utils/shift_vertex_signal_scores.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def identities(events):
    return np.asarray([[1, 1, event] for event in events], dtype=np.uint64).reshape(-1, 3)


class SignalScoreTests(unittest.TestCase):
    def arrays(self):
        events = {"ids": identities([2**63+1, 2**63+2, 2**63+3]),
                  "source_index": np.array([0, 0, 1]), "point": np.array(["A", "A", "B"]),
                  "native_weight": np.array([2., 3., 4.]), "pair_count": np.array([2, 0, 0])}
        reco = {"ids": identities([2**63+1, 2**63+1]), "source_index": np.array([0, 0]),
                "event_index": np.array([0, 0]), "pair_index": np.array([0, 1]),
                "point": np.array(["A", "A"]), "native_weight": np.array([2., 2.])}
        labels = {"y": np.array([1, -1]), "signal_associated": np.array([True, False]),
                  "genuine_signal_associated": np.array([True, False])}
        metadata = {"complete": True, "all_parent_events_retained": True,
                    "sample_kind": "simulation_only", "sample_role": "validation_only",
                    "model_selection_allowed": False, "candidate_selection_applied": False,
                    "events": 3, "candidates": 2}
        return reco, labels, events, metadata

    def test_multiple_indexed_candidates_and_zero_candidate_parents(self):
        reco, labels, events, metadata = self.arrays()
        np.testing.assert_array_equal(module.validate_parent_alignment(reco, labels, events, metadata), [0, 0])

    def test_corrupt_parent_weights_indices_ids_and_multiplicities_rejected(self):
        cases = (("native_weight", np.array([2., 99.])), ("event_index", np.array([0, 2])),
                 ("ids", identities([2**63+1, 2**63+2])), ("pair_index", np.array([0, 0])),
                 ("pair_index", np.array([0, 99])),
                 ("source_index", np.array([0, 2])))
        for name, value in cases:
            reco, labels, events, metadata = self.arrays()
            reco[name] = value
            with self.subTest(column=name), self.assertRaises(ValueError):
                module.validate_parent_alignment(reco, labels, events, metadata)
        reco, labels, events, metadata = self.arrays()
        events["pair_count"] = np.array([1, 0, 0])
        with self.assertRaisesRegex(ValueError, "multiplicities"):
            module.validate_parent_alignment(reco, labels, events, metadata)

    def test_incomplete_population_and_truth_corruption_rejected(self):
        for flag in ("complete", "all_parent_events_retained"):
            reco, labels, events, metadata = self.arrays(); metadata[flag] = False
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "Complete covered"):
                module.validate_parent_alignment(reco, labels, events, metadata)
        reco, labels, events, metadata = self.arrays()
        labels["genuine_signal_associated"] = np.array([False, False])
        with self.assertRaisesRegex(ValueError, "association contract"):
            module.validate_parent_alignment(reco, labels, events, metadata)

    def test_empty_candidate_population_preserves_all_parents(self):
        reco, labels, events, metadata = self.arrays()
        for key in reco:
            reco[key] = reco[key][:0]
        for key in labels:
            labels[key] = labels[key][:0]
        metadata["candidates"] = 0; events["pair_count"][:] = 0
        self.assertEqual(module.validate_parent_alignment(reco, labels, events, metadata).shape, (0,))

    def test_inference_packet_excludes_truth_and_nuisance(self):
        names = np.array(["feature_" + str(i) for i in range(119)])
        reco = {"X": np.zeros((2, 119)), "feature_names": names,
                "topology": np.array(["Both-Both", "Other"]), "nuisance": object(),
                "labels": object(), "proper_lifetime": object(), "gen": object()}
        packet = module.prediction_packet(reco, names)
        self.assertEqual(set(packet), {"X", "feature_names", "topology"})
        changed = names.copy(); changed[[0, 1]] = changed[[1, 0]]
        with self.assertRaisesRegex(ValueError, "feature order"):
            module.prediction_packet(reco, changed)

    def test_exact_high_uint64_overlap_rejected_without_float_rounding(self):
        module.reject_parent_overlap(identities([2**63+1]), identities([2**63+2]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            module.reject_parent_overlap(identities([2**63+1]), identities([2**63+1]))

    def test_reference_flags_require_exact_choices_and_frozen_cuts(self):
        report = {"complete": True, "selection_uses_signal": False,
                  "model_selection_allowed": False, "thresholds_unchanged": True,
                  "models": {name: {"threshold": .5} for name in module.EXPECTED_MODELS},
                  "model_references": {name: {} for name in module.EXPECTED_MODELS}}
        module.validate_reference_flags(report)
        report["selection_uses_signal"] = True
        with self.assertRaisesRegex(ValueError, "Unsafe reference"):
            module.validate_reference_flags(report)
        report["selection_uses_signal"] = False; report["models"].pop(module.STRICT_MODEL)
        with self.assertRaisesRegex(ValueError, "eight fixed"):
            module.validate_reference_flags(report)

    def test_changed_source_digest_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.py"
            path.write_text("pass\n")
            ref = module.reference(path)
            module.verify(ref)
            path.write_text("raise RuntimeError\n")
            with self.assertRaisesRegex(ValueError, "Changed sealed artifact"):
                module.verify(ref)

    def test_explicit_old_and_generic_cache_source_reference_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.py"; path.write_text("pass\n")
            ref = module.reference(path)
            old = {"schema": "shift-rich-reconstructed-information-v1",
                   "protocol": ref, "source_snapshot": ref}
            checked = []; module.verify_cache_sources(old, checked)
            self.assertEqual(len(checked), 2)
            generic = {"schema": "shift-complete-signal-rich-cache-v1", "protocol": ref,
                       "source_snapshot": {"features.py": ref, "labels.py": ref}, "extractor": ref}
            checked = []; module.verify_cache_sources(generic, checked)
            self.assertEqual(len(checked), 4)
            generic["source_snapshot"]["features.py"] = {**ref, "sha256": "changed"}
            with self.assertRaisesRegex(ValueError, "Changed sealed"):
                module.verify_cache_sources(generic, [])
            with self.assertRaisesRegex(ValueError, "Unsupported rich"):
                module.verify_cache_sources({"schema": "unknown", "protocol": ref}, [])

    def test_numerical_replay_preserves_actual_scores_and_exact_cut_decisions(self):
        reco = {"ids": identities([1]), "pair_index": np.array([0]),
                "source_index": np.array([0]), "point": np.array(["A"])}
        old = dict(reco)
        for name in module.EXPECTED_MODELS:
            old[name] = np.array([0.8], dtype=np.float32)
        predicted = {name: old[name].copy() for name in module.EXPECTED_MODELS}
        first = module.EXPECTED_MODELS[0]
        predicted[first] = np.nextafter(predicted[first], np.float32(1.0))
        actual = predicted[first].copy()
        report = {"models": {name: {"threshold": .5} for name in module.EXPECTED_MODELS}}
        result = module.compare_reference_scores(predicted, old, reco, report)
        self.assertFalse(result["all_eight_score_vectors_bitwise_equal"])
        self.assertTrue(result["all_eight_threshold_decisions_identical"])
        self.assertEqual(result["models"][first]["changed_rows"], 1)
        np.testing.assert_array_equal(predicted[first], actual)

    def test_preloaded_live_module_rejected_even_with_identical_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            frozen = Path(temporary) / "frozen.py"
            live = Path(temporary) / "live.py"
            frozen.write_text("pass\n"); live.write_text("pass\n")
            refs = {"example_frozen_predictor.py": [module.reference(frozen)]}
            with patch.dict(module.sys.modules, {"example_frozen_predictor": SimpleNamespace(__file__=str(live))}):
                with self.assertRaisesRegex(ValueError, "outside frozen snapshots"):
                    module.validate_loaded_sources(refs)
            with patch.dict(module.sys.modules, {"example_frozen_predictor": SimpleNamespace(__file__=str(frozen))}):
                module.validate_loaded_sources(refs)

    def test_numerically_tiny_threshold_crossing_and_large_difference_rejected(self):
        reco = {"ids": identities([1]), "pair_index": np.array([0]),
                "source_index": np.array([0]), "point": np.array(["A"])}
        old = {**reco, **{name: np.array([.5]) for name in module.EXPECTED_MODELS}}
        predicted = {name: old[name].copy() for name in module.EXPECTED_MODELS}
        report = {"models": {name: {"threshold": .5} for name in module.EXPECTED_MODELS}}
        first = module.EXPECTED_MODELS[0]
        predicted[first] = np.array([.5 - 1e-8])
        with self.assertRaisesRegex(ValueError, "cut decisions"):
            module.compare_reference_scores(predicted, old, reco, report)
        predicted[first] = np.array([.6])
        with self.assertRaisesRegex(ValueError, "numerical replay"):
            module.compare_reference_scores(predicted, old, reco, report)


if __name__ == "__main__":
    unittest.main()
