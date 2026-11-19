# Code style: AI-code. AI-generated tests for scientific cross-fit leakage guards.
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from shift_classifier_crossfit import fitting_weights, fold_masks, ownership_partitions, validate_data, validation_cut, verify_model_readback


def cohort():
    # Many independent source components with repeated pairs and both classes
    # ensure the partition proof is tested without fitting a neural network.
    groups = np.repeat(["original-gen-" + str(i) for i in range(100)], 4)
    n = len(groups)
    return dict(X=np.zeros((n, 121)), feature_names=np.array(["x" + str(i) for i in range(121)]),
                y=np.tile([0, 1, -1, 1], 100), process=np.tile(["qcd", "qcd", "signal", "signal"], 100),
                point=np.tile(["qcd-low", "qcd-low", "dp-point", "dp-point"], 100),
                category=np.full(n, "Both-Both"), ids=np.column_stack((np.ones(n, dtype=int), np.ones(n, dtype=int), np.arange(n))),
                pair_index=np.zeros(n, dtype=int), weight=np.linspace(1, 1000, n), mass=np.ones(n), group_id=groups)


class CrossFitLeakageTest(unittest.TestCase):
    def test_original_source_and_reweighted_responses_never_cross_partitions(self):
        data = cohort()
        data["point"][data["process"] == "signal"] = "reweighted-dp-point"
        for fold in (0, 1):
            train, validation, scored = fold_masks(data, fold)
            a, b, c = [set(data["group_id"][mask]) for mask in (train, validation, scored)]
            self.assertFalse(a & b or a & c or b & c)
            self.assertFalse(np.any((train | validation) & (data["y"] < 0)))
        halves, _ = ownership_partitions(data["group_id"])
        self.assertTrue(np.all(halves.reshape(-1, 4) == halves.reshape(-1, 4)[:, :1]))

    def test_sm_only_and_sm_bsm_use_identical_scored_rows(self):
        data = cohort()
        for fold in (0, 1):
            sm_train, _, sm_scored = fold_masks(data, fold, "sm-only")
            all_train, _, all_scored = fold_masks(data, fold, "sm-bsm")
            np.testing.assert_array_equal(sm_scored, all_scored)
            self.assertFalse(np.any(sm_train & (data["process"] == "signal")))
            self.assertTrue(np.any(all_train & (data["process"] == "signal")))

    def test_scored_features_labels_and_native_weights_cannot_change_partitions(self):
        data = cohort()
        before = ownership_partitions(data["group_id"])
        data["X"][:] = 1e9
        data["weight"][:] *= -100
        data["y"] = data["y"][::-1]
        data["point"][:] = "changed"
        after = ownership_partitions(data["group_id"])
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_cut_cannot_see_scored_half_or_signal_scores(self):
        y = np.ones(30, dtype=int)
        process = np.array(["qcd"] * 10 + ["signal"] * 10 + ["qcd"] * 10)
        validation = np.arange(30) < 20
        scores = np.linspace(0, 1, 30)
        cut = validation_cut(scores, y, validation, process)
        scores[(~validation) | (process == "signal")] = 1e9
        self.assertEqual(cut, validation_cut(scores, y, validation, process))

    def test_unknowns_are_rejected_by_fitting_weight_interface(self):
        with self.assertRaisesRegex(ValueError, "known-label"):
            fitting_weights(np.array([-1, 0, 1]), np.array(["qcd"] * 3), np.array(["point"] * 3))

    def test_weights_balance_labels_without_physics_or_excessive_rare_point_weight(self):
        y = np.array([0] * 100 + [1] * 100)
        process = np.array(["qcd"] * 100 + ["signal"] * 100)
        point = np.array(["qcd"] * 100 + ["large"] * 99 + ["rare"])
        weights = fitting_weights(y, process, point)
        self.assertAlmostEqual(weights[y == 0].sum(), weights[y == 1].sum())
        self.assertLess(weights[-1], 3 * weights[-2])
        self.assertAlmostEqual(weights.mean(), 1.0)

    def test_conflicting_original_event_ownership_is_a_hard_failure(self):
        data = cohort()
        data["event_owner_id"] = np.array(["event" + str(i) for i in range(len(data["y"]))])
        data["event_owner_id"][4] = data["event_owner_id"][0]
        with self.assertRaisesRegex(ValueError, "conflicting ownership"):
            validate_data(data)

    def test_serialization_readback_compares_identical_batch_shapes(self):
        class BatchShapePredictor:
            def predict_proba(self, X, categories):
                value=np.float32(.5)+np.float32(len(X))*np.float32(2**-24)
                return np.tile(np.array([1-value,value]),(len(X),1))
        data=cohort();model=BatchShapePredictor();probe=np.arange(40)
        self.assertFalse(np.array_equal(model.predict_proba(data['X'],data['category'])[:40,1],
                                        model.predict_proba(data['X'][probe],data['category'][probe])[:,1]))
        verify_model_readback(model,BatchShapePredictor(),data['X'],data['category'],probe)


if __name__ == "__main__":
    unittest.main()
