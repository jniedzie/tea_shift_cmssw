# Code style: AI-code. AI-generated source-grouped RECO121 cross-fit training.
"""Score each ownership group with a model trained on the other half.

The retained November 18 RECO121 trainer supplies the network, preprocessing,
loss and checkpoint restoration. This module owns only cohort partitions,
declared fitting weights, validation cuts and cross-fit provenance. Results
remain development results because earlier samples have already been inspected.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from shift_classifier_labels import TASK_NAMES, TASK_LABEL_SCHEMA, task_contract, task_labels_for_data


SM_PROCESSES = ("qcd", "dy", "jpsi")
RECIPE = dict(
    architecture="shared", strength=3.0, category_strength=3.0,
    conditional_strength=1.0, acceptance_strength=0.0, hidden=(64, 32),
    epochs=200, patience=25, warmup_epochs=20, batch_size=512,
    feature_transform="signedlog", learning_rate=0.0005, dropout=0.1,
    weight_decay=0.0005, min_delta=1e-5,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _uniform(key, purpose):
    value = hashlib.sha256((purpose + ":" + str(key)).encode()).digest()[:8]
    return int.from_bytes(value, "big") / 2**64


def ownership_partitions(group_id):
    """Use immutable ownership alone; labels, features and weights cannot enter."""
    groups = np.asarray(group_id)
    if groups.ndim != 1 or groups.dtype.kind not in "US" or np.any(groups == ""):
        raise ValueError("group_id must contain nonempty ownership strings")
    fold = np.array([int(_uniform(g, "crossfit-half-v1") >= 0.5) for g in groups], dtype=np.int8)
    inner_validation = np.array([_uniform(g, "crossfit-inner-validation-v1") < 0.2 for g in groups])
    return fold, inner_validation


def validate_data(data):
    required = ("X", "feature_names", "y", "process", "point", "category", "ids",
                "pair_index", "weight", "mass", "group_id")
    missing = set(required) - set(data)
    if missing:
        raise ValueError("Missing cross-fit inputs: " + ", ".join(sorted(missing)))
    n = len(data["y"])
    if data["X"].shape != (n, 121) or data["feature_names"].shape != (121,):
        raise ValueError("Cross-fit training requires the exact RECO121 matrix")
    if data["ids"].shape != (n, 3) or data["ids"].dtype.kind not in "iu":
        raise ValueError("ids must retain integer run/lumi/event values")
    for name in ("y", "process", "point", "category", "pair_index", "weight", "mass", "group_id"):
        if data[name].shape != (n,):
            raise ValueError(name + " must align with pair rows")
    if data["y"].dtype.kind not in "iu" or not np.isin(data["y"], [-1, 0, 1]).all():
        raise ValueError("Truth labels must be integer -1/0/1; unknown is not fake")
    if not np.isfinite(data["weight"]).all():
        raise ValueError("Native weights must be finite and are retained separately")
    ownership_partitions(data["group_id"])
    # The extractor must join replays and reweights into original ownership
    # components. An optional globally unambiguous event owner also checks that
    # pairs from one original event have not been assigned different components.
    owners = {}
    for i in range(n):
        if "event_owner_id" in data:
            key = str(data["event_owner_id"][i])
        else:
            key = (str(data["process"][i]), str(data["point"][i]),
                   *map(int, data["ids"][i]))
        group = str(data["group_id"][i])
        if key in owners and owners[key] != group:
            raise ValueError("One original event has conflicting ownership groups")
        owners[key] = group
    return n


def fold_masks(data, scored_fold, fit_mode="sm-bsm"):
    if scored_fold not in (0, 1) or fit_mode not in ("sm-only", "sm-bsm"):
        raise ValueError("Invalid scored fold or fitting mode")
    fold, inner = ownership_partitions(data["group_id"])
    allowed = np.ones(len(fold), dtype=bool)
    if fit_mode == "sm-only":
        allowed = np.isin(data["process"], SM_PROCESSES)
    labelled = (data["y"] >= 0) & allowed
    train = (fold != scored_fold) & ~inner & labelled
    validation = (fold != scored_fold) & inner & labelled
    scored = fold == scored_fold
    sets = [set(data["group_id"][mask]) for mask in (train, validation, scored)]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Original ownership crosses training/validation/scored partitions")
    for mask in (train, validation):
        if set(data["y"][mask]) != {0, 1}:
            raise ValueError("Each fit needs both known labels in training and inner validation")
    if not np.any(validation & (data["y"] == 1) & np.isin(data["process"], SM_PROCESSES)):
        raise ValueError("Inner validation lacks genuine SM pairs for cut calibration")
    return train, validation, scored


def fitting_weights(y, process, point, minimum=50):
    """Equal class totals, smoothed process totals, then smoothed point totals.

    An occupied process/point receives relative mass min(1,n/50), avoiding a
    huge multiplier for a single rare example. Native physics weights do not
    enter this empirical objective and must remain available for evaluation.
    """
    y, process, point = map(np.asarray, (y, process, point))
    if y.shape != process.shape or y.shape != point.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Fitting weights require aligned known-label rows")
    weights = np.zeros(len(y), dtype=float)
    for label in (0, 1):
        processes = np.unique(process[y == label])
        if not len(processes):
            raise ValueError("Both known labels require empirical fitting support")
        process_mass = np.array([min(1.0, np.sum((y == label) & (process == p)) / minimum)
                                 for p in processes])
        for p, pmass in zip(processes, process_mass):
            mask = (y == label) & (process == p)
            points = np.unique(point[mask])
            point_mass = np.array([min(1.0, np.sum(mask & (point == q)) / minimum) for q in points])
            for q, qmass in zip(points, point_mass):
                rows = mask & (point == q)
                weights[rows] = 0.5 * pmass / process_mass.sum() * qmass / point_mass.sum() / rows.sum()
    return weights * len(y)


def validation_cut(scores, y, validation, process, retention=0.9):
    """Calibrate a fold's cut on its own inner-validation genuine SM only."""
    selected = np.asarray(validation) & (np.asarray(y) == 1) & np.isin(process, SM_PROCESSES)
    values = np.sort(np.asarray(scores)[selected])
    if not len(values) or not np.isfinite(values).all() or not 0 < retention <= 1:
        raise ValueError("A finite genuine-SM inner-validation score sample is required")
    return float(values[int(np.floor((1.0 - retention) * (len(values) - 1)))])


def _counts(data, mask):
    return {str(process): {str(label): int(np.sum(mask & (data["process"] == process) & (data["y"] == label)))
                          for label in (-1, 0, 1)}
            for process in np.unique(data["process"])}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def verify_model_readback(model, restored, X, categories, probe):
    """Compare identical inference batches; batch shape can change float32 BLAS rounding."""
    original = model.predict_proba(X[probe], categories[probe])[:, 1]
    recovered = restored.predict_proba(X[probe], categories[probe])[:, 1]
    np.testing.assert_array_equal(recovered, original)


def task_training_data(data, task="legacy-common-origin"):
    """Use separate target arrays internally; never rewrite original binary y."""
    y, reason = task_labels_for_data(data, task)
    return dict(data, y=y, task_y=y, task_reason=reason)


def run_crossfit(input_path, output_dir, frozen_source, fit_mode="sm-bsm", epochs=200, scored_fold=None,
                 task="legacy-common-origin", recipe_name=None):
    input_path, output_dir, frozen_source = map(Path, (input_path, output_dir, frozen_source))
    data = dict(np.load(input_path, allow_pickle=False))
    n = validate_data(data)
    training_data = task_training_data(data, task)
    if scored_fold not in (None, 0, 1):
        raise ValueError("scored_fold must be 0, 1 or None for both")
    plans = {fold: fold_masks(training_data, fold, fit_mode)
             for fold in ((0, 1) if scored_fold is None else (scored_fold,))}
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    source = frozen_source.resolve()
    source_hashes = {path.name: digest(path) for path in sorted(source.glob("*.py"))}
    if not {"vertex_pointing.py", "neural_vertex_categories.py"} <= set(source_hashes):
        raise ValueError("Missing qualified frozen RECO121 training closure")
    frozen_protocol = source.parent / "protocol.json"
    if not frozen_protocol.is_file():
        raise ValueError("Frozen source must retain its original protocol.json")
    original_protocol = json.loads(frozen_protocol.read_text())
    expected_hashes = original_protocol.get("source_sha256", {})
    if not expected_hashes or any(source_hashes.get(name) != value for name, value in expected_hashes.items()):
        raise ValueError("Frozen source differs from its qualified protocol hashes")
    # Import only after the leakage and support guards have passed. The
    # separate CLI process avoids replacing imports in another user's session.
    sys.path.insert(0, str(source))
    pointing = importlib.import_module("vertex_pointing")
    if Path(pointing.__file__).resolve().parent != source:
        raise RuntimeError("vertex_pointing was imported from another runtime")
    if tuple(map(str, data["feature_names"])) != tuple(pointing.POINTING_FEATURE_NAMES):
        raise ValueError("Input RECO121 feature order differs from the frozen schema")
    import torch
    torch.set_num_threads(1)
    nuisance = pointing.transfer_nuisance(data["X"], data["mass"])
    output_dir.mkdir(parents=True, exist_ok=False)
    recipe_name = recipe_name or ("legacy-disco" if task == "legacy-common-origin" else "bce")
    if recipe_name not in ("legacy-disco", "bce"):
        raise ValueError("Unknown declared training recipe")
    recipe = dict(RECIPE, epochs=epochs)
    if recipe_name == "bce":
        recipe.update(strength=0., category_strength=0., conditional_strength=0., acceptance_strength=0.)
    protocol = dict(
        schema="shift-vertex-source-crossfit-v2", fit_mode=fit_mode,
        task=task, task_label_schema=TASK_LABEL_SCHEMA, task_contract=task_contract(task), recipe_name=recipe_name,
        task_recipe="task-separated-reco121-" + recipe_name + "-v1",
        task_label_sha256=hashlib.sha256(training_data["y"].tobytes()).hexdigest(),
        input_path=str(input_path.resolve()), input_sha256=digest(input_path),
        frozen_source=str(source), source_sha256=source_hashes,
        frozen_protocol=str(frozen_protocol), frozen_protocol_sha256=digest(frozen_protocol),
        training=recipe, seeds=[71, 72],
        selected_scored_fold=scored_fold,
        partitions="Two ownership-hash halves; 20% inner validation within fit half only",
        grouping_contract="group_id connects all responses sharing original GEN ownership; extractor supplies closure",
        weights="Equal class; smoothed process and point masses min(1,n/50); native weights retained separately",
        cut="Unweighted 90% task-positive SM retention on the fit half's inner validation only; ties retained",
        labels="Declared task truth only; original binary y preserved; unknown excluded; no fitted-vertex residual negatives",
        inference="Exact frozen RECO121 and reconstructed category only; no source/process/point/truth attributes",
        interpretation="Exploratory source-grouped cross-fit; earlier samples inspected; no blind or deployment claim",
    )
    _write_json(output_dir / "protocol.json", protocol)
    scores = np.full(n, np.nan)
    cuts = np.full(n, np.nan)
    rows = []
    for fold, (train, validation, scored) in plans.items():
        fold_dir = output_dir / ("fold" + str(fold))
        fold_dir.mkdir()
        train_weights = fitting_weights(training_data["y"][train], data["process"][train], data["point"][train])
        val_weights = fitting_weights(training_data["y"][validation], data["process"][validation], data["point"][validation])
        model, history = pointing.train_pointing_neural(
            data["X"][train], training_data["y"][train], train_weights, nuisance[train],
            data["X"][validation], training_data["y"][validation], val_weights, nuisance[validation],
            feature_names=tuple(map(str, data["feature_names"])),
            train_category=data["category"][train], val_category=data["category"][validation],
            train_process=data["process"][train], val_process=data["process"][validation],
            seed=71 + fold, **recipe,
        )
        model.classifier_task_ = task
        model.task_label_schema_ = TASK_LABEL_SCHEMA
        model.task_contract_ = task_contract(task)
        calibration = np.full(n, np.nan)
        calibration[validation] = model.predict_proba(data["X"][validation], data["category"][validation])[:, 1]
        cut = validation_cut(calibration, training_data["y"], validation, data["process"])
        scores[scored] = model.predict_proba(data["X"][scored], data["category"][scored])[:, 1]
        cuts[scored] = cut
        model_path = fold_dir / "model.pkl"
        with model_path.open("xb") as stream:
            pickle.dump(model, stream)
        with model_path.open("rb") as stream:
            restored = pickle.load(stream)
        with gzip.open(fold_dir / "history.json.gz", "wt") as stream:
            json.dump(history, stream)
        np.savez_compressed(fold_dir / "partitions.npz", train=np.flatnonzero(train),
                            validation=np.flatnonzero(validation), scored=np.flatnonzero(scored),
                            validation_scores=calibration[validation],
                            train_fitting_weights=train_weights, validation_fitting_weights=val_weights)
        probe = np.flatnonzero(scored)[:40]
        verify_model_readback(model, restored, data["X"], data["category"], probe)
        groups = {name: sorted(set(map(str, data["group_id"][mask])))
                  for name, mask in (("train", train), ("validation", validation), ("scored", scored))}
        receipt = dict(scored_fold=fold, seed=71 + fold, threshold=cut,
                       model_sha256=digest(model_path), best_epoch=history["best_epoch"],
                       groups=groups, no_source_overlap=True,
                       counts={name: _counts(training_data, mask) for name, mask in (("train", train), ("validation", validation), ("scored", scored))})
        _write_json(fold_dir / "receipt.json", receipt)
        rows.append(receipt)
    scored_rows = np.logical_or.reduce([plan[2] for plan in plans.values()])
    if (not np.isfinite(scores[scored_rows]).all()
            or np.any((scores[scored_rows] < 0) | (scores[scored_rows] > 1))
            or not np.isfinite(cuts[scored_rows]).all()):
        raise RuntimeError("Every row must receive exactly one valid unseen-source score and validation cut")
    if digest(input_path) != protocol["input_sha256"]:
        raise RuntimeError("Input changed during training")
    if any(digest(source / name) != expected for name, expected in source_hashes.items()):
        raise RuntimeError("Frozen training source changed during cross-fit")
    fold, inner = ownership_partitions(data["group_id"])
    predictions = {key: value[scored_rows] for key, value in data.items() if key not in ("X", "feature_names")}
    predictions.update(row_index=np.flatnonzero(scored_rows), crossfit_score=scores[scored_rows], crossfit_cut=cuts[scored_rows],
                       crossfit_pass=scores[scored_rows] >= cuts[scored_rows], scored_fold=fold[scored_rows], inner_validation_role=inner[scored_rows])
    predictions.update(task_y=training_data["y"][scored_rows], task_reason=training_data["task_reason"][scored_rows],
                       task_name=np.full(int(scored_rows.sum()), task))
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    report = dict(protocol, completed=True, rows=int(scored_rows.sum()), total_input_rows=n,
                  unscored_rows=int((~scored_rows).sum()), models=rows,
                  predictions_sha256=digest(output_dir / "predictions.npz"),
                  no_source_overlap=True, deployment_ready=False,
                  all_labels=_counts(training_data, np.ones(n, dtype=bool)),
                  original_binary_labels=_counts(data, np.ones(n, dtype=bool)))
    _write_json(output_dir / "completion.json", report)
    return report


def combine_folds(input_path, fold_directories, output_dir):
    """Join two independent half jobs after exact input, ownership and row checks."""
    if len(fold_directories) != 2:
        raise ValueError("Exactly two completed half jobs are required")
    input_path, output_dir = Path(input_path), Path(output_dir)
    data = dict(np.load(input_path, allow_pickle=False)); n = validate_data(data)
    expected_hash = digest(input_path); reports = []; predictions = []; paths = {}
    for directory in map(Path, fold_directories):
        report = json.loads((directory / "completion.json").read_text())
        if not report.get("completed") or report.get("input_sha256") != expected_hash:
            raise ValueError("Half job is incomplete or uses a different input")
        fold = report.get("selected_scored_fold")
        if fold not in (0, 1) or fold in paths or len(report["models"]) != 1:
            raise ValueError("Two distinct single-fold jobs are required")
        pred = dict(np.load(directory / "predictions.npz", allow_pickle=False))
        if digest(directory / "predictions.npz") != report["predictions_sha256"]:
            raise ValueError("Half predictions differ from completion receipt")
        training_data = task_training_data(data, report.get("task", "legacy-common-origin"))
        train, validation, scored = fold_masks(training_data, fold, report["fit_mode"])
        np.testing.assert_array_equal(pred["row_index"], np.flatnonzero(scored))
        for key in ("ids", "process", "point", "category", "group_id", "y", "weight", "mass"):
            np.testing.assert_array_equal(pred[key], data[key][scored])
        if "task_y" in pred:
            np.testing.assert_array_equal(pred["task_y"], training_data["y"][scored])
            np.testing.assert_array_equal(pred["task_reason"], training_data["task_reason"][scored])
            np.testing.assert_array_equal(pred["task_name"], report.get("task", "legacy-common-origin"))
        elif report.get("task", "legacy-common-origin") != "legacy-common-origin":
            raise ValueError("Separate-task predictions must retain explicit task truth")
        receipt = report["models"][0]
        for name, mask in (("train", train), ("validation", validation), ("scored", scored)):
            if receipt["groups"][name] != sorted(set(map(str, data["group_id"][mask]))):
                raise ValueError("Half model ownership differs from its declared partition")
        if digest(directory / ("fold" + str(fold)) / "model.pkl") != receipt["model_sha256"]:
            raise ValueError("Half model differs from its receipt")
        paths[fold] = directory.resolve(); reports.append(report); predictions.append(pred)
    for key in ("fit_mode", "training", "source_sha256", "frozen_protocol_sha256", "cut", "weights",
                "task", "task_recipe", "task_label_schema", "task_label_sha256", "recipe_name"):
        if reports[0].get(key) != reports[1].get(key):
            raise ValueError("Half jobs use different " + key)
    if set(predictions[0]) != set(predictions[1]):
        raise ValueError("Half prediction schemas differ")
    joined = {key: np.concatenate([pred[key] for pred in predictions]) for key in predictions[0]}
    order = np.argsort(joined["row_index"])
    np.testing.assert_array_equal(joined["row_index"][order], np.arange(n))
    joined = {key: value[order] for key, value in joined.items()}
    if not np.isfinite(joined["crossfit_score"]).all() or not np.isfinite(joined["crossfit_cut"]).all():
        raise ValueError("Missing finite cross-fit scores or cuts")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("completion.json", "protocol.json", "predictions.npz", "fold0", "fold1")):
        raise FileExistsError("Refuse to overwrite an existing combined cross-fit result")
    for fold, directory in paths.items():
        (output_dir / ("fold" + str(fold))).symlink_to(directory / ("fold" + str(fold)), target_is_directory=True)
    np.savez_compressed(output_dir / "predictions.npz", **joined)
    report = dict(reports[0], rows=n, total_input_rows=n, unscored_rows=0,
                  selected_scored_fold=None, models=sorted([r["models"][0] for r in reports], key=lambda r: r["scored_fold"]),
                  combined_half_jobs={str(fold): str(path) for fold, path in sorted(paths.items())},
                  predictions_sha256=digest(output_dir / "predictions.npz"))
    _write_json(output_dir / "protocol.json", {key: value for key, value in report.items() if key not in ("models", "predictions_sha256")})
    _write_json(output_dir / "completion.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--fit-mode", choices=("sm-only", "sm-bsm"), default="sm-bsm")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--scored-fold", type=int, choices=(0, 1))
    parser.add_argument("--combine-folds", type=Path, nargs=2)
    parser.add_argument("--task", choices=TASK_NAMES, default="legacy-common-origin")
    parser.add_argument("--recipe", choices=("legacy-disco", "bce"))
    args = parser.parse_args()
    if args.combine_folds:
        result = combine_folds(args.input, args.combine_folds, args.output)
    else:
        if args.frozen_source is None:
            parser.error("--frozen-source is required for fitting")
        result = run_crossfit(args.input, args.output, args.frozen_source, args.fit_mode, args.epochs, args.scored_fold,
                              args.task, args.recipe)
    print(json.dumps({key: result[key] for key in ("completed", "rows", "fit_mode", "no_source_overlap", "deployment_ready")}))


if __name__ == "__main__":
    main()
