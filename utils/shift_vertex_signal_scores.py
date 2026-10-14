#!/usr/bin/env python3
# Code style: AI-code. AI-generated locked RECO-only signal inference adapter.
"""Score complete signal caches with the eight previously frozen classifiers.

All model choices, source snapshots and cuts come from a hash-checked reference
evaluation. Only X, feature_names and reconstructed topology reach prediction.
This adapter performs no training, signal-based selection or cut adjustment.
"""
import argparse
import hashlib
import importlib
import json
import numbers
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np


INFO_MODELS = (
    "info_global_rich_s71", "info_global_rich_s72",
    "info_shared_rich_accept10_s72", "info_shared_rich_accept10_s71",
)
STRICT_MODEL = "dnn_r4_l3_c3_e0_h64_s71"
BDT_MODELS = (
    "reference_uniform_bdt_3p0_s71", "expanded_d3_s0_n400", "expanded_d3_s3_n400",
)
EXPECTED_MODELS = INFO_MODELS + (STRICT_MODEL,) + BDT_MODELS
IDENTITY_COLUMNS = ("ids", "pair_index", "source_index", "point")


def reference(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def verify(ref):
    if reference(ref["path"])["sha256"] != ref["sha256"]:
        raise ValueError("Changed sealed artifact: " + ref["path"])
    return ref


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_reference_flags(report):
    for name, required in (("complete", True), ("selection_uses_signal", False),
                           ("model_selection_allowed", False), ("thresholds_unchanged", True)):
        require(report.get(name) is required, "Unsafe reference evaluation flag: " + name)
    require(set(report["models"]) == set(EXPECTED_MODELS), "Reference must define exactly the eight fixed models")
    require(set(report["model_references"]) == set(EXPECTED_MODELS), "Reference model artifacts differ")
    for name in EXPECTED_MODELS:
        value = report["models"][name]["threshold"]
        require(not isinstance(value, bool) and isinstance(value, numbers.Real)
                and np.isfinite(value), "Invalid reference threshold: " + name)


def prediction_packet(reco, expected_names):
    """Return the entire inference allowlist; discard truth and nuisance fields."""
    names = np.asarray(reco["feature_names"])
    require(names.shape == (119,) and len(set(map(str, names))) == 119,
            "Explicit unique 119-feature schema required")
    require(list(map(str, names)) == list(expected_names), "Rich feature order differs")
    x = np.asarray(reco["X"])
    require(x.ndim == 2 and x.shape[1] == 119 and x.dtype.kind in "iuf",
            "Invalid reconstructed 119-column feature matrix")
    require(not np.isinf(x).any(), "Infinite reconstructed feature values")
    topology = np.asarray(reco["topology"])
    require(topology.shape == (len(x),) and topology.dtype.kind in "US",
            "Reconstructed topology rows differ")
    require(set(map(str, topology)).issubset({"Both-Both", "Near-Both", "Both-Far", "Near-Far", "Other"}),
            "Unknown reconstructed vertex category")
    return {"X": x, "feature_names": names, "topology": topology}


def _integer_vector(value, length, name):
    value = np.asarray(value)
    require(value.shape == (length,) and value.dtype.kind in "iu" and np.all(value >= 0),
            "Invalid integer identity/count column: " + name)
    return value


def _ids(value, name):
    value = np.asarray(value)
    require(value.ndim == 2 and value.shape[1] == 3 and value.dtype == np.dtype("uint64"),
            "Exact uint64 (run,lumi,event) IDs required: " + name)
    return value


def validate_parent_alignment(reco, labels, events, metadata):
    """Keep all covered parents and allow multiple indexed candidates per parent."""
    require(metadata.get("complete") is True and metadata.get("all_parent_events_retained") is True,
            "Complete covered GEN-parent cache required; missing outputs are unknown")
    require(metadata.get("sample_kind") == "simulation_only"
            and metadata.get("sample_role") == "validation_only"
            and metadata.get("model_selection_allowed") is False,
            "Unqualified signal sample role")
    require(metadata.get("candidate_selection_applied") is False,
            "A candidate-prefiltered cache cannot establish full GEN response")
    parent_ids = _ids(events["ids"], "events")
    pair_ids = _ids(reco["ids"], "reco")
    n_events, n_pairs = len(parent_ids), len(pair_ids)
    require(n_events == metadata["events"] and n_pairs == metadata["candidates"], "Cache counts differ")
    require(len(set(map(tuple, parent_ids))) == n_events, "Duplicate GEN parents")
    parent_sources = _integer_vector(events["source_index"], n_events, "parent source_index")
    pair_sources = _integer_vector(reco["source_index"], n_pairs, "pair source_index")
    pair_event_index = _integer_vector(reco["event_index"], n_pairs, "event_index")
    pair_index = _integer_vector(reco["pair_index"], n_pairs, "pair_index")
    pair_count = _integer_vector(events["pair_count"], n_events, "pair_count")
    require(len({(int(source), int(event), int(pair)) for source, event, pair in
                 zip(pair_sources, pair_event_index, pair_index)}) == n_pairs,
            "Duplicate indexed candidate rows")
    for key, count in (("parent", n_events), ("pair", n_pairs)):
        data = events if key == "parent" else reco
        require(np.asarray(data["point"]).shape == (count,), "Point rows differ: " + key)
        weight = np.asarray(data["native_weight"])
        require(weight.shape == (count,) and weight.dtype.kind in "iuf"
                and np.isfinite(weight).all() and np.all(weight >= 0), "Invalid native weights: " + key)
    require(set(pair_sources).issubset(set(parent_sources)), "Orphan candidate source")
    mapped = np.empty(n_pairs, dtype=np.int64)
    for source in np.unique(parent_sources):
        parent_rows = np.flatnonzero(parent_sources == source)
        candidates = np.flatnonzero(pair_sources == source)
        index = pair_event_index[candidates]
        require(np.all(index < len(parent_rows)), "Candidate event index outside source")
        rows = parent_rows[index]
        require(np.array_equal(pair_ids[candidates], parent_ids[rows]), "Candidate parent IDs differ")
        require(np.array_equal(reco["point"][candidates], events["point"][rows]), "Candidate physical points differ")
        require(np.array_equal(reco["native_weight"][candidates], events["native_weight"][rows]), "Candidate weights differ")
        mapped[candidates] = rows
    observed = np.zeros(n_events, dtype=np.int64)
    np.add.at(observed, mapped, 1)
    require(np.array_equal(observed, pair_count), "Full-parent candidate multiplicities differ")
    require(np.all(pair_index < pair_count[mapped]), "Candidate pair index outside parent multiplicity")
    y = np.asarray(labels["y"])
    require(y.shape == (n_pairs,) and y.dtype.kind in "iu" and np.isin(y, [-1, 0, 1]).all(),
            "Invalid truth-label rows")
    for name in ("signal_associated", "genuine_signal_associated"):
        require(np.asarray(labels[name]).shape == (n_pairs,) and labels[name].dtype.kind == "b",
                "Invalid boolean signal association: " + name)
    require(np.array_equal(labels["genuine_signal_associated"], (y == 1) & labels["signal_associated"]),
            "Signal association contract differs")
    return mapped


def reject_parent_overlap(new_ids, old_ids):
    new_ids, old_ids = _ids(new_ids, "signal parents"), _ids(old_ids, "original development parents")
    require(not set(map(tuple, new_ids)).intersection(set(map(tuple, old_ids))),
            "Signal GEN parents overlap original SM training/development")


def verify_cache_sources(metadata, checked):
    """Read the two explicit cache provenance formats, never guess ref shapes."""
    schema = metadata.get("schema")
    refs = [metadata["protocol"]]
    if schema == "shift-rich-reconstructed-information-v1":
        refs.append(metadata["source_snapshot"])
        if "extractor" in metadata:
            refs.append(metadata["extractor"])
        refs.extend(metadata.get("sources", {}).values())
    elif schema == "shift-complete-signal-rich-cache-v1":
        sources = metadata["source_snapshot"]
        require(isinstance(sources, dict) and bool(sources), "Empty generic cache source mapping")
        for filename, ref in sources.items():
            require(Path(filename).name == filename and filename.endswith(".py"), "Invalid generic cache source name")
            refs.append(ref)
        refs.append(metadata["extractor"])
    else:
        raise ValueError("Unsupported rich signal cache schema: " + str(schema))
    for ref in refs:
        require(isinstance(ref, dict) and "path" in ref and "sha256" in ref,
                "Invalid source reference for rich signal cache schema")
        verify(ref); checked.append(ref)


def verified_snapshot(protocol_ref, checked):
    verify(protocol_ref); checked.append(protocol_ref)
    protocol = read(protocol_ref["path"])
    snapshot = Path(protocol_ref["path"]).parent / "source_snapshot"
    refs = {}
    for name, expected in protocol.get("sources", protocol.get("source", {})).items():
        require(Path(name).name == name and name.endswith(".py"), "Invalid frozen source name")
        actual = reference(snapshot / name)
        require(actual["sha256"] == expected["sha256"], "Changed frozen predictor source: " + name)
        refs[name] = actual; checked.append(actual)
    require(bool(refs), "Empty frozen source manifest")
    return protocol, snapshot, refs


def validate_loaded_sources(source_refs):
    """Reject preloaded live modules even when their filenames look correct."""
    for filename, allowed in source_refs.items():
        name = filename[:-3]
        if name not in sys.modules:
            continue
        loaded = Path(sys.modules[name].__file__).resolve()
        require(str(loaded) in {ref["path"] for ref in allowed}, "Predictor imported outside frozen snapshots: " + name)
        require(reference(loaded)["sha256"] == allowed[0]["sha256"], "Loaded predictor source differs: " + name)


def import_predictors(snapshots, source_refs):
    for snapshot in reversed(snapshots):
        sys.path.insert(0, str(snapshot))
    validate_loaded_sources(source_refs)
    modules = {name: importlib.import_module(name) for name in
               ("features", "neural_vertex", "neural_vertex_scheduled", "neural_vertex_categories",
                "neural_vertex_information_study", "uniform_bdt", "inference", "portable_inference")}
    validate_loaded_sources(source_refs)
    return modules


def resolve_models(report, checked):
    """Trace every model/cut back through its original locked validation record."""
    validate_reference_flags(report)
    for key in ("selection", "protocol", "cache", "previous_signal_transfer"):
        verify(report[key]); checked.append(report[key])
    verify(report["artifacts"]["scores"]); checked.append(report["artifacts"]["scores"])
    lock = read(report["selection"]["path"])
    require(lock["selection_uses_test"] is False, "Rich model selection used development test")
    require(set(lock["representative_models"] + lock["paired_seed_models"]) == set(INFO_MODELS),
            "Rich model choices differ from locked representatives")
    require(lock["protocol"] == report["protocol"], "Reference protocol differs from lock")
    for key in ("validation", "family_validation"):
        verify(lock[key]); checked.append(lock[key])
    rich_candidates = read(lock["validation"]["path"])
    old = read(report["previous_signal_transfer"]["path"])
    require(old.get("complete") is True and old.get("model_selection_allowed") is False
            and old.get("model_choices_and_thresholds_unchanged") is True
            and old.get("selection_uses_fresh_signal") is False, "Unqualified previous reference evaluation")
    for key in ("global_selection", "diagnostic_selection"):
        verify(old[key]); checked.append(old[key])
    old_lock = read(old["diagnostic_selection"]["path"])
    require(old_lock["selection_uses_test"] is False, "Old model selection used development test")
    for key in ("protocol", "validation", "previous_bdt_selection"):
        verify(old_lock[key]); checked.append(old_lock[key])
    old_candidates = read(old_lock["validation"]["path"])
    bdt_lock = read(old_lock["previous_bdt_selection"]["path"])
    entries, protocols, snapshots, source_refs = {}, {}, [], {}
    for name in EXPECTED_MODELS:
        if name in INFO_MODELS:
            entry, selection = rich_candidates[name], lock
        elif name == STRICT_MODEL:
            entry, selection = old_candidates[name], old_lock
        else:
            selection = bdt_lock
            entry = {"model": report["model_references"][name], "protocol": bdt_lock["protocol"]}
        cut = report["models"][name]["threshold"]
        require(cut == selection["thresholds"][name], "Reference cut differs from validation lock: " + name)
        if name not in INFO_MODELS:
            require(cut == old["models"][name]["threshold"], "Previous reference cut differs: " + name)
            require(report["model_references"][name] == old["model_references"][name], "Previous model artifact differs: " + name)
        if name in BDT_MODELS:
            for ref in entry["model"].values():
                verify(ref); checked.append(ref)
        else:
            verify(entry["model"]); checked.append(entry["model"])
            require(entry["model"] == report["model_references"][name], "Locked model artifact differs: " + name)
            verify(entry["history"]); checked.append(entry["history"])
        protocol, snapshot, refs = verified_snapshot(entry["protocol"], checked)
        for filename, ref in refs.items():
            allowed = source_refs.setdefault(filename, [])
            require(not allowed or allowed[0]["sha256"] == ref["sha256"], "Incompatible shared frozen source: " + filename)
            if ref not in allowed:
                allowed.append(ref)
        if snapshot not in snapshots:
            snapshots.append(snapshot)
        entries[name], protocols[name] = entry, protocol
    return entries, protocols, snapshots, source_refs


def predict_models(report, entries, modules, packet):
    raw50 = packet["X"][:, :50]
    require(tuple(packet["feature_names"][:50]) == tuple(modules["features"].FEATURE_NAMES),
            "Raw50 reference feature contract differs")
    predicted, parity = {}, {}
    for name in EXPECTED_MODELS:
        entry = entries[name]
        if name in BDT_MODELS:
            with Path(entry["model"]["native"]["path"]).open("rb") as stream:
                bundle = pickle.load(stream)
            portable = modules["portable_inference"].load_model(entry["model"]["portable"]["path"])
            values = modules["inference"].predict(bundle, raw50)
            require(np.array_equal(values, modules["portable_inference"].predict(portable, raw50)),
                    "BDT portable/native parity differs: " + name)
            parity[name] = {"bitwise_portable_native_parity": True}
        else:
            with Path(entry["model"]["path"]).open("rb") as stream:
                bundle = pickle.load(stream)
            require(bundle.get("requires_gen") is False, "GEN-dependent predictor rejected: " + name)
            require(bundle["threshold"] == report["models"][name]["threshold"], "Bundled cut differs: " + name)
            if name in INFO_MODELS:
                values = modules["neural_vertex_information_study"].predict(bundle, packet)
            else:
                require(tuple(bundle["feature_names"]) == tuple(packet["feature_names"][:50])
                        and np.array_equal(bundle["columns"], np.arange(50)), "Strict DNN raw50 contract differs")
                values = bundle["model"].predict_proba(raw50)[:, 1]
            with Path(entry["model"]["path"]).open("rb") as stream:
                again = pickle.load(stream)
            reloaded = (modules["neural_vertex_information_study"].predict(again, packet)
                        if name in INFO_MODELS else again["model"].predict_proba(raw50)[:, 1])
            require(np.array_equal(values, reloaded), "Neural reload parity differs: " + name)
            parity[name] = {"bitwise_reload_parity": True}
        require(values.shape == (len(raw50),) and np.isfinite(values).all(), "Invalid score vector: " + name)
        predicted[name] = values
        parity[name]["rows"] = len(raw50)
    return predicted, parity


def compare_reference_scores(predicted, previous, reco, report):
    """Keep actual scores; audit floating-point replay and exact cut decisions."""
    for key in IDENTITY_COLUMNS:
        require(np.array_equal(reco[key], previous[key]), "Reference replay identity differs: " + key)
    models = {}
    for name in EXPECTED_MODELS:
        values, old = np.asarray(predicted[name]), np.asarray(previous[name])
        require(values.shape == old.shape and np.isfinite(old).all(), "Reference score shape/values differ: " + name)
        maximum = float(np.max(np.abs(values - old))) if len(values) else 0.0
        require(np.allclose(values, old, rtol=0, atol=1e-6), "Reference numerical replay differs: " + name)
        cut = report["models"][name]["threshold"]
        require(np.array_equal(values >= cut, old >= cut), "Reference replay cut decisions differ: " + name)
        models[name] = {"bitwise_equal": bool(np.array_equal(values, old)),
            "changed_rows": int(np.sum(values != old)), "max_absolute_difference": maximum,
            "absolute_tolerance": 1e-6, "relative_tolerance": 0,
            "threshold_decisions_identical": True}
    return {"reference_population_replayed": True,
        "all_eight_score_vectors_bitwise_equal": all(row["bitwise_equal"] for row in models.values()),
        "all_eight_threshold_decisions_identical": True, "models": models,
        "interpretation": "Actual predictor outputs preserved; CPU/backend rounding can differ while frozen-cut decisions agree."}


def run(cache, reference_evaluation, output):
    cache, reference_evaluation, output = [Path(path).resolve() for path in (cache, reference_evaluation, output)]
    require(output.is_relative_to(Path(__file__).resolve().parents[1] / "plots"), "Output must be inside repository plots")
    require(not output.exists(), "Refusing to overwrite scoring evidence")
    manifest = cache / "inputs_resolved.json"
    if not manifest.exists():
        manifest = cache / "inputs.json"
    metadata = read(manifest)
    checked = [reference(manifest), reference(reference_evaluation)]
    for ref in metadata["cache"].values():
        verify(ref); checked.append(ref)
    verify_cache_sources(metadata, checked)
    reco, labels, events = [load(metadata["cache"][key]["path"]) for key in ("reco", "labels", "events")]
    validate_parent_alignment(reco, labels, events, metadata)
    report = read(reference_evaluation)
    entries, protocols, snapshots, source_refs = resolve_models(report, checked)
    first_protocol = protocols[INFO_MODELS[0]]
    verify(first_protocol["events"]); checked.append(first_protocol["events"])
    reject_parent_overlap(events["ids"], load(first_protocol["events"]["path"])["ids"])
    packet = prediction_packet(reco, first_protocol["input_feature_names"])
    modules = import_predictors(snapshots, source_refs)
    predicted, parity = predict_models(report, entries, modules, packet)
    validate_loaded_sources(source_refs)
    replay = {"reference_population_replayed": False}
    if reference(manifest)["sha256"] == report["cache"]["sha256"]:
        old_scores = load(report["artifacts"]["scores"]["path"])
        replay = compare_reference_scores(predicted, old_scores, reco, report)
    output.mkdir(parents=True)
    source = output / "source_snapshot"; source.mkdir()
    shutil.copyfile(__file__, source / Path(__file__).name)
    np.savez_compressed(output / "scores.npz", **{key: reco[key] for key in IDENTITY_COLUMNS}, **predicted)
    summary = {"schema": "shift-locked-rich-signal-scores-v1", "complete": True,
        "selection_uses_signal": False, "model_selection_allowed": False, "thresholds_unchanged": True,
        "selection": report["selection"], "protocol": report["protocol"], "cache": reference(manifest),
        "previous_signal_transfer": report["previous_signal_transfer"],
        "reference_evaluation": reference(reference_evaluation), "model_references": report["model_references"],
        "models": {name: {"threshold": report["models"][name]["threshold"]} for name in EXPECTED_MODELS},
        "representative_models": report["representative_models"], "paired_seed_models": report["paired_seed_models"],
        "counts": {"all_GEN_parents": len(events["ids"]), "pairs": len(reco["ids"])},
        "inference_keys": sorted(packet), "requires_gen_for_inference": False,
        "parity": parity, "reference_replay": replay, "verified_references": checked,
        "artifacts": {"scores": reference(output / "scores.npz")},
        "source": reference(source / Path(__file__).name), "physics_ready": False,
        "limitations": ["Fixed models and cuts only; no signal-based model choice or retuning.",
            "Complete supplied parents require processing provenance; missing outputs are unknown.",
            "Scores alone do not establish trigger/readout efficiency or data/MC transfer."]}
    with (output / "summary.json").open("x") as target:
        json.dump(summary, target, indent=2, allow_nan=False); target.write("\n")
    print(json.dumps({"output": str(output), "pairs": len(reco["ids"]), "models": len(predicted), "replay": replay}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.reference_evaluation, args.output)
