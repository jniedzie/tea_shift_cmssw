#!/usr/bin/env python3
# Code style: AI-code. AI-generated locked complete-model signal transfer check.
"""Assess previously inspected signals after the full-input family lock.

Only full-size models from the locked current-study BDT/DNN winner families
are opened. All three declared seeds and their original validation cuts are
retained. Signals are reporting inputs, never used to choose a model or cut.
"""
import argparse
import importlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np

from shift_vertex_efficiency import summarize_vertex_efficiency
from shift_vertex_signal_scores import (
    load, prediction_packet, read, reference, reject_parent_overlap, require,
    validate_loaded_sources, validate_parent_alignment, verify, verify_cache_sources,
)


SEEDS = (71, 72, 73)
IDENTITY_COLUMNS = ("ids", "pair_index", "source_index", "point")


def selected_models(protocol, lock):
    """Preserve all seeds of full-size winners; no ranking on signal outcomes."""
    require(protocol.get("domain") == "current_CMSIR5" and lock.get("domain") == "current_CMSIR5",
            "This transfer check requires the separate current CMSIR5 study")
    require(lock.get("selection_uses_test") is False and lock.get("all81fits_before_development") is True,
            "A complete train/validation-only family lock is required")
    require(tuple(protocol["seeds"]) == SEEDS, "Study seed schema differs")
    roles = {}
    for algorithm in ("bdt", "dnn"):
        for role in ("rejection", "acceptance"):
            family = lock["winners"][algorithm][role]
            if family is None:
                continue
            require(lock["families"][family]["algorithm"] == algorithm, "Locked family algorithm differs")
            roles.setdefault(family, []).append(algorithm + "_" + role)
    selected = [spec for spec in protocol["recipes"] if spec["family"] in roles and spec["fraction"] == 1.]
    for family in roles:
        members = [spec for spec in selected if spec["family"] == family]
        require(len(members) == 3 and {spec["seed"] for spec in members} == set(SEEDS),
                "Full-size locked family must retain all three seeds")
    require(bool(selected), "No locked full-size winner models")
    return selected, roles


def unknown_response_bounds(parent_ids, weights, pair_ids, labels, scores, cut, qualified):
    """Conservative label-only full-parent bounds; do not guess unknown truth."""
    lookup = {tuple(identity): index for index, identity in enumerate(parent_ids)}
    known, unresolved = np.zeros(len(parent_ids), dtype=bool), np.zeros(len(parent_ids), dtype=bool)
    for identity, label, value, valid in zip(pair_ids, labels, scores, qualified):
        parent = lookup[tuple(identity)]
        if value >= cut:
            if valid:
                known[parent] = True
            elif label == -1:
                unresolved[parent] = True
    possible = known | unresolved
    total_weight = float(np.sum(weights))
    return {
        "known_selected_parents": int(known.sum()),
        "passing_unknown_additional_parents": int(np.sum(unresolved & ~known)),
        "upper_selected_parents": int(possible.sum()), "GEN_parents": len(parent_ids),
        "lower_empirical_efficiency": float(known.mean()) if len(known) else None,
        "upper_empirical_efficiency": float(possible.mean()) if len(possible) else None,
        "lower_weighted_efficiency": float(np.sum(weights[known]) / total_weight) if total_weight else None,
        "upper_weighted_efficiency": float(np.sum(weights[possible]) / total_weight) if total_weight else None,
        "interpretation": "Label uncertainty only: upper assigns passing unknown candidates to the signal; known fake/non-signal genuine candidates stay excluded. These are not confidence intervals.",
    }


def point_metadata(point, ids, weights, response, checked):
    """Tie mass/c-tau labels to the same audited complete GEN parent population."""
    entry = response["points"][point]
    require(entry["response_available"] is True and entry["unknown_GEN_response_events"] == 0,
            "Point lacks a complete established GEN response: " + point)
    require(entry["events"] == len(ids), "Point GEN denominator differs: " + point)
    ref = entry["proper_length_source"]
    verify(ref); checked.append(ref)
    audit = read(ref["path"])
    audited_ids = np.asarray([row["id"] for row in audit["event_rows"]], dtype=np.uint64)
    audited_weights = np.asarray([row["weight"] for row in audit["event_rows"]], dtype=float)
    require(np.array_equal(ids, audited_ids) and np.array_equal(weights, audited_weights),
            "Signal parent IDs/weights differ from proper-lifetime authority: " + point)
    ctau = float(entry["proper_ctau_mm"])
    require(np.isfinite(ctau) and ctau > 0
            and np.isclose(ctau, audit["expected_proper_length_mm"], rtol=1e-10, atol=0),
            "Mean proper lifetime differs from GEN audit: " + point)
    return {"mass_gev": entry["mass_gev"], "proper_ctau_mm": ctau,
            "events": len(ids), "proper_length_authority": ref}


def summarize_predictions(reco, labels, events, predicted, thresholds, point_info):
    result = {}
    qualified = labels["genuine_signal_associated"]
    for name, values in predicted.items():
        model = {"threshold": thresholds[name], "points": {}}
        model["overall"] = summarize_vertex_efficiency(events["ids"], events["native_weight"],
            reco["ids"], labels["y"], values, thresholds[name], qualified_genuine_mask=qualified)
        for point in sorted(point_info):
            parents, pairs = events["point"] == point, reco["point"] == point
            response = summarize_vertex_efficiency(events["ids"][parents], events["native_weight"][parents],
                reco["ids"][pairs], labels["y"][pairs], values[pairs], thresholds[name],
                qualified_genuine_mask=qualified[pairs])
            response["unknown_truth_bounds"] = unknown_response_bounds(events["ids"][parents],
                events["native_weight"][parents], reco["ids"][pairs], labels["y"][pairs],
                values[pairs], thresholds[name], qualified[pairs])
            model["points"][point] = response
        result[name] = model
    return result


def render(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    points = sorted(summary["points"], key=lambda name: (summary["points"][name]["mass_gev"],
                                                       summary["points"][name]["proper_ctau_mm"]))
    names = list(summary["models"])
    labels = [f"{summary['points'][point]['mass_gev']:g} GeV\ncτ={summary['points'][point]['proper_ctau_mm']:.3g} mm" for point in points]
    rowlabels = [name.replace("_f100_", ", ").replace("_", " ") for name in names]
    footer = "Previously inspected signal points; no model or cut selection here. No real recording acceptance or data/MC correction."
    with PdfPages(output / "signal_transfer.pdf") as pdf:
        for field, title in (("genuine_pair_retention", "Known signal genuine-pair retention at frozen validation cuts"),
                             ("full_parent_selected", "Known signal selected events / all generated parents")):
            matrix = np.full((len(names), len(points)), np.nan)
            for i, name in enumerate(names):
                for j, point in enumerate(points):
                    value = summary["models"][name]["points"][point][field]["empirical"]["efficiency"]
                    if value is not None:
                        matrix[i, j] = 100 * value
            fig, ax = plt.subplots(figsize=(13.5, max(5.8, .47 * len(names) + 2.5)))
            im = ax.imshow(matrix, vmin=0, vmax=100 if field == "genuine_pair_retention" else 25,
                           cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(points)), labels, rotation=20, ha="right")
            ax.set_yticks(range(len(names)), rowlabels, fontsize=9)
            for i, name in enumerate(names):
                for j, point in enumerate(points):
                    stats = summary["models"][name]["points"][point][field]["empirical"]
                    text = f"{stats['passed']}/{stats['total']}" if stats["total"] else "undefined"
                    ax.text(j, i, text, ha="center", va="center", fontsize=9,
                            color="white" if matrix[i, j] < (50 if field == "genuine_pair_retention" else 12.5) else "black")
            ax.set_title(title + "\nAll three seeds of each validation-locked winner family; full training size")
            fig.colorbar(im, ax=ax, label="Efficiency (%)")
            fig.text(.03, .015, footer, fontsize=8)
            fig.tight_layout(rect=(0, .045, 1, 1)); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(14.5, 7))
        for i, name in enumerate(names):
            stats = [summary["models"][name]["points"][point]["genuine_pair_retention"]["empirical"] for point in points]
            supported = np.asarray([row["efficiency"] is not None and row["wilson95"] is not None for row in stats])
            x = np.arange(len(points))[supported] + (i - (len(names) - 1) / 2) * .055
            values = np.asarray([row["efficiency"] for row in stats if row["efficiency"] is not None and row["wilson95"] is not None])
            bounds = np.asarray([row["wilson95"] for row in stats if row["efficiency"] is not None and row["wilson95"] is not None])
            if len(values):
                ax.errorbar(x, 100*values, yerr=100*np.maximum(0, [values-bounds[:, 0], bounds[:, 1]-values]),
                            fmt="o", capsize=2, label=rowlabels[i], markersize=4)
        ax.set_xticks(range(len(points)), labels, rotation=20, ha="right")
        ax.set_ylim(0, 102); ax.set_ylabel("Known signal genuine pairs retained (%)")
        ax.set_title("Small-sample uncertainty: empirical Wilson 95% intervals\nIntervals omitted if multiple qualified candidates share a parent")
        ax.legend(fontsize=8, ncol=2, loc="lower left"); ax.grid(alpha=.2)
        fig.text(.03, .015, footer, fontsize=8); fig.tight_layout(rect=(0, .045, 1, 1)); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(13.5, max(5.8, .47 * len(names) + 2.5)))
        ax.axis("off")
        cells = [[f"{100*summary['models'][name]['points'][point]['unknown_truth_bounds']['lower_empirical_efficiency']:.1f}–"
                  f"{100*summary['models'][name]['points'][point]['unknown_truth_bounds']['upper_empirical_efficiency']:.1f}%"
                  for point in points] for name in names]
        table = ax.table(cellText=cells, rowLabels=rowlabels, colLabels=labels, cellLoc="center", loc="center")
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2.0)
        fig.suptitle("Conservative bounds from unresolved truth labels\n"
                     "Known-selected to known-selected OR passing-unknown parents / all GEN parents",
                     fontsize=11, y=.94)
        fig.text(.03, .025, "These ranges describe unknown labels, not statistical confidence intervals. Known non-signal genuine and fake pairs stay excluded.", fontsize=8)
        fig.tight_layout(rect=(.29, .065, .99, .88)); pdf.savefig(fig); plt.close(fig)


def run(training, cache, response_reference, output):
    training, cache, response_reference, output = [Path(path).resolve() for path in
                                                  (training, cache, response_reference, output)]
    # This guard precedes any cache or model opening.
    require((training / "selection_locked.json").exists(), "Selection lock required before signal evaluation")
    require(output.is_relative_to(Path(__file__).resolve().parents[1] / "plots"), "Final outputs must be in repository plots")
    require(not output.exists(), "Refusing signal transfer evidence overwrite")
    lock = read(training / "selection_locked.json"); verify(lock["protocol"])
    require(lock["protocol"] == reference(training / "protocol.json"), "Locked protocol belongs to another study")
    protocol = read(lock["protocol"]["path"])
    specs, roles = selected_models(protocol, lock)
    require(len(protocol["recipes"]) == 81 and set(lock["validation_fits"]) == {spec["name"] for spec in protocol["recipes"]},
            "The locked complete study must contain all 81 declared fit receipts")
    checked = [reference(training / "selection_locked.json"), lock["protocol"], reference(response_reference)]
    for ref in lock["validation_fits"].values():
        verify(ref); checked.append(ref)
    source_refs = {}
    for filename, ref in protocol["sources"].items():
        require(Path(ref["path"]).resolve() == (training / "source_snapshot" / filename).resolve(), "Predictor source outside frozen study")
        verify(ref); checked.append(ref); source_refs[filename] = [ref]
    manifest = cache / "inputs_resolved.json"
    metadata = read(manifest); checked.append(reference(manifest))
    for ref in metadata["cache"].values():
        verify(ref); checked.append(ref)
    verify_cache_sources(metadata, checked)
    reco, labels, events = [load(metadata["cache"][name]["path"]) for name in ("reco", "labels", "events")]
    validate_parent_alignment(reco, labels, events, metadata)
    verify(protocol["events"]); checked.append(protocol["events"])
    reject_parent_overlap(events["ids"], load(protocol["events"]["path"])["ids"])
    packet = prediction_packet(reco, protocol["input_feature_names"])
    response = read(response_reference)
    point_info = {str(point): point_metadata(str(point), events["ids"][events["point"] == point],
                    events["native_weight"][events["point"] == point], response, checked) for point in sorted(set(events["point"]))}
    sys.path.insert(0, str(training / "source_snapshot"))
    validate_loaded_sources(source_refs)
    for name in ("vertex_complete_bdt", "neural_vertex_categories", "vertex_complete_neural"):
        importlib.import_module(name)
    validate_loaded_sources(source_refs)
    predicted, thresholds, model_refs, parity = {}, {}, {}, {}
    for spec in specs:
        name = spec["name"]
        fit = read(lock["validation_fits"][name]["path"])
        require(fit["recipe"] == spec and fit["protocol"] == lock["protocol"], "Selected fit recipe/protocol differs")
        require(fit["validation"]["threshold"] == lock["thresholds"][name], "Frozen validation threshold differs")
        for key in ("model", "history"):
            verify(fit[key]); checked.append(fit[key])
        with Path(fit["model"]["path"]).open("rb") as stream:
            bundle = pickle.load(stream)
        require(bundle["recipe"] == spec and bundle["requires_gen"] is False
                and bundle["category_required"] is True and tuple(bundle["feature_names"]) == tuple(protocol["input_feature_names"])
                and bundle["threshold"] == lock["thresholds"][name], "Selected model inference contract/cut differs")
        require(tuple(bundle["model"].categories_) == tuple(protocol["category_names"]), "Fixed model category schema differs")
        score = bundle["model"].predict_proba(packet["X"], packet["topology"])[:, 1]
        with Path(fit["model"]["path"]).open("rb") as stream:
            again = pickle.load(stream)
        require(np.array_equal(score, again["model"].predict_proba(packet["X"], packet["topology"])[:, 1]), "Signal model reload parity differs")
        require(score.shape == labels["y"].shape and np.isfinite(score).all(), "Invalid signal predictions")
        predicted[name], thresholds[name], model_refs[name] = score, bundle["threshold"], fit["model"]
        parity[name] = {"bitwise_reload": True, "rows": len(score)}
    validate_loaded_sources(source_refs)
    summary = {"schema": "shift-complete-vertex-signal-transfer-v1", "complete": True,
        "domain": "current_CMSIR5", "selection_uses_signal": False, "model_selection_allowed": False,
        "thresholds_unchanged": True, "previously_inspected_signals": True, "independent_confirmation": False,
        "selection": reference(training / "selection_locked.json"), "protocol": lock["protocol"],
        "cache": reference(manifest), "response_reference": reference(response_reference), "points": point_info,
        "selected_family_roles": roles, "model_references": model_refs, "parity": parity,
        "inference_keys": ["X119", "reconstructed_topology"], "verified_references": checked,
        "models": summarize_predictions(reco, labels, events, predicted, thresholds, point_info),
        "counts": {"GEN_parents": len(events["ids"]), "pairs": len(reco["ids"]),
            "qualified_genuine": int(labels["genuine_signal_associated"].sum()),
            "raw_genuine": int(np.sum(labels["y"] == 1)), "fake": int(np.sum(labels["y"] == 0)),
            "unknown": int(np.sum(labels["y"] == -1))}, "physics_ready": False,
        "limitations": ["Previously inspected signals; no new independent confirmation or signal-based selection.",
            "Few qualified genuine pairs per point; seeds are predictions on the same parents, not independent samples.",
            "Direct retention at measured mass/c-tau anchors; no interpolation or physical-rate/data correction.",
            "All covered GEN parents retained; real recording acceptance remains unresolved."]}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"; snapshot.mkdir()
    for name in (Path(__file__).name, "shift_vertex_efficiency.py", "shift_vertex_signal_scores.py"):
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    np.savez_compressed(output / "scores.npz", **{key: reco[key] for key in IDENTITY_COLUMNS}, **predicted)
    render(summary, output)
    summary["artifacts"] = {"scores": reference(output / "scores.npz"), "pdf": reference(output / "signal_transfer.pdf")}
    summary["sources"] = [reference(path) for path in snapshot.iterdir()]
    with (output / "summary.json").open("x") as target:
        json.dump(summary, target, indent=2, allow_nan=False); target.write("\n")
    print(json.dumps({"output": str(output), "models": len(predicted), "counts": summary["counts"]}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--response-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.training, args.cache, args.response_reference, args.output)
