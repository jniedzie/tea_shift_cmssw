#!/usr/bin/env python3
"""Histogram saved Double BDT scores without rerunning reconstruction or scoring.

Outputs contain only new BDT histograms. They can be merged with an ordinary
histogram file using hadd without duplicating cutflow or normalization metadata.
Truth is imported exclusively for MC diagnostic labels, never for a score/cut.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import uproot

CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "configs"
sys.path.insert(0, str(CONFIG_DIRECTORY))
from shift_bdt_histograms import (
    CATEGORIES, DIRECTORY, LABEL_SUFFIXES, MAP_AXES, SCORE_BINS,
    SCORE_BRANCH, histogram_name,
)

RECO_BRANCHES = (
    "nShiftMuon", "nShiftDimuonVertex", SCORE_BRANCH,
    "ShiftDimuonVertex_muonIdx1", "ShiftDimuonVertex_muonIdx2",
    "ShiftDimuonVertex_mass", "ShiftDimuonVertex_vx", "ShiftDimuonVertex_vy",
    "ShiftDimuonVertex_vz", "ShiftDimuonVertex_topologyMin",
    "ShiftDimuonVertex_topologyMax", "ShiftDimuonVertex_isOS",
    "ShiftDimuonVertex_dcaValid", "ShiftDimuonVertex_dca",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_labels(directory):
    path = Path(directory) / "labels.py"
    spec = importlib.util.spec_from_file_location("shift_bdt_mc_diagnostic_labels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def categories(pair):
    """Use the existing TEA categories; they do not inspect the BDT score."""
    result = [""]
    if int(pair["isOS"]) == 1 and int(pair["dcaValid"]) == 1 and 0.0 <= float(pair["dca"]) <= 50.0:
        result.append("Good")
    topology = {
        (0, 2): "Near-Both", (0, 3): "Near-Far",
        (2, 2): "Both-Both", (2, 3): "Both-Far",
    }.get((int(pair["topologyMin"]), int(pair["topologyMax"])), "Other")
    result.append(topology)
    return result


def validate_score_branch(tree):
    if SCORE_BRANCH not in tree:
        raise ValueError(f"Missing required saved BDT score branch: {SCORE_BRANCH}")
    if tree[SCORE_BRANCH].typename != "double[]":
        raise ValueError(f"BDT score must be Double[], found {tree[SCORE_BRANCH].typename}")


def run(input_path, output_path, classifier_directory, *, sample_kind="simulation",
        weight_branch="shiftSamplingGenWeight", chunk_events=1000, receipt_path=None):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    output_path = Path(output_path)
    partial = Path(str(output_path) + ".incomplete")
    receipt_path = Path(receipt_path) if receipt_path else Path(str(output_path) + ".json")
    for path in (output_path, partial, receipt_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite BDT diagnostics: {path}")
    if sample_kind not in ("simulation", "data"):
        raise ValueError("sample_kind must be simulation or data")
    if chunk_events < 1:
        raise ValueError("chunk_events must be positive")

    labels, labels_path = (load_labels(classifier_directory) if sample_kind == "simulation" else (None, None))
    required = list(RECO_BRANCHES)
    if labels is not None:
        required.extend(labels.TRUTH_BRANCHES)
        required.append(weight_branch)
    receipt = {
        "schema": "shift-dimuon-bdt-histograms-v1", "complete": False,
        "input": str(input_path), "output": str(output_path),
        "sample_kind": sample_kind, "score_branch": SCORE_BRANCH,
        "score_selection_applied": False,
        "weights_branch": weight_branch if labels is not None else None,
        "weight_application_count": 1 if labels is not None else 0,
        "labels_source": str(labels_path) if labels_path else None,
        "labels_sha256": digest(labels_path) if labels_path else None,
        "events": 0, "pairs": 0, "category_pairs": {category: 0 for category in CATEGORIES},
        "label_pairs": {label: 0 for label in LABEL_SUFFIXES[1:]},
        "pair_sumw": 0.0, "pair_sumw2": 0.0,
        "event_sumw": 0.0, "event_sumw2": 0.0,
        "nonfinite_map_coordinates": {axis: 0 for axis in MAP_AXES},
    }
    root_file = None
    try:
        with uproot.open(input_path) as source:
            tree = source["Events"]
            validate_score_branch(tree)  # Must fail even if every event has zero pairs.
            missing = set(required) - set(tree.keys())
            if missing:
                raise ValueError("Missing BDT diagnostic branches: " + ", ".join(sorted(missing)))
            metadata_key = SCORE_BRANCH + "Metadata"
            if metadata_key in source:
                receipt["saved_score_metadata"] = str(source[metadata_key])

            output_path.parent.mkdir(parents=True, exist_ok=True)
            root_file = ROOT.TFile.Open(str(partial), "CREATE")
            if not root_file or root_file.IsZombie():
                raise RuntimeError(f"Cannot create BDT histogram file: {partial}")
            root_file.mkdir(DIRECTORY).cd()
            histograms = {}
            maps = {}
            for category in CATEGORIES:
                for label in LABEL_SUFFIXES if labels is not None else ("",):
                    for unweighted in (False, True):
                        name = histogram_name(category, label, unweighted)
                        histogram = ROOT.TH1D(name, name, *SCORE_BINS)
                        histogram.Sumw2()
                        histograms[category, label, unweighted] = histogram
                for axis, (bins, low, high, title) in MAP_AXES.items():
                    name = histogram_name(category) + "Vs" + axis
                    histogram = ROOT.TH2D(name, name, bins, low, high, *SCORE_BINS)
                    histogram.GetXaxis().SetTitle(title)
                    histogram.GetYaxis().SetTitle("dimuon vertex BDT score")
                    histogram.Sumw2()
                    maps[category, axis] = histogram

            pair_fields = [key for key in RECO_BRANCHES if key.startswith("ShiftDimuonVertex_")]
            for arrays in tree.iterate(required, step_size=chunk_events, library="ak", how=dict):
                for event_index in range(len(arrays["nShiftDimuonVertex"])):
                    count = int(arrays["nShiftDimuonVertex"][event_index])
                    n_muons = int(arrays["nShiftMuon"][event_index])
                    if count < 0 or n_muons < 0:
                        raise ValueError("Negative collection length")
                    for field in pair_fields:
                        if len(arrays[field][event_index]) != count:
                            raise ValueError(f"Pair/score collection length differs: {field}")
                    weight = float(arrays[weight_branch][event_index]) if labels is not None else 1.0
                    if not math.isfinite(weight):
                        raise ValueError("Nonfinite configured event weight")
                    receipt["events"] += 1
                    receipt["event_sumw"] += weight
                    receipt["event_sumw2"] += weight * weight
                    truth = ({key: arrays[key][event_index] for key in labels.TRUTH_BRANCHES}
                             if labels is not None else None)
                    if truth is not None:
                        for key in labels.TRUTH_BRANCHES:
                            expected = n_muons if key.startswith("ShiftMuon_") else len(truth["GenPart_pdgId"])
                            if len(truth[key]) != expected:
                                raise ValueError(f"MC label collection length differs: {key}")
                    for pair_index in range(count):
                        pair = {key[len("ShiftDimuonVertex_"):]: arrays[key][event_index][pair_index]
                                for key in pair_fields}
                        score = float(pair["bdtScore"])
                        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                            raise ValueError("Saved BDT score must be finite and between zero and one")
                        first, second = int(pair["muonIdx1"]), int(pair["muonIdx2"])
                        if first == second or not 0 <= first < n_muons or not 0 <= second < n_muons:
                            raise ValueError("Invalid retained dimuon reference")
                        label = None
                        if labels is not None:
                            value, reason = labels.pair_label(truth, first, second)
                            label = {1: "Genuine", 0: "Accidental", -1: "Unknown"}[value]
                            receipt["label_pairs"][label] += 1
                        receipt["pairs"] += 1
                        receipt["pair_sumw"] += weight
                        receipt["pair_sumw2"] += weight * weight
                        coordinates = {"Mass": float(pair["mass"]), "Vz": float(pair["vz"]),
                                       "VertexR": math.hypot(float(pair["vx"]), float(pair["vy"]))}
                        for axis, coordinate in coordinates.items():
                            if not math.isfinite(coordinate):
                                receipt["nonfinite_map_coordinates"][axis] += 1
                        for category in categories(pair):
                            receipt["category_pairs"][category] += 1
                            for suffix in ("", label) if label is not None else ("",):
                                histograms[category, suffix, False].Fill(score, weight)
                                histograms[category, suffix, True].Fill(score)
                            for axis, coordinate in coordinates.items():
                                if math.isfinite(coordinate):
                                    maps[category, axis].Fill(coordinate, score, weight)
            root_file.Write()
            root_file.Close()
            root_file = None

        with uproot.open(partial) as check:
            inclusive = check[f"{DIRECTORY}/{histogram_name()}"]
            raw = check[f"{DIRECTORY}/{histogram_name(unweighted=True)}"]
            if not math.isclose(float(np.sum(inclusive.values(flow=True))), receipt["pair_sumw"],
                                rel_tol=1e-11, abs_tol=1e-9):
                raise RuntimeError("Saved weighted BDT histogram differs from the retained pair sum")
            if int(round(float(np.sum(raw.values(flow=True))))) != receipt["pairs"]:
                raise RuntimeError("Saved raw BDT histogram differs from the retained pair count")
            if labels is not None:
                labelled = sum(check[f"{DIRECTORY}/{histogram_name(label=label)}"].values(flow=True)
                               for label in LABEL_SUFFIXES[1:])
                if not np.allclose(labelled, inclusive.values(flow=True), rtol=1e-11, atol=1e-9):
                    raise RuntimeError("MC score subsets do not partition the inclusive histogram")
        receipt["output_sha256"] = digest(partial)
        partial.rename(output_path)
        receipt["complete"] = True
        with receipt_path.open("x") as target:
            json.dump(receipt, target, indent=2, sort_keys=True)
            target.write("\n")
        return receipt
    except Exception as error:
        if root_file is not None:
            root_file.Close()
        receipt["error"] = str(error)
        if not receipt_path.exists():
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with receipt_path.open("x") as target:
                json.dump(receipt, target, indent=2, sort_keys=True)
                target.write("\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--classifier-directory", default=str(
        Path(__file__).resolve().parents[2] / "CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier"))
    parser.add_argument("--sample-kind", choices=("simulation", "data"), default="simulation")
    parser.add_argument("--weight-branch", default="shiftSamplingGenWeight")
    parser.add_argument("--chunk-events", type=int, default=1000)
    args = parser.parse_args()
    receipt = run(args.input, args.output, args.classifier_directory, sample_kind=args.sample_kind,
                  weight_branch=args.weight_branch, chunk_events=args.chunk_events, receipt_path=args.receipt)
    print(json.dumps({key: receipt[key] for key in ("complete", "events", "pairs", "output")}, sort_keys=True))


if __name__ == "__main__":
    main()
