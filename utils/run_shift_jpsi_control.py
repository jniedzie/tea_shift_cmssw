#!/usr/bin/env python3
# Code style: AI-code. AI-generated runner for the TEA J/psi control-region chain.
"""Run one sealed sample through SHIFT's TEA skimmer, histogrammer and merger."""

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from shift_jpsi_dimuon import jpsi_dimuon
from shift_streaming_histograms import file_record, merge, publish, root_file, sha256, stage, write_json


PROJECT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT / "configs"
MERGER = PROJECT / "tea/apps/examples/merge.py"


def run(command, log, timeout, libraries=None):
    env = None
    if libraries:
        env = os.environ.copy()
        files = ("libcore.so", "libextensions.so", "libhistogramming.so")
        env["LD_PRELOAD"] = ":".join(str(libraries / name) for name in files)
    with log.open("xb") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       timeout=timeout, cwd=log.parent, env=env)


def skim_counts(path, original_events, kind, weight):
    source = root_file(path)
    try:
        tree = source.Get("Events")
        seal = source.Get("ShiftJpsiSkimMetadata")
        if not tree or not seal:
            raise ValueError("The skimmer did not produce a sealed Events tree")
        metadata = json.loads(str(seal.GetString()))
        selected_events = int(tree.GetEntries())
        selected_pairs = metadata["selected_pairs"]
        window = (metadata["mass_min_gev"], metadata["mass_max_gev"])
        if (metadata["schema"] != "shift-jpsi-skim-v1" or metadata["endpoints"] != "inclusive"
                or window != jpsi_dimuon["mass"]
                or metadata["input_kind"] != kind or metadata["weight_branch"] != weight
                or metadata["original_events"] != original_events
                or metadata["selected_events"] != selected_events
                or not 0 <= selected_events <= original_events
                or selected_pairs < selected_events or (selected_events == 0 and selected_pairs != 0)):
            raise ValueError("Skim metadata disagrees with the exact original input or output tree")
        return selected_events, selected_pairs
    finally:
        source.Close()


def skim_input(index, row, kind, weight, binaries, work, libraries):
    source = row["path"]
    output = work / f"skim_{index:06d}.root"
    with tempfile.TemporaryDirectory(dir=work, prefix="input_") as scratch:
        if source.startswith("/eos/") or source.startswith("root://"):
            local = Path(scratch) / "nano.root"
            stage(source, local, row.get("sha256"))
            source = str(local)
        elif row.get("sha256") is not None and sha256(source) != row["sha256"]:
            raise ValueError("An original ROOT input differs from its sealed checksum")
        arguments = [str(binaries / "shift_skimmer"), "--config", str(CONFIGS / "shift_skimmer_config.py"),
                     "--input_path", source, "--output_trees_path", str(output), "--input_kind", kind]
        if weight:
            arguments.extend(["--weight_branch", weight])
        run(arguments, work / f"skim_{index:06d}.log", timeout=7200, libraries=libraries)
    return index, output


def attach_response_weight(path, authority):
    """Preserve the published W/(p*q) policy on an MC skim."""
    import ROOT
    probability = authority["probability"]
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("Invalid frozen MC response probability")
    expected_identity = authority.get("event_id")
    if probability != 1 and expected_identity is None:
        raise ValueError("A sampled MC response needs its original event identity")
    source = root_file(path, "UPDATE")
    try:
        tree = source.Get("Events")
        seal = json.loads(str(source.Get("ShiftJpsiSkimMetadata").GetString()))
        branch_name = "shiftResponseGenWeight"
        if (seal["weight_branch"] != "shiftSamplingGenWeight"
                or tree.GetBranch(branch_name) or (expected_identity and seal["original_events"] != 1)):
            raise ValueError("Input does not match the frozen MC response draw")
        weights = []
        for event in tree:
            if expected_identity is not None:
                identity = [int(getattr(event, key)) for key in ("run", "luminosityBlock", "event")]
                if identity != expected_identity:
                    raise ValueError("Skim event differs from its frozen MC response draw")
            weight = float(event.shiftSamplingGenWeight) / probability
            if not math.isfinite(weight):
                raise ValueError("The committed MC response weight is nonfinite")
            weights.append(weight)
        buffer = array("d", [0.0])
        branch = tree.Branch(branch_name, buffer, branch_name + "/D")
        for weight in weights:
            buffer[0] = weight
            if branch.Fill() < 0:
                raise ValueError("Could not append the MC response weight")
        source.cd()
        if tree.Write("", ROOT.TObject.kOverwrite) <= 0:
            raise ValueError("Could not persist the MC response weight")
        seal["weight_branch"] = branch_name
        if ROOT.TObjString(json.dumps(seal, allow_nan=False)).Write(
                "ShiftJpsiSkimMetadata", ROOT.TObject.kOverwrite) <= 0:
            raise ValueError("Could not persist the MC response weight seal")
    finally:
        source.Close()


def add_exposure(path, record, normalization):
    import ROOT
    source = root_file(path, "UPDATE")
    try:
        source.cd()
        ROOT.TObjString(json.dumps(record, sort_keys=True, allow_nan=False)).Write(
            "ShiftJpsiControlExposure", ROOT.TObject.kOverwrite)
        if normalization:
            ROOT.TObjString(json.dumps(normalization["value"], sort_keys=True, allow_nan=False)).Write(
                normalization["key"], ROOT.TObject.kOverwrite)
    finally:
        source.Close()


def verify_merge(path, expected_events, expected_pairs):
    source = root_file(path)
    try:
        flow = source.Get("rawEventsCutFlow")
        if not flow or float(flow.GetBinContent(1)) != expected_events:
            raise ValueError("Merged raw event cutflow differs from the skims")
        for suffix in ("", "Good", "Near-Both", "Near-Far", "Both-Both", "Both-Far", "Other"):
            name = f"dimuon/ShiftDimuonVertex{suffix}_logMass"
            if not source.Get(name):
                raise ValueError("The merged control histograms lack a required dimuon category")
        inclusive = source.Get("dimuon/ShiftDimuonVertex_mass")
        if not inclusive or inclusive.GetEntries() != expected_pairs:
            raise ValueError("The merged mass histogram disagrees with the exact skim pair count")
        both_both = source.Get("dimuon/ShiftDimuonVertexBoth-Both_mass")
        if not both_both or not 0 <= both_both.GetEntries() <= expected_pairs:
            raise ValueError("The merged Both-Both count disagrees with the inclusive control pairs")
        return int(both_both.GetEntries())
    finally:
        source.Close()


def validate_request(request):
    if request.get("schema") != "shift-jpsi-control-request-v1":
        raise ValueError("Require an exact SHIFT control-region sample request")
    kind = request["input_kind"]
    if kind not in ("collision-data", "mc"):
        raise ValueError("Declare the input kind of the complete source sample")
    paths = [row["path"] for row in request["inputs"]]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Require distinct exact input files")
    if (any(type(row["expected_events"]) is not int or row["expected_events"] < 0
            for row in request["inputs"])
            or sum(row["expected_events"] for row in request["inputs"]) != request["input_events"]):
        raise ValueError("The requested original input count must close exactly")
    weight = request["weight_branch"]
    if kind == "collision-data":
        lumi = request["recorded_luminosity_pb_inverse"]
        if (weight or not isinstance(lumi, (int, float)) or not math.isfinite(lumi) or lumi <= 0
                or not request["closed_lumisections"] or any("sha256" not in row for row in request["inputs"])):
            raise ValueError("Data require exact certified recorded exposure and SHA-pinned inputs")
    else:
        if not isinstance(weight, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", weight):
            raise ValueError("MC requires its original configured event weight")
        normalization = request["normalization"]
        value = normalization["value"]
        if normalization["key"] == "ShiftStreamingNormalization":
            denominator, cross_section = value["full_source_trials"], value["sigma_nd_pb"]
        elif normalization["key"] == "ShiftFullGenNormalization":
            denominator, cross_section = value["full_source_sumw"], value["cross_section_pb"]
        else:
            raise ValueError("MC requires a recognized full-source normalization record")
        if any(not math.isfinite(number) or number <= 0 for number in (denominator, cross_section)):
            raise ValueError("MC normalization must carry a positive full-source denominator and cross section")
    output = Path(request["output_directory"])
    if not output.is_absolute() or output.exists():
        raise ValueError("Require a new absolute per-sample control-region directory")
    return output


def process(request_path, binaries, work, workers, libraries):
    request = json.loads(request_path.read_text())
    output = validate_request(request)
    kind, weight = request["input_kind"], request["weight_branch"]
    source_weight = request.get("source_weight_branch", weight)
    if source_weight != weight and (
            kind != "mc" or source_weight != "shiftSamplingGenWeight" or weight != "shiftResponseGenWeight"
            or any("response_sampling" not in row or not row.get("sha256") for row in request["inputs"])):
        raise ValueError("The exact input pins and MC response sampling authority are required")
    work.mkdir(mode=0o700)
    print(f"{request['sample']}: skimming {len(request['inputs'])} exact input files", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        products = list(pool.map(lambda item: skim_input(
            *item, kind, source_weight, binaries, work, libraries), enumerate(request["inputs"])))
    skims = []
    for index, path in products:
        row = request["inputs"][index]
        if source_weight != weight:
            attach_response_weight(path, row["response_sampling"])
        skims.append((index, path, *skim_counts(path, row["expected_events"], kind, weight)))
    kept = [row for row in skims if row[2]]
    histogram_inputs = kept or skims[:1]
    print(f"{request['sample']}: filling histograms from {len(histogram_inputs)} skims", flush=True)

    def histogram(row):
        index, skim, _, _ = row
        path = work / f"hist_{index:06d}.root"
        config = work / f"hist_{index:06d}.py"
        config.write_text(
            "import shift_paths\n\n"
            f"shift_paths.histogram_input_path = {str(skim)!r}\n"
            f"shift_paths.histogram_output_path = {str(path)!r}\n\n"
            "from shift_histogrammer_config import *\n"
        )
        run([str(binaries / "shift_histogrammer"), "--config", str(config)],
            work / f"hist_{index:06d}.log", timeout=1800, libraries=libraries)
        return path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        histograms = list(pool.map(histogram, histogram_inputs))
    merged = work / "ntuple_0.root"
    merge(histograms, merged, MERGER, work)
    retained_events = sum(count for _, _, count, _ in kept)
    retained_pairs = sum(count for _, _, _, count in kept)
    both_both_pairs = verify_merge(merged, retained_events, retained_pairs)
    record = {
        "schema": "shift-jpsi-control-publication-v1", "sample": request["sample"],
        "input_kind": kind, "input_request_sha256": sha256(request_path),
        "source_input_files": len(skims), "input_events": request["input_events"],
        "selected_events": retained_events, "selected_pairs": retained_pairs, "weight_branch": weight,
        "selected_pairs_both_both": both_both_pairs,
        "recorded_luminosity_pb_inverse": request.get("recorded_luminosity_pb_inverse"),
        "closed_lumisections": request.get("closed_lumisections"),
        "halo_selection": request.get("halo_selection"),
        "absolute_mc_normalization_qualified_for_data": False,
    }
    add_exposure(merged, record, request.get("normalization"))
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    skim_files = []
    for _, path, _, _ in (kept or skims[:1]):
        publish(path, pending / "nanoAOD")
        skim_files.append(file_record(path, str(output / "nanoAOD" / path.name)))
    publish(merged, pending / "histograms_merged")
    histogram_output = file_record(merged, str(output / "histograms_merged" / merged.name))
    publication = work / "ready.json"
    write_json(publication, {**record, "skims": skim_files, "histogram": histogram_output})
    publish(publication, pending)
    pending.rename(output)
    print(f"{request['sample']}: {retained_events} selected events from {len(skims)} input files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--libraries", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    options = parser.parse_args()
    os.umask(0o077)
    process(options.request.resolve(), options.binaries.resolve(), options.work.resolve(), options.workers,
            options.libraries.resolve() if options.libraries else None)


if __name__ == "__main__":
    main()
