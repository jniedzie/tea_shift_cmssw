#!/usr/bin/env python3
"""Combine bin score histograms with full GEN exposure before process overlays."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "configs"))
from shift_bdt_histograms import DIRECTORY


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bin_scale(row):
    """SM bin mixture uses sigma/sumw(parent GEN); signal shapes use unit scale."""
    process = row["process"]
    if process in ("jpsi", "qcd", "dy"):
        cross_section = float(row["cross_section_pb"])
        exposure = float(row["full_gen_sumw"])
        if not math.isfinite(cross_section) or cross_section <= 0 or not math.isfinite(exposure) or exposure <= 0:
            raise ValueError("SM shape combination requires positive cross section and full GEN exposure")
        return cross_section / exposure
    if process == "signal":
        return 1.0
    raise ValueError(f"Unknown process: {process}")


def prepare(manifest_path, output_directory):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "shift-dimuon-bdt-bin-inputs-v1" or manifest.get("complete") is not True:
        raise ValueError("Plot preparation requires a complete BDT bin histogram input manifest")
    rows = manifest["samples"]
    if not rows:
        raise ValueError("Empty BDT histogram input manifest")
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite plot preparation: {output_directory}")
    output_directory.mkdir(parents=True)
    groups = {}
    receipt = {
        "schema": "shift-dimuon-bdt-plot-inputs-v1", "complete": False,
        "source_manifest": str(manifest_path), "source_manifest_sha256": sha256(manifest_path),
        "normalization": "SM bins combined with cross_section_pb/full_gen_sumw; process shapes normalized only during plotting",
        "score_selection_applied": False, "samples": [], "inputs": [],
    }
    seen = set()
    try:
        for row in rows:
            identity = row["name"]
            if identity in seen:
                raise ValueError(f"Repeated BDT histogram input: {identity}")
            seen.add(identity)
            path = row["histogram_path"]
            expected = row.get("histogram_sha256")
            actual = sha256(path)
            if expected is not None and expected != actual:
                raise ValueError(f"BDT histogram checksum differs: {path}")
            process = row["process"]
            # Distinct BSM mass/lifetime samples remain distinct overlays.
            group = row.get("plot_group", identity if process == "signal" else process)
            if not isinstance(group, str) or not group or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in group):
                raise ValueError(f"Unsafe plot group: {group}")
            scale = bin_scale(row)
            source = ROOT.TFile.Open(str(path), "READ")
            if not source or source.IsZombie():
                raise RuntimeError(f"Cannot open BDT histogram input: {path}")
            directory = source.GetDirectory(DIRECTORY)
            if not directory:
                source.Close()
                raise ValueError(f"Missing {DIRECTORY} histogram directory: {path}")
            current = {}
            for key in directory.GetListOfKeys():
                histogram = key.ReadObj()
                if not histogram.InheritsFrom("TH1"):
                    continue
                name = histogram.GetName()
                histogram.SetDirectory(0)
                # Preserve unweighted counts as counts; all weighted score
                # spectra and coordinate maps receive the SM bin mixture scale.
                histogram.Scale(1.0 if name.endswith("Unweighted") else scale)
                current[name] = histogram
            source.Close()
            if not current:
                raise ValueError(f"No BDT histograms in: {path}")
            if group not in groups:
                groups[group] = {"process": process, "label": row.get("label", group), "histograms": current}
            else:
                target = groups[group]
                if target["process"] != process or set(target["histograms"]) != set(current):
                    raise ValueError(f"Inconsistent process or histogram inventory within group: {group}")
                for name, histogram in current.items():
                    if not target["histograms"][name].Add(histogram):
                        raise ValueError(f"Incompatible BDT histogram binning: {group}/{name}")
            receipt["inputs"].append({**row, "histogram_sha256": actual, "applied_shape_scale": scale, "plot_group": group})
        for group, values in groups.items():
            path = output_directory / f"{group}.root"
            file = ROOT.TFile.Open(str(path), "CREATE")
            file.mkdir(DIRECTORY).cd()
            for histogram in values["histograms"].values():
                histogram.Write()
            file.Close()
            receipt["samples"].append({
                "name": group, "process": values["process"], "label": values["label"],
                "histogram_path": str(path.resolve()), "histogram_sha256": sha256(path),
            })
        receipt["complete"] = True
    except Exception as error:
        receipt["error"] = str(error)
        (output_directory / "plot_inputs.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise
    (output_directory / "plot_inputs.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Complete shift-dimuon-bdt-bin-inputs-v1 JSON")
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    receipt = prepare(args.inputs, args.output_directory)
    print(json.dumps({"complete": receipt["complete"], "samples": len(receipt["samples"]),
                      "plot_inputs": str(Path(args.output_directory) / "plot_inputs.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
