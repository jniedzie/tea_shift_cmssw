# Code style: AI-code. AI-generated BDT diagnostic utility.
"""Validate ROOT histogram inventories for BDT score and event-mixing plots.

Per-bin cross sections and full parent GEN weights define process mixtures.
Raw counts and distinct physical signal points retain their own meaning.
This module handles input I/O; drawing belongs to the selected plot config.
"""

import hashlib
import json
import math
from pathlib import Path


_PROCESS_LABELS = {"qcd": "QCD", "dy": "DY", "jpsi": "J/#psi"}

def _plot_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def plot_input_rows(manifest):
    """Accept hand-written inputs and the histogram merger's frozen inventory."""
    samples, rows = manifest.get("samples"), manifest.get("rows")
    if samples is not None and rows is not None and samples != rows:
        raise ValueError("Conflicting samples and rows in BDT histogram input manifest")
    selected = samples if samples is not None else rows
    if not isinstance(selected, list) or not selected:
        raise ValueError("Empty or invalid BDT histogram input inventory")
    return selected

def readable_histogram_path(path):
    """Resolve CERN's equivalent EOS user mounts without changing provenance."""
    original = Path(path)
    if original.is_file():
        return original
    parts = original.parts
    alternate = None
    if len(parts) >= 5 and parts[:3] == ("/", "eos", "user") and len(parts[3]) == 1:
        alternate = Path("/eos", "home-" + parts[3], *parts[4:])
    elif len(parts) >= 4 and parts[:2] == ("/", "eos") and parts[2].startswith("home-") and len(parts[2]) == 6:
        alternate = Path("/eos/user", parts[2][-1], *parts[3:])
    if alternate is not None and alternate.is_file():
        return alternate
    raise FileNotFoundError(f"Cannot read BDT histogram input: {original}")

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

def prepare_process_histograms(manifest_path, output_directory, *, reuse=False):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "shift-dimuon-bdt-bin-inputs-v1" or manifest.get("complete") is not True:
        raise ValueError("Plot preparation requires a complete BDT bin histogram input manifest")
    rows = plot_input_rows(manifest)
    output_directory = Path(output_directory)
    if output_directory.exists() and reuse:
        receipt = json.loads((output_directory / "plot_inputs.json").read_text())
        if (receipt.get("schema") != "shift-dimuon-bdt-plot-inputs-v1" or
                receipt.get("complete") is not True or
                receipt.get("source_manifest_sha256") != _plot_digest(manifest_path)):
            raise ValueError("Existing process histograms belong to another or incomplete inventory")
        if len(receipt.get("inputs", [])) != len(rows):
            raise ValueError("Prepared process input inventory differs")
        for original, frozen in zip(rows, receipt["inputs"]):
            if any(frozen.get(key) != value for key, value in original.items()):
                raise ValueError("Prepared process input inventory differs")
            path = readable_histogram_path(original["histogram_path"])
            if _plot_digest(path) != frozen["histogram_sha256"]:
                raise ValueError("Source histogram checksum differs during reuse: " + str(path))
            if frozen["applied_shape_scale"] != bin_scale(original):
                raise ValueError("Prepared process normalization differs")
        for row in receipt["samples"]:
            path = readable_histogram_path(row["histogram_path"])
            if _plot_digest(path) != row["histogram_sha256"]:
                raise ValueError("Prepared process histogram checksum differs: " + str(path))
        return receipt
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite plot preparation: {output_directory}")
    output_directory.mkdir(parents=True)
    groups = {}
    receipt = {
        "schema": "shift-dimuon-bdt-plot-inputs-v1", "complete": False,
        "source_manifest": str(manifest_path), "source_manifest_sha256": _plot_digest(manifest_path),
        "normalization": "SM bins combined with cross_section_pb/full_gen_sumw; process shapes normalized only during plotting",
        "score_selection_applied": False, "samples": [], "inputs": [],
        "physics_valid": False, "signal_normalization_validated": False,
    }
    seen = set()
    try:
        for row in rows:
            identity = row["name"]
            if identity in seen:
                raise ValueError(f"Repeated BDT histogram input: {identity}")
            seen.add(identity)
            path = readable_histogram_path(row["histogram_path"])
            expected = row.get("histogram_sha256")
            actual = _plot_digest(path)
            if expected is not None and expected != actual:
                raise ValueError(f"BDT histogram checksum differs: {path}")
            process = row["process"]
            # Distinct BSM mass/lifetime samples remain distinct overlays.
            group = row.get("plot_group", identity if process == "signal" else process)
            if not isinstance(group, str) or not group or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in group):
                raise ValueError(f"Unsafe plot group: {group}")
            if process == "signal" and group != identity:
                raise ValueError("Distinct signal points must remain separate plot groups")
            scale = bin_scale(row)
            source = ROOT.TFile.Open(str(path), "READ")
            if not source or source.IsZombie() or source.TestBit(ROOT.TFile.kRecovered):
                raise RuntimeError(f"Cannot open BDT histogram input: {path}")
            directory = source.GetDirectory("dimuon_bdt")
            if not directory:
                source.Close()
                raise ValueError(f"Missing dimuon_bdt histogram directory: {path}")
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
                label = _PROCESS_LABELS.get(process, row.get("label", group)) if group == process else row.get("label", group)
                groups[group] = {"process": process, "label": label, "histograms": current}
            else:
                target = groups[group]
                if target["process"] != process or set(target["histograms"]) != set(current):
                    raise ValueError(f"Inconsistent process or histogram inventory within group: {group}")
                for name, histogram in current.items():
                    if not target["histograms"][name].Add(histogram):
                        raise ValueError(f"Incompatible BDT histogram binning: {group}/{name}")
            receipt["inputs"].append({**row, "histogram_read_path": str(path),
                                      "histogram_sha256": actual, "applied_shape_scale": scale, "plot_group": group})
        for group, values in groups.items():
            path = output_directory / f"{group}.root"
            file = ROOT.TFile.Open(str(path), "CREATE")
            file.mkdir("dimuon_bdt").cd()
            for histogram in values["histograms"].values():
                histogram.Write()
            file.Close()
            receipt["samples"].append({
                "name": group, "process": values["process"], "label": values["label"],
                "histogram_path": str(path.resolve()), "histogram_sha256": _plot_digest(path),
            })
        receipt["complete"] = True
    except Exception as error:
        receipt["error"] = str(error)
        (output_directory / "plot_inputs.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise
    (output_directory / "plot_inputs.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt

def configured_inventory(manifest_path, output, preparation_directory=None):
    """Resolve complete immutable inventories without discovering live inputs."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("complete") is not True:
        raise ValueError("Plot inventory must be complete")
    if manifest.get("schema") == "shift-dimuon-bdt-bin-inputs-v1":
        directory = preparation_directory or Path(output) / "process_shapes"
        return prepare_process_histograms(path, directory, reuse=True)
    if manifest.get("schema") not in (
        "shift-dimuon-bdt-plot-inputs-v1", "shift-event-mixing-plot-inputs-v1",
    ):
        raise ValueError("Unsupported immutable plot inventory schema")
    rows = plot_input_rows(manifest)
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Repeated immutable plot sample name")
    for row in rows:
        read_path = readable_histogram_path(row["histogram_path"])
        expected = row.get("histogram_sha256", row.get("sha256"))
        if row.get("histogram_sha256") and row.get("sha256") and row["histogram_sha256"] != row["sha256"]:
            raise ValueError("Conflicting prepared histogram checksums")
        if expected and _plot_digest(read_path) != expected:
            raise ValueError("Prepared histogram checksum differs: " + str(read_path))
    if manifest["schema"] == "shift-event-mixing-plot-inputs-v1":
        if (manifest.get("physics_valid") is not False or manifest.get("rate_validated") is not False or
                manifest.get("fit_reproduction_validated") is not True):
            raise ValueError("Event-mixing inventory must retain its unvalidated-rate scope")
        if _plot_digest(manifest["report"]) != manifest["report_sha256"]:
            raise ValueError("Event-mixing report checksum differs")
        if any(row.get("kind") not in ("model", "trials") for row in rows):
            raise ValueError("Unknown event-mixing histogram kind")
    return manifest
