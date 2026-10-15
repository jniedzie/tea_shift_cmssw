#!/usr/bin/env python3
# Code style: AI-code. AI-generated event-counting inputs for provisional limits.
"""Prepare disjoint, unselected event yields for the TEA limits tools.

The first stored reconstructed dimuon determines the event's channel. Events
without a dimuon remain in the full GEN denominator. There are no mass, charge,
vertex-quality, classifier, truth or kinematic cuts added by this calculation.
The stored vertices retain the existing reconstruction definition, including
its opposite-sign requirement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from shift_baseline_sensitivity import digest


CATEGORIES = ["Near-Both", "Near-Far", "Both-Both", "Both-Far", "Other"]
TOPOLOGIES = {(0, 2): "Near-Both", (0, 3): "Near-Far",
              (2, 2): "Both-Both", (2, 3): "Both-Far"}


def validate_limit_input_inventory(inputs):
    """Check frozen file metadata and full-parent rate authorities before use."""
    def checked_json(reference):
        path = Path(reference["path"])
        if digest(path) != reference["sha256"]:
            raise ValueError("Limit input authority changed: " + str(path))
        return json.loads(path.read_text())

    authority_ref = inputs.get("normalization_authority")
    if not authority_ref:
        raise ValueError("Limits require the preserved full GEN background authority")
    authority = checked_json(authority_ref)
    if (authority.get("schema") != "shift-preserved-full-GEN-normalization-v1"
            or authority.get("complete") is not True
            or authority.get("selected_counts_cutflow_and_native_Nano_Runs_not_used_as_denominators") is not True):
        raise ValueError("Invalid full GEN background normalization authority")
    authority_directory = Path(authority_ref["path"]).parent
    for record in authority["records"].values():
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Normalization records must stay in their frozen bundle")
        checked_json(dict(path=str(authority_directory / relative), sha256=record["sha256"]))
    names = [sample["name"] for sample in inputs["backgrounds"] + inputs["signals"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate limit sample name")
    for sample in inputs["backgrounds"] + inputs["signals"]:
        metadata = sample.get("source_file_stats", [])
        if {item["path"] for item in metadata} != set(sample["files"]):
            raise ValueError("Every limit input requires frozen file metadata: " + sample["name"])
        for item in metadata:
            current = Path(item["path"]).stat()
            if (current.st_size, current.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                raise ValueError("Limit input changed since preparation: " + item["path"])
    if set(authority["strata"]) != {sample["name"] for sample in inputs["backgrounds"]}:
        raise ValueError("Background strata differ from the full GEN authority")
    for sample in inputs["backgrounds"]:
        declared = authority["strata"][sample["name"]]
        if (sample["weight_branch"] != "shiftSamplingGenWeight"
                or sample["full_gen_sumw"] != declared["sumw"]
                or sample["selected_cross_section_pb"] != declared["cross_section_pb"]):
            raise ValueError("Background sampling weight or full GEN rate differs: " + sample["name"])
    for sample in inputs["signals"]:
        rate = checked_json(sample["rate_reference"])
        audit = checked_json(sample["gen_audit"])
        union = checked_json(sample["grouping_reference"])
        marker = checked_json(sample["scored_group_reference"])
        if (sample["weight_branch"] != "genWeight" or sample.get("partial") is not False
                or sample.get("branching_included_in_sigma") is not True
                or sample.get("epsilon_correction_already_in_sigma") is not True):
            raise ValueError("Signal input convention differs: " + sample["name"])
        if (rate["point"] != sample["name"] or union["point"] != sample["name"]
                or rate.get("runtime_validated") is not True
                or audit.get("runtime_validated") is not True
                or union.get("complete") is not True or marker.get("complete") is not True
                or marker.get("selection_applied") is not False):
            raise ValueError("Signal authority is incomplete or selected: " + sample["name"])
        parent = rate["full_gen"]
        if (sample["full_gen_sumw"] != parent["sum_weights"]
                or sample["full_gen_sumw"] != union["native_sumw"]
                or sample["full_gen_sumw"] != audit["sum_weights"]
                or sample["expected_processed_events"] != parent["accepted_events"]
                or sample["expected_processed_events"] != union["events"]
                or sample["expected_processed_events"] != audit["events"]
                or sample["expected_processed_events"] != marker["events"]):
            raise ValueError("Signal lost its full parent denominator: " + sample["name"])
        physical_rates = rate["physical_selected_cross_sections_pb"]
        if len(physical_rates) != 1 or sample["selected_cross_section_pb"] != physical_rates[0]["value_pb"]:
            raise ValueError("Signal rate differs from its physical selected rate: " + sample["name"])
        identities = union["event_ids"]
        if (len(identities) != parent["accepted_events"]
                or len({tuple(row) for row in identities}) != len(identities)
                or {tuple(row) for row in identities} != {tuple(row) for row in audit["event_ids"]}):
            raise ValueError("Signal grouped identities differ from full GEN: " + sample["name"])
        identity_hash = hashlib.sha256(''.join("%d:%d:%d\n" % tuple(row) for row in identities).encode()).hexdigest()
        if sample["expected_event_ids_sha256"] != identity_hash:
            raise ValueError("Signal identity digest differs from grouping: " + sample["name"])
    return dict(background_strata=len(inputs["backgrounds"]), signal_points=len(inputs["signals"]),
                source_files=sum(len(sample["files"]) for sample in inputs["backgrounds"] + inputs["signals"]),
                signal_full_parent_events=sum(sample["expected_processed_events"] for sample in inputs["signals"]))


def event_category(minimum, maximum):
    if not 0 <= minimum <= maximum <= 4:
        raise ValueError("Invalid dimuon topology: " + str((minimum, maximum)))
    return TOPOLOGIES.get((minimum, maximum), "Other")


def count_sample(sample):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    branches = ["run", "luminosityBlock", "event", "nShiftDimuonVertex",
                "ShiftDimuonVertex_topologyMin", "ShiftDimuonVertex_topologyMax",
                sample["weight_branch"]]
    counts = dict(events=0, sumw=0., sumw2=0., no_dimuon_events=0,
                  multiple_dimuon_events=0, maximum_weight=0., sources=[],
                  categories={name: dict(events=0, sumw=0., sumw2=0.)
                              for name in CATEGORIES})
    identities = set()
    identity_hash = hashlib.sha256()
    for filename in sample["files"]:
        path = Path(filename)
        before = path.stat()
        root_file = ROOT.TFile(str(path), "READ")
        if not root_file or root_file.IsZombie() or root_file.TestBit(ROOT.TFile.kRecovered):
            raise RuntimeError("Unreadable or recovered Nano input: " + str(path))
        try:
            tree = root_file.Get("Events")
            if not tree:
                raise RuntimeError("Missing Events tree: " + str(path))
            tree.SetBranchStatus("*", 0)
            for branch in branches:
                if not tree.GetBranch(branch):
                    raise RuntimeError("Missing counting branch: " + branch)
                tree.SetBranchStatus(branch, 1)
            for event in tree:
                identity = (int(event.run), int(event.luminosityBlock), int(event.event))
                if identity in identities:
                    raise ValueError("Duplicate event within " + sample["name"] + ": " + str(identity))
                identities.add(identity)
                identity_hash.update(("%d:%d:%d\n" % identity).encode())
                weight = float(getattr(event, sample["weight_branch"]))
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError("Counting inputs require finite nonnegative weights")
                counts["events"] += 1
                counts["sumw"] += weight
                counts["sumw2"] += weight * weight
                counts["maximum_weight"] = max(counts["maximum_weight"], weight)
                pairs = int(event.nShiftDimuonVertex)
                if pairs == 0:
                    counts["no_dimuon_events"] += 1
                    continue
                if pairs > 1:
                    counts["multiple_dimuon_events"] += 1
                category = event_category(int(event.ShiftDimuonVertex_topologyMin[0]),
                                          int(event.ShiftDimuonVertex_topologyMax[0]))
                row = counts["categories"][category]
                row["events"] += 1
                row["sumw"] += weight
                row["sumw2"] += weight * weight
        finally:
            root_file.Close()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Input changed during counting: " + str(path))
        counts["sources"].append(dict(path=str(path), bytes=after.st_size,
                                      mtime_ns=after.st_mtime_ns))
    counts["event_ids_sha256"] = identity_hash.hexdigest()
    if sum(row["events"] for row in counts["categories"].values()) != counts["events"] - counts["no_dimuon_events"]:
        raise ValueError("Disjoint category accounting failed")
    expected = sample.get("expected_processed_events")
    if expected is not None and counts["events"] != expected:
        raise ValueError("Processed population changed: " + sample["name"])
    expected_hash = sample.get("expected_event_ids_sha256")
    if expected_hash is not None and counts["event_ids_sha256"] != expected_hash:
        raise ValueError("Processed identities changed: " + sample["name"])
    for row in counts["categories"].values():
        row["ess"] = row["sumw"] ** 2 / row["sumw2"] if row["sumw2"] else 0.
    return counts


def predict(sample, counts, luminosity_pb):
    denominator = float(sample["full_gen_sumw"])
    sigma = float(sample["selected_cross_section_pb"])
    if not math.isfinite(denominator + sigma) or denominator <= 0 or sigma <= 0:
        raise ValueError("Nonpositive full GEN denominator or cross section")
    scale = luminosity_pb * sigma / denominator
    return dict(yields={name: scale * row["sumw"] for name, row in counts["categories"].items()},
                mc_errors={name: scale * math.sqrt(row["sumw2"]) for name, row in counts["categories"].items()},
                counts=counts, full_gen_sumw=denominator,
                cross_section_pb=sigma, reference_cross_section_pb=sigma,
                partial=sample.get("partial", False), source=sample)


def prepare(inputs_path, output, luminosity_pb):
    inputs = json.loads(inputs_path.read_text())
    if not inputs["backgrounds"] or not inputs["signals"]:
        raise ValueError("Need explicit MC signal and background inventories")
    if not math.isfinite(luminosity_pb) or luminosity_pb <= 0:
        raise ValueError("Luminosity must be positive")
    validate_limit_input_inventory(inputs)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "event_counts.json"
    input_hash = digest(inputs_path)
    receipt = dict(inputs_sha256=input_hash, counts={})
    if checkpoint.exists():
        receipt = json.loads(checkpoint.read_text())
        if receipt["inputs_sha256"] != input_hash:
            raise ValueError("Existing counts belong to another inventory")
        for counts in receipt["counts"].values():
            for source in counts["sources"]:
                current = Path(source["path"]).stat()
                if (current.st_size, current.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
                    raise ValueError("Counted input changed: " + source["path"])
    for sample in inputs["backgrounds"] + inputs["signals"]:
        if sample["name"] not in receipt["counts"]:
            receipt["counts"][sample["name"]] = count_sample(sample)
            temporary = checkpoint.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            temporary.replace(checkpoint)
        row = receipt["counts"][sample["name"]]
        if row["event_ids_sha256"] != sample["expected_event_ids_sha256"]:
            raise ValueError("Count receipt identities differ from the inventory: " + sample["name"])
        if row["events"] != sample["expected_processed_events"]:
            raise ValueError("Count receipt population differs from the inventory: " + sample["name"])
        print(sample["name"], row["events"], "events;", row["events"] - row["no_dimuon_events"], "with dimuon", flush=True)
    report = dict(schema="shift-simplified-limit-yields-v1", luminosity_pb=luminosity_pb,
                  categories=CATEGORIES, backgrounds={}, signals={},
                  inputs=dict(path=str(inputs_path.resolve()), sha256=input_hash),
                  source=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
                  selection="First stored reconstructed dimuon per event; no added charge, quality, mass, kinematic, classifier or truth cut",
                  trigger_efficiency=1., scale_factors=1., halo_fraction=.01,
                  assumptions=["3 fb^-1 is assumed effective target exposure; halo fraction is not applied again",
                    "100% trigger efficiency is assumed; detector replay reconstruction remains in the MC acceptance",
                    "Stored dimuons retain the existing reconstruction definition, including its opposite-sign requirement",
                    "Separate 30% signal and background normalization nuisances, each correlated across categories",
                    "Nominal finite-MC yields only; no finite-template nuisance or tail correction",
                    "Pure-DY signal rates and source ownership remain provisional",
                    "Zero selected signal MC gives unresolved acceptance, not proof of zero physical response",
                    "No collision data; background-only Asimov expected limits"])
    for kind in ("backgrounds", "signals"):
        for sample in inputs[kind]:
            row = predict(sample, receipt["counts"][sample["name"]], luminosity_pb)
            if kind == "signals":
                row["sample_kind"] = "epsilon" if "_epsilon_" in sample["name"] else "lifetime"
                for key in ("mass_gev", "epsilon", "mean_lab_flight_m", "proper_ctau_mm"):
                    if key in sample:
                        row[key] = sample[key]
            report[kind][sample["name"]] = row
    report["background_totals"] = {name: sum(row["yields"][name] for row in report["backgrounds"].values())
                                    for name in CATEGORIES}
    (output / "yields.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(background_totals=report["background_totals"], signals=len(report["signals"]))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--luminosity-pb", type=float, default=3000.)
    args = parser.parse_args()
    prepare(args.inputs, args.output, args.luminosity_pb)


if __name__ == "__main__":
    main()
