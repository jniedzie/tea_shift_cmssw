#!/usr/bin/env python3
# Code style: AI-code. AI-generated blinded aggregate preparation.
"""Prepare the current control plots from exact complete-lumisection inventories.

This is an offline publication check. The plot renderer consumes its aggregate
JSON and does not need the private inputs or production receipts.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import json
import multiprocessing
import os
from pathlib import Path
import tempfile

import publish_shift_blinded_data as publisher
import shift_blinded_data_summary as blind
from plot_shift_data_control_region import render_control_region


TOPOLOGY = ("ShiftDimuonVertex_topologyMin", "ShiftDimuonVertex_topologyMax")
IDENTITY = ("run", "luminosityBlock", "event")
require = publisher.require
load = publisher.load


def scope(mask):
    return {
        (int(run), section)
        for run, ranges in mask.items()
        for first, last in ranges
        for section in range(first, last + 1)
    }


def make_qualification(policy, code, manifest, authority):
    return {
        "schema": "shift-data-blinding-qualification-v2",
        "purpose": "jpsi-control-commissioning",
        "approved_mass_min_gev": policy["mass_min_gev"],
        "approved_mass_max_gev": policy["mass_max_gev"],
        "policy_sha256": blind.digest_json(policy),
        "executable_sha256": code["reader_sha256"],
        "input_provenance_sha256": blind.digest_json(manifest),
        "mc_as_data_passed": True,
        "independent_review_passed": True,
        "authorization": publisher.pin(authority["authorization"]),
        "mc_response_evidence": publisher.pin(authority["mc_response"]),
    }


def read_publication(receipt_path, inputs, policy):
    receipt_path = Path(receipt_path)
    receipt = load(receipt_path)
    summary_path = receipt_path.with_name("blinded_summary.json")
    manifest = load(receipt_path.with_name("provenance.json"))
    qualified = load(receipt_path.with_name("qualification.json"))
    summary = load(summary_path)
    blind.validate_summary(summary)
    require(receipt["complete"] and receipt["output_readback_verified"]
            and summary["policy"] == policy
            and receipt["processed_identity_sha256"] == summary["processed_identity_sha256"]
            and receipt["allowed_pairs"] == summary["allowed_pairs"]
            and publisher.sha256(summary_path) == receipt["outputs"]["blinded_summary.json"]["sha256"]
            and blind.digest_json(manifest) == summary["input_provenance_sha256"],
            "The previously published approved summary changed")
    published = blind.validate_manifest(manifest, [item["path"] for item in manifest["inputs"]], "collision-data")
    require(all(inputs.get(path) == item for path, item in published.items()),
            "The current cohort does not contain its exact previously published inputs")
    return summary, published, qualified


def scan_counts(inventory, exposure, policy, qualification, manifest, previous_path):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    paths = sorted(row["path"] for row in manifest["inputs"])
    expected = blind.validate_manifest(manifest, paths, "collision-data")
    digest = blind.digest_json(manifest)
    all_pairs = blind.BlindedSummary(policy, "collision-data", qualification, digest)
    both_both = blind.BlindedSummary(policy, "collision-data", qualification, digest)
    prior = None
    if previous_path:
        prior_summary, prior_inputs, prior_qualification = read_publication(previous_path, expected, policy)
        prior = blind.BlindedSummary(policy, "collision-data", prior_qualification,
                                     prior_summary["input_provenance_sha256"])
    expected_scope = scope(exposure["closed_lumimask"])
    lower = policy["mass_min_gev"]
    upper = policy["mass_max_gev"]
    provenance = []
    for path in paths:
        original = expected[path]
        blind.verify_input_bytes(original)
        source = ROOT.TFile.Open(path, "READ")
        try:
            require(source and not source.IsZombie() and not source.TestBit(ROOT.TFile.kRecovered),
                    "A restricted input ROOT file is unreadable")
            require(source.GetUUID().AsString() == original["root_uuid"], "The input ROOT UUID changed")
            tree = source.Get("Events")
            require(tree and tree.InheritsFrom("TTree"), "A restricted input has no Events tree")
            for name, leaf_types in zip(IDENTITY, (("UInt_t",), ("UInt_t",), ("ULong64_t",))):
                blind.checked_leaf(tree, name, leaf_types)
            blind.checked_leaf(tree, blind.COUNT_BRANCH, ("UInt_t", "Int_t"))
            blind.checked_leaf(tree, blind.MASS_BRANCH, ("Float_t", "Double_t"), blind.COUNT_BRANCH)
            leaves = [blind.checked_leaf(tree, name, ("Int_t",), blind.COUNT_BRANCH) for name in TOPOLOGY]
            branches = [tree.GetBranch(name) for name in TOPOLOGY]
            tree.SetBranchStatus("*", 0)
            for name in (*IDENTITY, blind.COUNT_BRANCH, blind.MASS_BRANCH):
                tree.SetBranchStatus(name, 1)
            entries = int(tree.GetEntries())
            for entry in range(entries):
                require(tree.GetEntry(entry) > 0, "A restricted input entry could not be read")
                identity = tuple(int(getattr(tree, name)) for name in IDENTITY)
                require(identity[:2] in expected_scope and identity not in all_pairs.identities,
                        "Input ownership differs from the closed lumisections")
                count = int(getattr(tree, blind.COUNT_BRANCH))
                masses = [float(mass) for mass in getattr(tree, blind.MASS_BRANCH)]
                all_pairs.add_event(identity, count, masses)
                if prior is not None and path in prior_inputs:
                    prior.add_event(identity, count, masses)
                indices = [index for index, mass in enumerate(masses) if lower <= mass <= upper]
                selected = []
                if indices:
                    require(all(branch.GetEntry(entry, 1) > 0 for branch in branches)
                            and all(leaf.GetLen() == count for leaf in leaves),
                            "The approved candidates lack their matching topology values")
                    for index in indices:
                        if all(int(leaf.GetValue(index)) == 2 for leaf in leaves):
                            selected.append(masses[index])
                both_both.add_event(identity, len(selected), selected)
            provenance.append({
                "path": path, "root_uuid": original["root_uuid"], "entries": entries,
                "sha256": original["sha256"], "bytes": original["bytes"],
            })
        except blind.BlindingError:
            raise
        except Exception:
            raise blind.BlindingError("The restricted control-region scan failed") from None
        finally:
            if source:
                source.Close()
        blind.verify_input_bytes(original)
    if prior is not None:
        previous_provenance = [row for row in provenance if row["path"] in prior_inputs]
        require(prior.result(previous_provenance) == prior_summary,
                "The independent scan did not reproduce the previous ordinary publication")

    inclusive = all_pairs.result(provenance)
    category = both_both.result(provenance)
    require(inclusive["unique_processed_events"] == inventory["expected_unique_events"] == exposure["selected_events"]
            and inclusive["processed_identity_sha256"] == inventory["expected_identity_sha256"]
            == exposure["selected_identity_sha256"] == category["processed_identity_sha256"]
            and all(a >= b for a, b in zip(inclusive["mass_bin_counts"], category["mass_bin_counts"])),
            "The full control-region scan disagrees with its closed input ownership")
    return {
        "edges": inclusive["mass_bin_edges_gev"], "all_bins": inclusive["mass_bin_counts"],
        "both_both_bins": category["mass_bin_counts"],
    }


def scan_source(item, authority, policy, code):
    inventory = load(item["inventory"])
    exposure = load(item["exposure"])
    manifest, _ = publisher.validated_inputs(inventory)
    qualification = make_qualification(policy, code, manifest, authority)
    return scan_counts(inventory, exposure, policy, qualification, manifest, item.get("previous_publication"))


def build_aggregate(items, authority):
    import sys
    workspace = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(workspace / "shift_cmssw_workflow/scripts"))
    from audit_shift_data_luminosity import bril_coverage

    policy, code, _ = publisher.validate_authority(*(Path(authority[name]) for name in (
        "policy", "authorization", "mc_proof", "mc_response", "review")))
    occupied = set()
    by_selection = {}
    sources = []
    for item in items:
        selection = item["selection"]
        require(selection in ("CSC loose", "CSC tight"), "Unknown halo selection")
        exposure = load(item["exposure"])
        declared = exposure.get("halo_selection", "CSC-loose")
        require(declared == selection.replace(" ", "-"), "The halo selection does not match its sealed exposure")
        complete = scope(exposure["closed_lumimask"])
        require(complete and len(complete) == exposure["closed_lumisections"]
                and complete.isdisjoint(occupied) and exposure["mc_normalization_qualified"] is False,
                "Certified lumisections overlap or fail complete coverage")
        occupied.update(complete)
        bril = exposure["bril_csv"]
        csv_path = Path(bril["path"])
        require(publisher.sha256(csv_path) == bril["sha256"], "The exact recorded luminosity rows changed")
        recorded = bril_coverage(csv_path, complete)["recorded_pb_inverse"]
        require(recorded == exposure["recorded_luminosity_pb_inverse"], "Recorded luminosity differs from BRIL")
        sources.append((item, exposure, recorded))

    edges = None
    with ProcessPoolExecutor(max_workers=min(4, len(items)), mp_context=multiprocessing.get_context("spawn")) as pool:
        scans = [pool.submit(scan_source, item, authority, policy, code) for item in items]
        for (item, exposure, lumi), pending in zip(sources, scans):
            bins = pending.result()
            if edges is None:
                edges = bins["edges"]
            require(edges == bins["edges"], "The control-window binning differs across samples")
            selection = item["selection"]
            sample = by_selection.setdefault(selection, {
                "selection": selection, "streams": [], "closed_lumisections": 0,
                "recorded_luminosity_pb_inverse": Decimal(0),
                "all_bins": [0] * (len(edges) - 1), "both_both_bins": [0] * (len(edges) - 1),
            })
            for dataset in exposure["datasets"]:
                primary = dataset.split("/")[1]
                if primary not in sample["streams"]:
                    sample["streams"].append(primary)
            sample["closed_lumisections"] += exposure["closed_lumisections"]
            sample["recorded_luminosity_pb_inverse"] += Decimal(str(lumi))
            for name in ("all_bins", "both_both_bins"):
                sample[name] = [left + right for left, right in zip(sample[name], bins[name])]
            print(f"Verified {selection}: {exposure['closed_lumisections']} lumisections, "
                  f"{sum(bins['all_bins'])} All and {sum(bins['both_both_bins'])} Both-Both pairs", flush=True)
    for sample in by_selection.values():
        sample["recorded_luminosity_pb_inverse"] = float(sample["recorded_luminosity_pb_inverse"])
        sample["streams"].sort()
    return {"mass_bin_edges_gev": edges, "samples": list(by_selection.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        inputs = load(args.inputs)
        require(inputs, "Require at least one exact closed input scope")
        aggregate = build_aggregate(inputs, load(args.authority))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".pending-cr-", dir=args.output_dir) as temporary:
            staging = Path(temporary)
            paths = render_control_region(aggregate, staging)
            json_path = staging / "blinded_plot_counts.json"
            json_path.write_text(json.dumps(aggregate, sort_keys=True, indent=2, allow_nan=False) + "\n")
            for path in (*paths, json_path):
                path.replace(args.output_dir / path.name)
    except blind.BlindingError as error:
        raise SystemExit(f"Control-region publication failed: {error}") from None
    except Exception:
        raise SystemExit("Control-region publication failed during preparation or delivery.") from None


if __name__ == "__main__":
    main()
