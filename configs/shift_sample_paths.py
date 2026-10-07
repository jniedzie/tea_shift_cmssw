"""Select one merged MC artifact without opening its event data."""
import json
import hashlib
from collections import Counter
from pathlib import Path
import re


def single_root_file(directory):
    """Resolve exactly one ROOT file without opening its contents."""
    directory = Path(directory)
    if not directory.is_dir():
        raise RuntimeError(f"ROOT input directory does not exist or is not accessible: '{directory}'")
    files = sorted(path for path in directory.glob("*.root") if path.is_file())
    if len(files) != 1:
        names = ", ".join(path.name for path in files) or "none"
        raise RuntimeError(
            f"Expected exactly one ROOT file in '{directory}', found {len(files)}: {names}"
        )
    return str(files[0])


def latest_merged_sample(samples_dir):
    directory = Path(samples_dir)
    legacy_pattern = re.compile(r"ntuple_0_([0-9a-f]{7,40}(?:-dirty-[0-9a-f]{8})?)\.root")
    candidates = []
    for path in directory.glob("ntuple*.root"):
        match = legacy_pattern.fullmatch(path.name)
        if match:
            candidates.append((0, path.stat().st_mtime_ns, path.name, path, match.group(1)))
            continue
        if not path.name.startswith(("ntuple_sampling_complete_", "ntuple_complete_")):
            continue
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            raise RuntimeError(f"Merged sample is missing its validation record: {sidecar}")
        record = json.loads(sidecar.read_text())
        checksum = record.get("sha256", "")
        if (record.get("status") != "validated" or
                Path(record.get("output", "")) != path or
                not re.fullmatch(r"[0-9a-f]{64}", checksum)):
            raise RuntimeError(f"Invalid merged-sample provenance: {sidecar}")
        # A validated complete merge supersedes a legacy, possibly partial one.
        # Use its recorded content hash, not the current CMSSW checkout hash.
        candidates.append((1, path.stat().st_mtime_ns, path.name, path, checksum[:12]))
    if not candidates:
        raise RuntimeError(f"No versioned or validated complete merged ntuples found in '{directory}'")
    _, _, _, path, provenance = max(candidates)
    return str(path), provenance


def validated_nano_files(base_path, campaign, production_directory=None):
    """Return all complete canonical Nano files with the usual per-bin outputs."""
    prefix = "shift_detector_representative_"
    if not campaign.startswith(prefix):
        raise RuntimeError(f"Unknown complete Nano campaign: {campaign}")
    root = Path(production_directory) if production_directory is not None else (
        Path(__file__).resolve().parents[2] / "validation" / ("ntuple_production_" + campaign[len(prefix):]))
    manifest_path, sampling_path = root / "manifest.json", root / "sampling_plan.json"
    audit_path = root / "canonical_inventory_summary.json"
    inventory_path = audit_path.with_name("canonical_inventory.jsonl")
    if not audit_path.is_file():
        raise RuntimeError(f"Canonical Nano audit is not available: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if not audit.get("ready"):
        raise RuntimeError(f"Nano production is incomplete: {audit.get('complete_jobs', 0)}/"
                           f"{audit.get('expected_jobs', 0)} validated files; histogramming is not ready")
    checksum = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if audit.get("errors") or audit.get("manifest_sha256") != checksum(manifest_path) or (
        audit.get("sampling_plan_sha256") != checksum(sampling_path) or
        audit.get("inventory_sha256") != checksum(inventory_path)
    ):
        raise RuntimeError("Canonical Nano audit does not match the frozen production")
    manifest = json.loads(manifest_path.read_text())
    final = json.loads((root / "production_complete.json").read_text())
    tiers = ("GEN", "SIM", "DIGIHLT", "RECO", "NANO")
    if not final.get("complete") or final.get("jobs") != manifest["jobs"] or (
        final.get("events") != manifest["events"] or final.get("tier_events") != {
            name: {tier: count for tier in tiers} for name, count in manifest["strata"].items()}
    ):
        raise RuntimeError("Final Nano completion counts do not match all process/bin tiers")
    rows = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    if len(rows) != manifest["jobs"] or {row["job"] for row in rows} != set(range(manifest["jobs"])):
        raise RuntimeError("Canonical Nano inventory does not contain every unique planned job")
    totals = Counter()
    files = []
    for row in sorted(rows, key=lambda value: value["job"]):
        stratum, events, job = row["source_stratum"], row["events"], row["job"]
        nano = Path(manifest["eos_output"]) / stratum / f"job{job:07d}" / "nano.root"
        if stratum not in manifest["strata"] or row.get("nano_path") != str(nano) or (
            not row.get("complete") or row.get("validated_tier_events") != {tier: events for tier in tiers}
            or not all(row.get(flag) is True for flag in ("canonical_marker_verified", "payload_sizes_verified",
                                                         "semantic_worker_audits_verified"))
        ):
            raise RuntimeError(f"Unvalidated canonical Nano file: job {job}")
        totals[stratum] += events
        process, bin_name = stratum.split("_", 1)
        input_path = Path(base_path) / "ntuple_production" / campaign / stratum / f"job{job:07d}" / "nano.root"
        output_path = Path(base_path) / process / f"{campaign}_{bin_name.replace('toinf', 'to-1')}" / "histograms" / f"histograms_job{job:07d}.root"
        files.append((str(input_path), "", str(output_path)))
    if dict(totals) != manifest["strata"] or len({row[2] for row in files}) != len(files):
        raise RuntimeError("Canonical Nano per-bin event counts or histogram outputs differ")
    # Parent GEN normalization remains in the existing sampling plan and event
    # ledger. These selected Nano Runs/cutflow counts never replace that exposure.
    return files
