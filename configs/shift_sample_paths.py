"""Select validated MC inputs without opening their event data."""
import ast
import json
import hashlib
import math
from collections import Counter
from pathlib import Path
import re

from shift_storage_paths import resolve_storage_path, tier_directory


def single_root_file(directory, *, allow_missing=False):
    """Resolve one ROOT file; optionally return None for unavailable inputs."""
    directory = Path(directory)
    try:
        if not directory.is_dir():
            if allow_missing:
                return None
            raise RuntimeError(f"ROOT input directory does not exist or is not accessible: '{directory}'")
        files = sorted(path for path in directory.iterdir() if path.suffix == ".root" and path.is_file())
    except OSError as exc:
        if allow_missing:
            return None
        raise RuntimeError(f"ROOT input directory is not accessible: '{directory}': {exc}") from exc
    if not files and allow_missing:
        return None
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
        recorded_output = record.get("output", "")
        resolved_output = resolve_storage_path(recorded_output) if recorded_output else ""
        if (record.get("status") != "validated" or
                Path(resolved_output) != path or
                not re.fullmatch(r"[0-9a-f]{64}", checksum)):
            raise RuntimeError(f"Invalid merged-sample provenance: {sidecar}")
        # A validated complete merge supersedes a legacy, possibly partial one.
        # Use its recorded content hash, not the current CMSSW checkout hash.
        candidates.append((1, path.stat().st_mtime_ns, path.name, path, checksum[:12]))
    if not candidates:
        raise RuntimeError(f"No versioned or validated complete merged ntuples found in '{directory}'")
    _, _, _, path, provenance = max(candidates)
    return str(path), provenance


def validated_nano_files(base_path, campaign, production_directory=None, *, allow_incomplete=False):
    """Return validated canonical Nano files, optionally freezing a partial inventory."""
    prefix = "shift_detector_representative_"
    if not campaign.startswith(prefix):
        raise RuntimeError(f"Unknown complete Nano campaign: {campaign}")
    root = Path(production_directory) if production_directory is not None else (
        Path(__file__).resolve().parents[2] / "validation" / ("ntuple_production_" + campaign[len(prefix):]))
    final_path = root / "production_complete.json"
    final = None
    if final_path.is_file():
        final = json.loads(final_path.read_text())
    elif not allow_incomplete:
        status_path = root / 'live_status.json'
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get('health') == 'failed' and status.get('terminal'):
                raise RuntimeError(
                    f"Nano production stopped with failures: {status.get('nano_jobs_done', 0)}/"
                    f"{status.get('nano_jobs_expected', '?')} Nano jobs completed; "
                    f"{len(status.get('failed_nano_jobs', []))} failed. "
                    "Repair the failed jobs, or use --allow-incomplete to merge the validated subset."
                )
        raise RuntimeError(
            f"Nano production is not marked complete: {final_path} is missing. "
            "Wait for the production controller to finish before histogramming all files."
        )
    manifest = json.loads((root / "manifest.json").read_text())
    sampling_path = root / "sampling_plan.json"
    if manifest.get("detector_sampling", {}).get("plan_sha256") != hashlib.sha256(sampling_path.read_bytes()).hexdigest():
        raise RuntimeError(f"Sampling plan does not match the frozen production: {sampling_path}")
    tiers = ("GEN", "SIM", "DIGIHLT", "RECO", "NANO")
    if final is not None and (
            not final.get("complete") or final.get("jobs") != manifest["jobs"] or (
                final.get("events") != manifest["events"] or final.get("tier_events") != {
                    name: {tier: count for tier in tiers} for name, count in manifest["strata"].items()}
            )):
        raise RuntimeError(f"Final Nano completion counts do not match all process/bin tiers: {final_path}")
    rows = [tuple(map(int, line.split())) for line in (root / "all_jobs.txt").read_text().splitlines()]
    if len(rows) != manifest["jobs"] or any(len(row) != 4 for row in rows) or (
        {row[3] for row in rows} != set(range(manifest["jobs"]))
    ):
        raise RuntimeError("Nano job inventory does not contain every unique planned job")
    sources = {source["index"]: source for source in manifest["sources"]}
    totals = Counter()
    files = []
    skipped = Counter()
    for source_index, skip, events, job in sorted(rows, key=lambda value: value[3]):
        source = sources[source_index]
        stratum = source["stratum"]
        receipt_path = root / "results" / f"status{job}.json"
        if not receipt_path.is_file():
            receipt_path = root / "results" / f"g{job // 500}" / f"status{job}.json"
        if not receipt_path.is_file():
            if allow_incomplete:
                skipped['missing receipts'] += 1
                continue
            raise RuntimeError(f"Validated Nano receipt is missing: {receipt_path}; check production completion")
        row = json.loads(receipt_path.read_text())
        if row.get('job') != job or row.get('events') != events or row.get('source_stratum') != stratum:
            raise RuntimeError(f"Unvalidated Nano receipt: {receipt_path}; job identity differs from the production plan")
        if allow_incomplete and final is None and row.get('complete') is False:
            skipped['unfinished or failed receipts'] += 1
            continue
        nano = Path(manifest["eos_output"]) / stratum / f"job{job:07d}" / "nano.root"
        if stratum not in manifest["strata"] or events <= 0 or skip < 0 or skip + events > source["events"] or (
            row.get("job") != job or row.get("events") != events or row.get("source_stratum") != stratum
            or row.get("nano_path") != str(nano) or row.get("exit_code") != 0 or row.get("complete") is not True
            or row.get("validated_tier_events") != {tier: events for tier in tiers}
            or not isinstance(row.get("nano_bytes"), int) or row["nano_bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("report_sha256") or "")
        ):
            raise RuntimeError(f"Unvalidated Nano receipt: {receipt_path}; check production completion")
        totals[stratum] += events
        process, bin_name = stratum.split("_", 1)
        # Validate the immutable receipt above before following its relocation.
        input_path = resolve_storage_path(
            Path(base_path) / "ntuple_production" / campaign / stratum / f"job{job:07d}" / "nano.root")
        output_path = tier_directory(base_path, process, bin_name, campaign, "histograms") / f"histograms_job{job:07d}.root"
        files.append((str(input_path), "", str(output_path)))
    if ((final is not None or not allow_incomplete) and dict(totals) != manifest["strata"]) or \
            len({row[2] for row in files}) != len(files):
        raise RuntimeError("Canonical Nano per-bin event counts or histogram outputs differ")
    # Parent GEN normalization remains in the existing sampling plan and event
    # ledger. These selected Nano Runs/cutflow counts never replace that exposure.
    if skipped:
        from Logger import warn
        warn(f"Nano production incomplete: {len(files)}/{manifest['jobs']} jobs ready.")
    return files


_REPRESENTATIVE_PREFIX = "shift_detector_representative_"


def _read_normalization_record(path, expected_digest=None):
    path = Path(path)
    payload = path.read_bytes()
    if expected_digest is not None and hashlib.sha256(payload).hexdigest() != expected_digest:
        raise RuntimeError(f"Full GEN normalization provenance hash differs: {path}")
    return json.loads(payload)


def full_gen_weight_sums(campaign, production_directory=None):
    """Return per-stratum parent GEN sums, or None for a legacy campaign.

    Recognized representative campaigns fail closed if their frozen sampling
    plan or original normalization authorities cannot be verified. Incomplete
    detector processing never changes these denominators.
    """
    if not campaign.startswith(_REPRESENTATIVE_PREFIX):
        return None
    if not re.fullmatch(r"shift_detector_representative_[0-9]{8}_v[0-9]+", campaign):
        raise RuntimeError(f"Invalid representative campaign: {campaign}")
    root = Path(production_directory) if production_directory is not None else (
        Path(__file__).resolve().parents[2] / "validation" /
        ("ntuple_production_" + campaign[len(_REPRESENTATIVE_PREFIX):]))
    try:
        manifest = _read_normalization_record(root / "manifest.json")
        if (manifest["schema"] != "shift-gen-to-nano-plan-v1" or
                Path(manifest["eos_output"]).name != campaign):
            raise RuntimeError("Detector manifest does not identify the requested campaign")
        sampling = manifest["detector_sampling"]
        if sampling["plan"] != "sampling_plan.json":
            raise RuntimeError("Detector manifest does not identify the canonical sampling plan")
        plan = _read_normalization_record(root / "sampling_plan.json", sampling["plan_sha256"])
        if (plan["schema"] != "shift-detector-pps-sample-v1" or
                plan["parent_manifest_sha256"] != sampling["parent_manifest_sha256"]):
            raise RuntimeError("Sampling plan does not identify the frozen full GEN parent")
        parent = _read_normalization_record(plan["parent_manifest"], plan["parent_manifest_sha256"])
        strata = plan["strata"]
        if (parent["schema"] != "shift-gen-to-nano-plan-v1" or
                set(strata) != set(manifest["strata"]) or set(strata) != set(parent["strata"])):
            raise RuntimeError("Full GEN and selected detector stratum inventories differ")

        authorities = plan["normalization_authorities"]
        for kind in ("ordinary", "weighted"):
            authority = authorities[kind]
            _read_normalization_record(authority["manifest"], authority["manifest_sha256"])
            if (authority["manifest"] != parent[kind + "_manifest"] or
                    authority["manifest_sha256"] != parent[kind + "_manifest_sha256"]):
                raise RuntimeError(f"The {kind} GEN authority differs from the full parent")
        weighted = authorities["weighted"]
        accounting = _read_normalization_record(weighted["accounting"], weighted["accounting_sha256"])
        if (accounting["schema"] != "shift-weighted-high-bin-final-accounting-v1" or
                accounting["complete"] is not True or
                accounting["manifest_sha256"] != weighted["manifest_sha256"] or
                accounting["strata"] != weighted["strata"]):
            raise RuntimeError("Weighted full GEN accounting differs from the sampling authority")
        if not set(accounting["strata"]).issubset(strata):
            raise RuntimeError("Weighted GEN authority contains an unknown stratum")

        result = {}
        for name, row in strata.items():
            exposure = row["original_sumw"]
            events = row["original_events"]
            if (isinstance(exposure, bool) or not isinstance(exposure, (int, float)) or
                    not math.isfinite(exposure) or exposure <= 0 or
                    isinstance(events, bool) or not isinstance(events, int) or events <= 0 or
                    events != parent["strata"][name] or
                    row["realized_events"] != manifest["strata"][name]):
                raise RuntimeError(f"Invalid full GEN exposure or event inventory for {name}")
            records = row["original_normalization_records"]
            if (not records or any(record["stratum"] != name for record in records) or
                    sum(record["events"] for record in records) != events):
                raise RuntimeError(f"Full GEN source accounting differs for {name}")
            if name in accounting["strata"]:
                original = accounting["strata"][name]
                if (original["accepted"] != events or
                        not math.isclose(exposure, original["sumw"], rel_tol=1e-12, abs_tol=0.)):
                    raise RuntimeError(f"Weighted full GEN sum differs for {name}")
            elif (exposure != events or
                    any(record["normalization_record"] is not None for record in records)):
                raise RuntimeError(f"Unit-weight full GEN exposure differs for {name}")
            result[name] = float(exposure)
        return result
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"Cannot verify full GEN plot normalization for {campaign}: {error}") from error


def load_nano_merge_groups(inventory_path, base_path):
    """Read exact same-sample input groups without discovering extra files."""
    inventory_path = Path(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    if not isinstance(inventory, dict):
        raise ValueError("Nano merge inventory must be a JSON object")
    if inventory.get("schema") == "shift-nano-merge-inputs-v1":
        groups = inventory.get("merge_input_groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError("Nano merge inventory contains no input groups")
        result = {}
        for destination, inputs in groups.items():
            if (not isinstance(destination, str) or not destination or
                    not isinstance(inputs, list) or not inputs or
                    any(not isinstance(path, str) or not path for path in inputs)):
                raise ValueError("Invalid exact Nano merge group")
            result[destination] = [resolve_storage_path(path) for path in inputs]
        return result
    if not {"campaign", "bins", "inputs_sha256"} <= set(inventory):
        raise ValueError("Unsupported Nano merge inventory")
    # Retained production inventories bind their preceding-step input tuples.
    # Their original logical paths own the process/bin identity after migration.
    payload = (inventory_path.parent / "histogram_inputs.txt").read_bytes()
    if hashlib.sha256(payload).hexdigest() != inventory["inputs_sha256"]:
        raise ValueError("Nano merge input inventory checksum differs")
    result, counts = {}, Counter()
    for line in payload.decode().splitlines():
        entry = ast.literal_eval(line)
        if (not isinstance(entry, (tuple, list)) or len(entry) != 3 or
                any(not isinstance(value, str) for value in entry)):
            raise ValueError("Invalid preceding-step Nano input tuple")
        nano = Path(entry[0])
        stratum = nano.parent.parent.name
        process, bin_name = stratum.split("_", 1)
        destination = str(tier_directory(base_path, process, bin_name,
                                         inventory["campaign"], "nanoAOD_merged"))
        result.setdefault(destination, []).append(resolve_storage_path(nano))
        counts[stratum] += 1
    if dict(counts) != inventory["bins"] or not result:
        raise ValueError("Nano merge groups differ from the frozen per-bin inventory")
    return result
