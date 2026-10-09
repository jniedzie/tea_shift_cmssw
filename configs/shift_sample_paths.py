"""Select validated MC inputs without opening their event data."""
import json
import hashlib
from collections import Counter
from pathlib import Path
import re


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
        input_path = Path(base_path) / "ntuple_production" / campaign / stratum / f"job{job:07d}" / "nano.root"
        output_path = Path(base_path) / process / f"{campaign}_{bin_name.replace('toinf', 'to-1')}" / "histograms" / f"histograms_job{job:07d}.root"
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
