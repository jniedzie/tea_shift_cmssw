# Code style: AI-code. AI-generated validation and provenance helper.
"""Find available plot inputs and audit production inputs when requested."""
import ast
import json
import hashlib
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
import re

from shift_storage_paths import _canonical_path, resolve_storage_path, tier_directory


def single_root_file(directory, *, allow_missing=False, cache_directory=None):
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
    source = files[0]
    if cache_directory is None:
        return str(source)
    # ROOT can fail on mounted EOS paths even when ordinary byte reads work.
    # Cache by content, verifying the copy before returning its local path.
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = Path(cache_directory)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{checksum}_{source.name}"
    if not target.is_file():
        with tempfile.TemporaryDirectory(dir=cache, prefix=".copy-") as temporary:
            copied = Path(temporary) / "input.root"
            shutil.copyfile(source, copied)
            if hashlib.sha256(copied.read_bytes()).hexdigest() != checksum:
                raise RuntimeError(f"ROOT input changed while being copied: {source}")
            copied.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
        raise RuntimeError(f"Cached ROOT input checksum differs: {target}")
    return str(target)


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
    """Return per-stratum parent GEN sums for offline normalization accounting.

    Recognized representative campaigns fail closed if their frozen sampling
    plan or original normalization authorities cannot be verified. Incomplete
    detector processing never changes these denominators. Ordinary plotting
    uses the available histogram files' cutFlow instead of this audit.
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


def validated_bsm_histogram_files(plan_path, histogram_directory=None):
    """Read every completed signal point; keep its rate unvalidated."""
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema") != "shift-dark-photon-grid-plan-v1" or not plan.get("prepared"):
        raise RuntimeError("Require a prepared dark-photon grid plan")
    points = plan.get("points", [])
    if not points or len({point["point"] for point in points}) != len(points):
        raise RuntimeError("Signal grid contains no points or duplicate names")
    result, identities, paths = [], set(), set()
    for point in points:
        directory = (Path(histogram_directory) / point["point"] if histogram_directory
                     else Path(point["detector_directory"]))
        report_path = directory / "histogram_report.json"
        report = json.loads(report_path.read_text())
        histogram = Path(resolve_storage_path(directory / "histograms.root"))
        contract = Path(point["signal_contract"])
        contract_payload = contract.read_bytes()
        if hashlib.sha256(contract_payload).hexdigest() != point["signal_contract_sha256"]:
            raise RuntimeError(f"Signal contract checksum differs: {point['point']}")
        model = json.loads(contract_payload)
        expected_events = point["events"]
        if (model.get("mass_gev") != point["mass_gev"] or
                model.get("epsilon") != point["epsilon"] or
                model.get("requested_events") != expected_events):
            raise RuntimeError(f"Signal model labels differ from the frozen contract: {point['point']}")
        event_ids = [tuple(event) for event in report.get("input_event_ids", [])]
        if (report.get("complete") is not True or point["detector_count"] != expected_events or
                len(event_ids) != expected_events or len(set(event_ids)) != expected_events or
                any(len(event) != 3 or event[0] != model.get("run_number") for event in event_ids) or
                identities.intersection(event_ids)):
            raise RuntimeError(f"Signal histogram coverage differs from the full GEN plan: {point['point']}")
        if (not histogram.is_file() or histogram in paths or
                histogram.stat().st_size != report.get("histogram_bytes") or
                hashlib.sha256(histogram.read_bytes()).hexdigest() != report.get("histogram_sha256")):
            raise RuntimeError(f"Signal histogram or contract checksum differs: {point['point']}")
        weight_sum = report.get("native_event_weight_sum")
        if not isinstance(weight_sum, (int, float)) or not math.isfinite(weight_sum):
            raise RuntimeError(f"Invalid native signal weight sum: {point['point']}")
        identities.update(event_ids)
        paths.add(histogram)
        result.append(dict(
            name=point["point"], histogram_path=str(histogram),
            mass_gev=point["mass_gev"], epsilon=point["epsilon"],
            mean_lab_flight_m=point.get("target_mean_lab_flight_m"),
            events=expected_events, initial_weight_sum=weight_sum,
            physics_valid=False, normalization_ready=False,
        ))
    return result


def _bsm_campaign_records(campaign, validation_directory):
    """Find the unique sealed production that owns this campaign name."""
    root = Path(validation_directory) if validation_directory is not None else (
        Path(__file__).resolve().parents[2] / "validation")
    schemas = {"shift-bdt-independent-signal-stage-v1",
               "shift-independent-signal-repartition-plan-v1"}
    candidates = []
    # Production evidence sits at the task root or one deployment below it.
    # Avoid recursive searches through frozen sources and detector scratch.
    for pattern in ("*/plan.json", "*/*/plan.json"):
        for path in sorted(root.glob(pattern)):
            try:
                plan = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(plan, dict) and plan.get("schema") in schemas and plan.get("campaign") == campaign:
                candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"Require one BSM production plan for campaign '{campaign}'; "
                           f"found {len(candidates)}")
    path = candidates[0]
    seal = _read_normalization_record(path.parent / "seal.json")
    if seal.get("campaign") != campaign:
        raise RuntimeError("BSM production seal identifies a different campaign")
    records = []
    for name in ("plan.json", "gen_inventory.json", "scoring_components.json"):
        source = path.parent / name
        checksum = seal.get("files", {}).get(str(source.resolve()))
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise RuntimeError(f"BSM production seal does not pin {source}")
        records.append(_read_normalization_record(source, checksum))
    return path.parent, *records, hashlib.sha256((path.parent / "seal.json").read_bytes()).hexdigest()


def _bsm_selector(values, label, *, allow_prompt=False):
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise RuntimeError(f"BSM {label} selector must be a list or None")
    for value in values:
        if allow_prompt and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"Invalid BSM {label} selection: {value!r}")
    return set(values)


def available_bsm_campaign_histograms(base_path, campaign, masses=None, mean_lab_flights=None,
                                      *, epsilons=None, cache_directory=None):
    """Find ready lifetime and epsilon samples without production validation files.

    The two selectors apply to their named sample families independently. Use
    an empty lifetime list for epsilon-only plots, or an empty epsilon list to
    keep only the existing prompt/lab-flight samples.
    """
    if campaign is None:
        return []
    if not isinstance(campaign, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", campaign):
        raise RuntimeError(f"Invalid BSM campaign: {campaign!r}")
    masses = _bsm_selector(masses, "mass")
    flights = _bsm_selector(mean_lab_flights, "mean lab flight", allow_prompt=True)
    epsilons = _bsm_selector(epsilons, "epsilon")
    if masses == set() or (flights == set() and epsilons == set()):
        return []
    from Logger import warn
    pattern = re.compile(r"darkphoton_(m([0-9]+(?:p[0-9]+)?)_(?:prompt|lab_([0-9]+(?:p[0-9]+)?)m|epsilon_([0-9]+(?:p[0-9]+)?(?:e[+-]?[0-9]+)?)))")
    points = []
    try:
        for directory in Path(base_path).iterdir():
            match = pattern.fullmatch(directory.name)
            if not match or not directory.is_dir():
                continue
            name = match.group(1)
            mass = float(match.group(2).replace("p", "."))
            flight = float(match.group(3).replace("p", ".")) if match.group(3) else None
            epsilon = float(match.group(4).replace("p", ".")) if match.group(4) else None
            if masses is not None and mass not in masses:
                continue
            if epsilon is None and flights is not None and flight not in flights:
                continue
            if epsilon is not None and epsilons is not None and epsilon not in epsilons:
                continue
            points.append((mass, flight, epsilon, name, directory))
    except OSError as error:
        warn(f"Cannot read BSM histogram directories in {base_path}: {error}. Continuing with available samples.")
        return []
    result = []
    for mass, flight, epsilon, name, directory in sorted(points, key=lambda point:
            (point[0], point[2] is not None, -1 if point[1] is None else point[1], point[2] or 0)):
        histogram_directory = directory / campaign / "histograms_merged"
        histogram = single_root_file(histogram_directory, allow_missing=True, cache_directory=cache_directory)
        if histogram is None:
            warn(f"Skipping BSM {name}: no histogram file is available in {histogram_directory}.")
            continue
        row = dict(name=name, histogram_path=histogram, mass_gev=mass, mean_lab_flight_m=flight)
        if epsilon is not None:
            row['epsilon'] = epsilon
        result.append(row)
    if masses is not None and flights is not None:
        present = {(point[0], point[1]) for point in points if point[2] is None}
        for mass, flight in sorted({(mass, flight) for mass in masses for flight in flights} - present,
                                  key=lambda point: (point[0], -1 if point[1] is None else point[1])):
            label = "prompt" if flight is None else f"mean lab flight {flight:g} m"
            warn(f"Skipping BSM mass {mass:g} GeV, {label}: no sample directory is available in {base_path}.")
    if masses is not None and epsilons is not None:
        present = {(point[0], point[2]) for point in points if point[2] is not None}
        for mass, epsilon in sorted({(mass, epsilon) for mass in masses for epsilon in epsilons} - present):
            warn(f"Skipping BSM mass {mass:g} GeV, epsilon {epsilon:g}: no sample directory is available in {base_path}.")
    return result


def _partial_bsm_plot_histogram(validation_directory, deployment, plan, row, parent,
                                plan_hash, seal_hash, cache_directory):
    """Use a prepared histogram of audited completed chunks, with its full GEN parent."""
    root = Path(validation_directory) if validation_directory is not None else (
        Path(__file__).resolve().parents[2] / "validation")
    candidates = []
    for path in root.glob(f"*/{row['point']}/point_receipt.json"):
        receipt = _read_normalization_record(path)
        if (receipt.get("schema") == "shift-bsm-plot-histogram-v1" and
                receipt.get("campaign") == plan["campaign"] and receipt.get("point") == row["point"]):
            candidates.append((path, receipt))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(f"Ambiguous prepared BSM plot histograms: {row['point']}")
    path, receipt = candidates[0]
    count, processed = row["events"], receipt.get("events")
    if (receipt.get("complete") is not True or receipt.get("partial") is not True or
            receipt.get("production_complete") is not False or receipt.get("selection_applied") is not False or
            receipt.get("plan_sha256") != plan_hash or receipt.get("seal_sha256") != seal_hash or
            receipt.get("parent_gen") != parent or receipt.get("full_gen_events") != count or
            type(processed) is not int or not 0 < processed < count or
            any(receipt.get(key) != row[key] for key in
                ("mass_gev", "epsilon", "proper_ctau_mm", "target_mean_lab_flight_m"))):
        raise RuntimeError(f"Prepared partial BSM histogram has unrelated provenance: {path}")
    expected = parent["event_ids"]
    sources, chunk_ids = [], set()
    for source in receipt.get("source_chunks", []):
        chunk_id = source.get("chunk")
        chunks = [chunk for chunk in row["chunks"] if chunk["chunk"] == chunk_id]
        if len(chunks) != 1 or chunk_id in chunk_ids:
            raise RuntimeError("Partial BSM histogram repeats or invents a detector chunk")
        chunk = chunks[0]
        origin = plan.get("retained_origins", {}).get(f"{row['point']}/{chunk_id}", dict(
            status_path=str(deployment / "statuses" / f"{row['point']}_{chunk_id:03d}.json"),
            plan_sha256=plan_hash, seal_sha256=seal_hash))
        if source.get("receipt_path") != origin["status_path"]:
            raise RuntimeError("Partial BSM histogram changed its source receipt authority")
        checksum = source.get("receipt_sha256")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise RuntimeError("Partial BSM histogram has no source receipt checksum")
        status = _read_normalization_record(source["receipt_path"], checksum)
        ids = expected[chunk["skip"]:chunk["skip"] + chunk["count"]]
        nano = status.get("published", {}).get("nanoAOD", {})
        if (status.get("complete") is not True or status.get("point") != row["point"] or
                status.get("chunk") != chunk_id or status.get("events") != chunk["count"] or
                status.get("expected_event_ids") != ids or status.get("native_sumw") != len(ids) or
                status.get("selection_applied") is not False or
                status.get("parent_gen_sha256") != parent["gen_sha256"] or
                status.get("plan_sha256") != origin["plan_sha256"] or
                status.get("seal_sha256") != origin["seal_sha256"] or
                source.get("nano_path") != chunk["nano"] or nano.get("path") != chunk["nano"] or
                source.get("nano_sha256") != nano.get("sha256") or
                status.get("nano_sha256") != nano.get("sha256")):
            raise RuntimeError("Partial BSM histogram includes an unvalidated detector chunk")
        chunk_ids.add(chunk_id)
        sources.extend(ids)
    included = {tuple(event) for event in sources}
    missing = [event for event in expected if tuple(event) not in included]
    if (len(sources) != processed or len(included) != processed or
            sources != receipt.get("input_event_ids") or receipt.get("missing_event_ids") != missing or
            [event for event in expected if tuple(event) in included] != sources or
            parent.get("events") != count or len(expected) != count or
            parent.get("native_sumw") != count or parent.get("native_sumw2") != count or
            receipt.get("native_event_weight_sum") != processed or
            receipt.get("native_event_weight_sum2") != processed):
        raise RuntimeError("Partial BSM histogram changed event coverage or native weights")
    grouped = Path(receipt["grouped_nano_path"])
    histogram = Path(receipt["histogram_path"])
    if (not grouped.is_relative_to(path.parent) or not histogram.is_relative_to(path.parent) or
            not grouped.is_file() or hashlib.sha256(grouped.read_bytes()).hexdigest() != receipt.get("grouped_nano_sha256") or
            histogram.name != "ntuple_0.root" or not histogram.is_file() or
            histogram.stat().st_size != receipt.get("histogram_bytes") or
            hashlib.sha256(histogram.read_bytes()).hexdigest() != receipt.get("histogram_sha256")):
        raise RuntimeError("Prepared partial BSM histogram or grouped Nano checksum differs")
    histogram_path = single_root_file(histogram.parent, cache_directory=cache_directory)
    from Logger import info
    info(f"Plotting BSM {row['point']} with {processed}/{count} processed events.")
    return dict(name=row["point"], histogram_path=histogram_path, mass_gev=row["mass_gev"],
        epsilon=row["epsilon"], mean_lab_flight_m=row["target_mean_lab_flight_m"],
        events=processed, generated_events=count, initial_weight_sum=parent["native_sumw"],
        physics_valid=False, normalization_ready=False, production_complete=False)


def validated_bsm_campaign_histograms(base_path, campaign, masses=None, mean_lab_flights=None,
                                      *, cache_directory=None, validation_directory=None):
    """Resolve available selected signal points from a sealed campaign.

    Flight values are the grid's estimated mean lab flights in metres, not c*tau.
    None selects every value; a None entry in the flight list selects prompt.
    Prepared histograms of completed subsets are accepted with the full GEN denominator.
    Points with no available histogram warn and are skipped.
    """
    if campaign is None:
        return []
    from Logger import warn
    if not isinstance(campaign, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", campaign):
        raise RuntimeError(f"Invalid BSM campaign: {campaign!r}")
    masses = _bsm_selector(masses, "mass")
    flights = _bsm_selector(mean_lab_flights, "mean lab flight", allow_prompt=True)
    if masses == set() or flights == set():
        return []
    root, plan, inventory, scoring, seal_hash = _bsm_campaign_records(campaign, validation_directory)
    model_hash = scoring.get("model_sha256")
    if not isinstance(model_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", model_hash):
        raise RuntimeError("BSM frozen scoring model has no valid SHA-256 digest")
    points = plan.get("points", [])
    names = [row["point"] for row in points]
    if (plan.get("prepared") is not True or not points or len(set(names)) != len(names) or
            inventory.get("schema") != "shift-independent-signal-GEN-inventory-v1" or
            inventory.get("complete") is not True or
            inventory.get("each_parent_ledger_counted_once") is not True or
            set(names) != set(inventory.get("points", {}))):
        raise RuntimeError("BSM campaign has an incomplete or duplicate point inventory")
    available = {(row["mass_gev"], row["target_mean_lab_flight_m"]) for row in points}
    if len(available) != len(points):
        raise RuntimeError("BSM campaign has duplicate mass/lifetime points")
    if ((masses is not None and masses - {mass for mass, _ in available}) or
            (flights is not None and flights - {flight for _, flight in available})):
        raise RuntimeError(f"Requested BSM mass/lifetime values are absent from campaign '{campaign}'")
    requested = {(mass, flight) for mass, flight in available
                 if (masses is None or mass in masses) and (flights is None or flight in flights)}
    missing = ({(mass, flight) for mass in masses for flight in flights} - available
               if masses is not None and flights is not None else set())
    if missing:
        raise RuntimeError(f"Requested BSM mass/lifetime points are absent from campaign '{campaign}': "
                           f"{sorted(missing, key=repr)}")
    plan_hash = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    repartitioned = plan["schema"] == "shift-independent-signal-repartition-plan-v1"
    result, identities, paths = [], set(), set()
    for row in points:
        if (row["mass_gev"], row["target_mean_lab_flight_m"]) not in requested:
            continue
        name, count = row["point"], row["events"]
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise RuntimeError(f"Invalid BSM point name: {name!r}")
        directory = Path(base_path) / ("darkphoton_" + name) / campaign
        prefix = (row["final_output_base"] if repartitioned else
                  str(Path(row["chunks"][0]["nano"]).parent.parent))
        # Compare local EOS aliases in the same canonical namespace.
        if _canonical_path(resolve_storage_path(directory)) != _canonical_path(resolve_storage_path(prefix)):
            raise RuntimeError(f"BSM point namespace differs from the selected campaign: {name}")
        report_path = root / "completed_points" / (name + ".json")
        required = (report_path, directory / "metadata/grouping/union.json",
                    directory / "metadata/bdt_grouped/complete.json",
                    directory / "histograms_merged/ntuple_0.root")
        missing_path = next((path for path in required if not path.is_file()), None)
        if missing_path is not None:
            partial = _partial_bsm_plot_histogram(validation_directory, root, plan, row,
                inventory["points"][name], plan_hash, seal_hash, cache_directory)
            if partial is not None:
                ids = {tuple(event) for event in inventory["points"][name]["event_ids"]}
                if identities.intersection(ids) or partial["histogram_path"] in paths:
                    raise RuntimeError("Partial BSM histogram shares another physical point's events or path")
                identities.update(ids)
                paths.add(partial["histogram_path"])
                result.append(partial)
                continue
            warn(f"Skipping unfinished BSM point {name}: required input is missing ({missing_path}). "
                 "Continuing with the available points.")
            continue
        report = _read_normalization_record(report_path)
        if report.get("complete") is not True:
            warn(f"Skipping unfinished BSM point {name}: production is not complete. "
                 "Continuing with the available points.")
            continue
        if (report.get("schema") != "shift-independent-signal-point-complete-v1" or
                report.get("point") != name or
                report.get("events") != count or report.get("seal_sha256") != seal_hash or
                (repartitioned and (report.get("plan_sha256") != plan_hash or
                                    report.get("final_output_base") != prefix))):
            raise RuntimeError(f"Selected BSM point has an invalid completion receipt: {name}")
        parent, union = inventory["points"][name], report["union"]
        scored = report["score_histograms"]
        marker = scored["marker"]
        if (union.get("complete") is not True or scored.get("complete") is not True or
                marker.get("complete") is not True):
            warn(f"Skipping unfinished BSM point {name}: grouping or histogramming is not complete. "
                 "Continuing with the available points.")
            continue
        expected = parent["event_ids"]
        ids = [tuple(event) for event in expected]
        if (type(count) is not int or count <= 0 or parent.get("events") != count or
                len(ids) != count or len(set(ids)) != count or identities.intersection(ids) or
                any(len(event) != 3 or any(type(value) is not int or value <= 0 for value in event)
                    or event[0] != row["run_number"] for event in ids) or
                any(parent.get(key) != row[key] for key in
                    ("mass_gev", "epsilon", "proper_ctau_mm", "run_number")) or
                union.get("schema") != "shift-independent-signal-union-v1" or
                union.get("point") != name or
                union.get("events") != count or union.get("event_ids") != expected or
                union.get("parent_gen") != parent or union.get("parent_gen_ledger_count") != 1 or
                union.get("histogramming_started_after_complete_grouping") is not True or
                (repartitioned and union.get("plan_sha256") != plan_hash) or
                union.get("seal_sha256") != seal_hash):
            raise RuntimeError(f"BSM full GEN population or model differs: {name}")
        for key in ("native_sumw", "native_sumw2"):
            value = union.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value != parent.get(key):
                raise RuntimeError(f"BSM native weights differ from the full GEN parent: {name}")
        if _read_normalization_record(directory / "metadata/grouping/union.json") != union:
            raise RuntimeError(f"BSM published grouping receipt differs: {name}")
        grouped = union["grouped"]
        grouped_hash = grouped.get("sha256")
        if (marker.get("schema") != "shift-merged-dimuon-bdt-complete-v1" or
                marker.get("events") != count or marker.get("selection_applied") is not False or
                marker.get("plan_sha256") != seal_hash or
                marker.get("model_sha256") != model_hash or
                not isinstance(grouped_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", grouped_hash) or
                grouped.get("path") != prefix + "/nanoAOD_merged_unscored/ntuple_0.root" or
                marker.get("source") != grouped["path"] or marker.get("source_sha256") != grouped_hash or
                _read_normalization_record(directory / "metadata/bdt_grouped/complete.json") != marker):
            raise RuntimeError(f"BSM histogram marker differs from the complete grouped source: {name}")
        published = marker["published"]["histograms"]
        if published.get("path") != prefix + "/histograms_merged/ntuple_0.root":
            raise RuntimeError(f"BSM histogram is outside the selected point namespace: {name}")
        histogram = single_root_file(directory / "histograms_merged", cache_directory=cache_directory)
        payload = Path(histogram)
        if (histogram in paths or payload.stat().st_size != published.get("bytes") or
                hashlib.sha256(payload.read_bytes()).hexdigest() != published.get("sha256")):
            raise RuntimeError(f"BSM histogram checksum differs: {name}")
        identities.update(ids)
        paths.add(histogram)
        result.append(dict(name=name, histogram_path=histogram, mass_gev=row["mass_gev"],
            epsilon=row["epsilon"], mean_lab_flight_m=row["target_mean_lab_flight_m"],
            events=count, initial_weight_sum=parent["native_sumw"],
            physics_valid=False, normalization_ready=False))
    return result
