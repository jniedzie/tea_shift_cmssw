# Code style: AI-code. AI-generated classifier dataset extraction and ownership audit.
"""Read explicitly inventoried MC Nano; keep reconstruction and truth separate.

This reader never draws admission, changes a pair selection, or supplies truth
to the reconstructed feature extractor. Source ownership is used only to prevent
shared GEN and source-bank responses from crossing training folds.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ownership_components(rows):
    """Join files sharing any declared original-GEN/source-bank ownership key."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    key_owner = {}
    for index, row in enumerate(rows):
        keys = row.get("source_group_keys", [])
        if not keys:
            raise ValueError("Missing original source ownership: " + str(row.get("path")))
        for key in keys:
            if not isinstance(key, str) or not key:
                raise ValueError("Invalid source ownership key")
            if key in key_owner:
                parent[find(index)] = find(key_owner[key])
            else:
                key_owner[key] = index
    component_keys = {}
    for index, row in enumerate(rows):
        component_keys.setdefault(find(index), set()).update(row["source_group_keys"])
    names = {root: hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
             for root, keys in component_keys.items()}
    return [names[find(index)] for index in range(len(rows))]


def read_nano(row, frozen_source, group_id):
    """Extract all retained pairs and strict labels from one immutable MC input."""
    import uproot

    source_dir = str(Path(frozen_source).resolve())
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from features import RECO_BRANCHES, extract_reco
    from labels import TRUTH_BRANCHES, extract_labels, label_diagnostics
    from shift_vertex_reco_information import BRANCHES, extract_extension
    from vertex_pointing import POINTING_FEATURE_NAMES, pointing_features
    from shift_classifier_labels import recover_pair_label

    if row.get("process") not in ("qcd", "dy", "jpsi", "signal"):
        raise ValueError("An explicitly classified simulation process is required")
    path = Path(row.get("nano_path", row.get("path", "")))
    stat = path.stat()
    signature = [stat.st_size, stat.st_mtime_ns]
    if row.get("bytes") is not None and stat.st_size != row["bytes"]:
        raise ValueError("Input size differs: " + str(path))
    expected_hash = row.get("sha256", row.get("nano_sha256"))
    if expected_hash and sha256(path) != expected_hash:
        raise ValueError("Input digest differs: " + str(path))
    weight_branch = row.get("weight_branch", "shiftSamplingGenWeight")
    fields = tuple(dict.fromkeys(RECO_BRANCHES + BRANCHES + TRUTH_BRANCHES + (
        "ShiftDimuonVertex_topologyMin", "ShiftDimuonVertex_topologyMax", weight_branch)))
    parts = []
    with uproot.open(path, timeout=90) as source:
        tree = source["Events"]
        missing = set(fields) - set(tree.keys())
        if missing:
            raise ValueError(str(path) + " missing: " + ", ".join(sorted(missing)))
        events = tree.num_entries
        if row.get("expected_events") is not None and events != row["expected_events"]:
            raise ValueError("Input event census differs: " + str(path))
        counts = tree["nShiftDimuonVertex"].array(library="np")
        parent_ids = np.column_stack([tree[key].array(library="np").astype(np.uint64)
                                     for key in ("run", "luminosityBlock", "event")])
        for start in range(0, events, 200):
            stop = min(start + 200, events)
            if not counts[start:stop].any():
                continue
            arrays = tree.arrays(fields, entry_start=start, entry_stop=stop,
                                 library="ak", how=dict)
            reco = extract_reco(arrays)
            features = pointing_features(np.column_stack((
                reco["X"], extract_extension(arrays, reco))))
            labels, reasons = extract_labels(arrays, reco)
            diagnostics = label_diagnostics(arrays, reco)
            original_labels = labels.copy()
            statuses, recovered, association_states, gen_states = [], [], [], []
            for index, (event_index, muons) in enumerate(zip(reco["event_index"], reco["muon_indices"])):
                truth = {key: arrays[key][event_index] for key in TRUTH_BRANCHES}
                label, reason, diagnosis = recover_pair_label(
                    truth, int(muons[0]), int(muons[1]), int(labels[index]), str(reasons[index]))
                labels[index], reasons[index] = label, reason
                statuses.append(diagnosis["status"])
                recovered.append(diagnosis["recovered"])
                association_states.append([leg["association"] for leg in diagnosis["legs"]])
                gen_states.append([leg["gen_state"] for leg in diagnosis["legs"]])
            categories, weights = [], []
            for event, pair in zip(reco["event_index"], reco["pair_index"]):
                topology = tuple(int(arrays["ShiftDimuonVertex_" + key][event][pair])
                                 for key in ("topologyMin", "topologyMax"))
                categories.append({(0, 2): "Near-Both", (0, 3): "Near-Far",
                                   (2, 2): "Both-Both", (2, 3): "Both-Far"}.get(topology, "Other"))
                weights.append(float(arrays[weight_branch][event]))
            n = len(labels)
            event_owner = [group_id + ":" + ":".join(map(str, identity))
                           for identity in reco["ids"]]
            part = dict(X=features, ids=reco["ids"], pair_index=reco["pair_index"],
                        muon_indices=reco["muon_indices"], y=labels,
                        original_y=original_labels, label_recovered=np.asarray(recovered),
                        label_status=np.asarray(statuses),
                        truth_leg_association=np.asarray(association_states),
                        truth_leg_gen_state=np.asarray(gen_states),
                        label_reason=reasons, category=np.asarray(categories),
                        weight=np.asarray(weights), mass=reco["nuisance"][:, 0],
                        process=np.full(n, row["process"]),
                        point=np.full(n, row.get("point", row.get("stratum", ""))),
                        domain=np.full(n, row.get("domain", "current")),
                        group_id=np.full(n, group_id), event_owner_id=np.asarray(event_owner),
                        nano_path=np.full(n, str(path)))
            for key, value in diagnostics.items():
                part["truth_" + key] = value
            parts.append(part)
        root_uuid = str(source.file.uuid)
    after = path.stat()
    if signature != [after.st_size, after.st_mtime_ns]:
        raise ValueError("Input changed during read: " + str(path))
    if parts:
        data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    else:
        data = None
    if int(counts.sum()) != (len(data["y"]) if data else 0):
        raise ValueError("Candidate extraction census differs")
    receipt = dict(path=str(path), input_sha256=expected_hash,
                   signature=signature, root_uuid=root_uuid, events=events,
                   pairs=int(counts.sum()), group_id=group_id, weight_branch=weight_branch,
                   parent_ids_sha256=hashlib.sha256(parent_ids.tobytes()).hexdigest(),
                   labels=dict(Counter(data["label_reason"])) if data else {})
    return data, receipt, np.asarray(POINTING_FEATURE_NAMES)


def xrootd_url(path):
    """Resolve the two CERN EOS instances explicitly; POSIX symlinks do not route clients."""
    path = str(path)
    if path.startswith("/eos/home-j/jniedzie/"):
        path = path.replace("/eos/home-j/jniedzie/", "/eos/user/j/jniedzie/", 1)
    if path.startswith("/eos/cms/"):
        return "root://eoscms.cern.ch/" + path
    if path.startswith("/eos/user/"):
        return "root://eosuser.cern.ch/" + path
    raise ValueError("No declared EOS instance for " + path)


def staged_read(row, frozen_source, group_id, cache_dir):
    """Keep one checked Nano in disposable local scratch, then remove that copy."""
    original = str(row.get("nano_path", row.get("path", "")))
    if not row.get("sha256", row.get("nano_sha256")):
        raise ValueError("Staging requires an explicit immutable Nano digest")
    with tempfile.TemporaryDirectory(prefix="nano-", dir=cache_dir) as directory:
        local = Path(directory) / "input.root"
        subprocess.run(["xrdcp", "--silent", "--nopbar", xrootd_url(original), str(local)],
                       check=True, timeout=300)
        local_row = dict(row, nano_path=str(local), path=str(local))
        data, receipt, names = read_nano(local_row, frozen_source, group_id)
        receipt["path"] = original
        receipt["staged_with"] = "xrdcp; exact declared digest checked before extraction"
        if data is not None:
            data["nano_path"] = np.full(len(data["y"]), original)
        return data, receipt, names


def extract_inventory(inventory, output, frozen_source, start=0, stop=None, cache_dir=None):
    """Extract an explicit slice; failures remain missing, with durable receipts."""
    inventory, output = Path(inventory), Path(output)
    manifest = json.loads(inventory.read_text())
    if manifest.get("do_not_read_collision_data") is not True or manifest.get("simulation_only") is not True:
        raise ValueError("Simulation-only inventory required")
    rows = manifest["rows"]
    end = len(rows) if stop is None else min(stop, len(rows))
    if start < 0 or start >= end:
        raise ValueError("Expected a nonempty, nonnegative inventory slice")
    source = Path(frozen_source).resolve()
    protocol_path = source.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    expected = protocol["source_sha256"]
    if not expected or any(sha256(source / name) != digest for name, digest in expected.items()):
        raise ValueError("Frozen extraction helper closure differs from its protocol")
    sys.path.insert(0, str(source))
    import importlib
    for name in ("features", "labels", "shift_vertex_reco_information", "vertex_pointing"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != source:
            raise ValueError("Extraction helper imported from another runtime: " + name)
    inventory_hash = sha256(inventory)
    groups = ownership_components(rows)
    output.mkdir(parents=True, exist_ok=False)
    parts, receipts, failures = [], [], []
    feature_names = None
    for index in range(start, end):
        try:
            if cache_dir is None:
                data, receipt, feature_names = read_nano(rows[index], frozen_source, groups[index])
            else:
                data, receipt, feature_names = staged_read(rows[index], frozen_source, groups[index], cache_dir)
            receipt["inventory_index"] = index
            receipts.append(receipt)
            if data is not None:
                parts.append(data)
            print(index, receipt["events"], receipt["pairs"], flush=True)
        except Exception as error:
            failures.append(dict(inventory_index=index, path=rows[index].get("path"),
                                 error=repr(error)))
            print("FAILED", index, repr(error), flush=True)
        report = dict(schema="shift-classifier-extraction-v1", simulation_only=True,
                      inventory_sha256=inventory_hash, start=start, stop=end,
                      source_sha256=expected, frozen_protocol_sha256=sha256(protocol_path),
                      files=receipts, failures=failures, complete=False)
        (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    if parts:
        data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        keys = [(str(path), *map(int, ids), int(pair)) for path, ids, pair in
                zip(data["nano_path"], data["ids"], data["pair_index"])]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate candidate within inventoried inputs")
        data["feature_names"] = feature_names
        np.savez_compressed(output / "data.npz", **data)
        report.update(data_sha256=sha256(output / "data.npz"), candidates=len(keys))
    else:
        report["candidates"] = 0
    report["complete"] = not failures
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise ValueError("Incomplete extraction: " + str(len(failures)) + " failed files")
    return report


def checked_input(reference, json_input=False):
    """Every assembly input is an explicit, immutable local readback."""
    path = Path(reference["path"])
    if not reference.get("sha256") or sha256(path) != reference["sha256"]:
        raise ValueError("Assembly input digest differs: " + str(path))
    if json_input:
        return json.loads(path.read_text())
    return dict(np.load(path, allow_pickle=False))


def concatenate_cohort(parts):
    """Retain extraction diagnostics, with missing cached metadata explicit."""
    columns = set().union(*(part.keys() for part in parts)) - {"feature_names"}
    combined = {}
    for name in sorted(columns):
        example = next(part[name] for part in parts if name in part)
        values = []
        for part in parts:
            if name in part:
                value = part[name]
                if value.shape[1:] != example.shape[1:]:
                    raise ValueError("Incompatible metadata shape: " + name)
            else:
                shape = (len(part["y"]),) + example.shape[1:]
                if example.dtype.kind in "US":
                    value = np.full(shape, "unavailable")
                elif example.dtype.kind == "b":
                    value = np.zeros(shape, dtype=bool)
                elif example.dtype.kind in "iu":
                    value = np.full(shape, -1, dtype=np.int64)
                elif example.dtype.kind in "fc":
                    value = np.full(shape, np.nan)
                else:
                    raise ValueError("Unsupported metadata type: " + name)
            if len(value) != len(part["y"]):
                raise ValueError("Misaligned cohort column: " + name)
            values.append(value)
        combined[name] = np.concatenate(values)
    combined["feature_names"] = parts[0]["feature_names"]
    return combined


def assemble_cohort(manifest_path, output):
    """Close the complete MC inventory before exposing a training cohort."""
    from shift_classifier_crossfit import fold_masks, ownership_partitions, validate_data

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    if not all(manifest.get(k) is True for k in ("simulation_only", "do_not_read_collision_data")):
        raise ValueError("Simulation-only assembly manifest required")
    names = ("cached_sm", "sm_preparation", "old_cache", "expanded_signal_proof", "global_ownership", "inventory")
    inputs = {name: checked_input(manifest[name], name not in ("cached_sm", "old_cache")) for name in names}
    sm, old = inputs["cached_sm"], inputs["old_cache"]
    preparation, expanded = inputs["sm_preparation"], inputs["expanded_signal_proof"]
    if preparation["dataset"]["sha256"] != manifest["cached_sm"]["sha256"]:
        raise ValueError("Cached SM preparation does not qualify this dataset")
    if preparation.get("global_groups_sha256") != manifest["global_ownership"]["sha256"]:
        raise ValueError("Cached SM and assembly use different ownership components")
    if len(sm["y"]) != preparation.get("rows", len(sm["y"])) or not np.all(np.isin(sm["process"], ("qcd", "dy", "jpsi"))):
        raise ValueError("Cached SM preparation census differs")
    if set(map(int, sm["cache_row"])) & set(preparation.get("omitted_cache_rows", [])):
        raise ValueError("Quarantined original-source rows entered cached SM")
    if expanded["full_current_data_sha256"] != manifest["old_cache"]["sha256"]:
        raise ValueError("Expanded signal proof qualifies another cache")
    if not all(r.get("original_source_ownership_qualified") is True for r in expanded["rows"]):
        raise ValueError("Expanded signal has unresolved source ownership")
    ownership, inventory = inputs["global_ownership"], inputs["inventory"]
    declared = ownership.get("source_manifest_sha256", {})
    for name, key in (("inventory", "current_bsm.json"), ("expanded_signal_proof", "expanded_signal_source_proof.json")):
        if declared.get(key) != manifest[name]["sha256"]:
            raise ValueError("Global ownership qualifies different " + name)
    rows = inventory["rows"]
    path_groups = {(r["path"], r["point"]): r["group_id"] for r in ownership["path_point_groups"] if "cache_row" not in r}
    if any((r.get("nano_path") or r["path"], r["point"]) not in path_groups for r in rows):
        raise ValueError("Global ownership does not cover the complete Nano inventory")
    sm = dict(sm)
    if "direct443" in old:
        cache_rows = sm["cache_row"]
        if not np.array_equal(sm["ids"], old["ids"][cache_rows]):
            raise ValueError("Cached SM identity differs from ancestry source cache")
        sm["direct443"] = old["direct443"][cache_rows]
    sm["owner_precision"] = np.array(["original_source_event"] * len(sm["y"]))
    coarse = set(preparation.get("current_nano_source_set_rows", []))
    sm["owner_precision"] = np.array(["source_group_response_identity" if int(i) in coarse else "original_source_event" for i in sm["cache_row"]])
    sm["label_metadata_available"] = np.zeros(len(sm["y"]), dtype=bool)
    sm["response_fingerprint"] = np.full(len(sm["y"]), "")
    sm["label_recovered"] = np.zeros(len(sm["y"]), dtype=bool)
    sm["feature_names"] = old["feature_names"]
    if not np.array_equal(inputs["cached_sm"]["feature_names"], old["feature_names"]):
        raise ValueError("Cached feature contracts differ")
    expanded_groups = {int(r["cache_row"]): r["group_id"] for r in ownership["expanded_cache_groups"]}
    indices = np.array([r["full_current_cache_row"] for r in expanded["rows"]], dtype=np.int64)
    if len(indices) != len(set(indices)) or not np.all(old["process"][indices] == "signal"):
        raise ValueError("Expanded signal cache-row identity differs")
    signal = {key: old[key][indices] for key in ("X", "y", "weight", "mass", "process", "point", "category", "domain", "ids", "pair_index", "label_reason")}
    proofs = {r["cache_source_index"]: r for r in expanded["source_proofs"]}
    signal.update(feature_names=old["feature_names"], original_y=signal["y"].copy(), cache_row=indices,
                  group_id=np.array([expanded_groups[int(i)] for i in indices]),
                  event_owner_id=np.array([r["event_owner_id"] for r in expanded["rows"]]),
                  nano_path=np.array([proofs[r["expanded_cache_source_index"]]["nano_path"] for r in expanded["rows"]]),
                  owner_precision=np.full(len(indices), "original_source_event"),
                  response_fingerprint=np.array([proofs[r["expanded_cache_source_index"]]["nano_sha256"] + ":" + ":".join(map(str, old["ids"][i])) + ":" + str(old["pair_index"][i]) for i, r in zip(indices, expanded["rows"])]),
                  label_recovered=np.zeros(len(indices), dtype=bool), label_metadata_available=np.zeros(len(indices), dtype=bool))
    if any(not np.array_equal(old["ids"][i], r["identity"]) or int(old["pair_index"][i]) != r["pair_index"] for i, r in zip(indices, expanded["rows"])):
        raise ValueError("Expanded signal identity proof differs")
    parts, seen_files, shard_proofs, file_census = [sm, signal], set(), [], []
    extraction_closure = None
    for shard in manifest["shards"]:
        receipt = checked_input(shard["receipt"], True)
        if shard.get("readback_verified") is not True or receipt.get("complete") is not True or receipt.get("failures"):
            raise ValueError("Incomplete or unverified shard readback")
        if receipt["inventory_sha256"] != manifest["inventory"]["sha256"]:
            raise ValueError("Shard used another frozen inventory")
        closure = (receipt.get("source_sha256"), receipt.get("frozen_protocol_sha256"))
        if extraction_closure is None:
            extraction_closure = closure
        elif closure != extraction_closure:
            raise ValueError("Shard extraction helper closures differ")
        numbers = [r["inventory_index"] for r in receipt["files"]]
        if sorted(numbers) != list(range(receipt["start"], receipt["stop"])) or seen_files.intersection(numbers):
            raise ValueError("Shard inventory coverage overlaps or has gaps")
        seen_files.update(numbers)
        file_census.extend(receipt["files"])
        for file in receipt["files"]:
            if not 0 <= file["inventory_index"] < len(rows):
                raise ValueError("Shard inventory index outside census")
            row = rows[file["inventory_index"]]
            if file["path"] != (row.get("nano_path") or row["path"]) or file["input_sha256"] != row["sha256"]:
                raise ValueError("Shard source identity differs from inventory")
            if row.get("expected_events") is not None and file["events"] != row["expected_events"]:
                raise ValueError("Shard processed-event count differs")
        candidates = sum(r["pairs"] for r in receipt["files"])
        if receipt["candidates"] != candidates:
            raise ValueError("Shard candidate census differs")
        if candidates == 0:
            if shard.get("data") is not None:
                empty = checked_input(shard["data"])
                if len(empty["y"]):
                    raise ValueError("Zero-candidate shard contains candidates")
            shard_proofs.append(dict(receipt=shard["receipt"], candidates=0))
            continue
        if not shard.get("data") or receipt["data_sha256"] != shard["data"]["sha256"]:
            raise ValueError("Missing qualified shard dataset")
        data = checked_input(shard["data"])
        if len(data["y"]) != candidates or not np.array_equal(data["feature_names"], old["feature_names"]):
            raise ValueError("Shard feature or candidate contract differs")
        path_counts = Counter(map(str, data["nano_path"]))
        if path_counts != Counter({r["path"]: r["pairs"] for r in receipt["files"] if r["pairs"]}):
            raise ValueError("Shard per-file candidate census differs")
        identities = [(str(path), *map(int, identity), int(pair)) for path, identity, pair in zip(data["nano_path"], data["ids"], data["pair_index"])]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate candidate inside a shard")
        by_path = {(rows[i].get("nano_path") or rows[i]["path"]): rows[i] for i in numbers}
        groups, owners, precision, fingerprints = [], [], [], []
        for path, point, process, identity, pair in zip(data["nano_path"], data["point"], data["process"], data["ids"], data["pair_index"]):
            row = by_path[str(path)]
            if str(point) != row["point"] or str(process) != row.get("process", "signal"):
                raise ValueError("Shard physical point or process differs")
            group = path_groups[(str(path), str(point))]
            sources = row.get("source_records", row.get("original_sources", []))
            source_hashes = {s["gen_sha256"] for s in sources if s.get("run_number") == int(identity[0]) and s.get("gen_sha256")}
            exact = len(source_hashes) == 1
            owner = ("GEN_SHA256:" + next(iter(source_hashes)) + ":source_event:") if exact else (group + ":response:")
            groups.append(group); owners.append(owner + ":".join(map(str, identity)))
            precision.append("original_gen_event_namespace" if exact else "source_group_response_identity")
            fingerprints.append(row["sha256"] + ":" + ":".join(map(str, identity)) + ":" + str(pair))
        data.update(group_id=np.array(groups), event_owner_id=np.array(owners), owner_precision=np.array(precision),
                    response_fingerprint=np.array(fingerprints), label_metadata_available=np.ones(candidates, dtype=bool), cache_row=np.full(candidates, -1, dtype=np.int64))
        parts.append(data)
        shard_proofs.append(dict(receipt=shard["receipt"], data=shard["data"], candidates=candidates))
    if seen_files != set(range(len(rows))):
        raise ValueError("Incomplete full-inventory shard coverage")
    data = concatenate_cohort(parts)
    known = data["original_y"] >= 0
    if np.any(known & (data["y"] != data["original_y"])):
        raise ValueError("Assembly changed a known truth label")
    recovered = data["label_recovered"]
    if np.any(recovered & ((data["original_y"] != -1) | (data["y"] < 0))):
        raise ValueError("Invalid unknown-label recovery accounting")
    keep, aliases, previous = [], [], {}
    for i, (point, owner, precision, response) in enumerate(zip(data["point"], data["event_owner_id"], data["owner_precision"], data["response_fingerprint"])):
        key = (str(point), str(owner), str(response))
        if response and precision != "source_group_response_identity" and key in previous:
            j = previous[key]
            if not all(np.array_equal(data[k][i], data[k][j], equal_nan=True) if data[k].dtype.kind in "fc" else np.array_equal(data[k][i], data[k][j]) for k in ("X", "y", "original_y", "weight", "mass")):
                raise ValueError("Alias response fingerprint has conflicting content")
            aliases.append(dict(removed_row=i, retained_row=j, point=str(point), event_owner_id=str(owner)))
        else:
            keep.append(i); previous[key] = i
    for key in set(data) - {"feature_names"}:
        data[key] = data[key][keep]
    validate_data(data)
    fold, inner = ownership_partitions(data["group_id"])
    data["crossfit_fold"], data["inner_validation_group"] = fold, inner
    ready = {}
    for mode in ("sm-only", "sm-bsm"):
        for f in (0, 1):
            masks = fold_masks(data, f, mode)
            ready[mode + ":scorefold" + str(f)] = {name: dict(rows=int(mask.sum()), labels=dict(Counter(map(str, data["y"][mask])))) for name, mask in zip(("train", "validation", "scored"), masks)}
    correlations = Counter(map(str, data["event_owner_id"]))
    replay_groups = {}
    for owner, count in correlations.items():
        if count <= 1:
            continue
        mask = data["event_owner_id"] == owner
        replay_groups[owner] = dict(candidates=count, points=sorted(set(map(str, data["point"][mask]))),
                                   domains=sorted(set(map(str, data["domain"][mask]))),
                                   precision=sorted(set(map(str, data["owner_precision"][mask]))))
    point_census = {}
    for point in sorted(set(r["point"] for r in rows)):
        selected_files = [file for file in file_census if rows[file["inventory_index"]]["point"] == point]
        observed = sum(file["pairs"] for file in selected_files)
        mask = (data["process"] == "signal") & (data["point"] == point) & data["label_metadata_available"]
        point_census[point] = dict(inventory_files=len(selected_files), processed_raw_MC_events=sum(file["events"] for file in selected_files),
                                  observed_raw_pairs=observed, retained_candidates=int(mask.sum()),
                                  labels=dict(Counter(map(str, data["y"][mask]))),
                                  observation_status="observed_candidates" if observed else "no_observed_candidates; efficiency and physical yield remain unknown")
    receipt = dict(schema="shift-classifier-assembled-cohort-v1", complete=True, simulation_only=True,
        manifest_sha256=sha256(manifest_path), inputs={name: manifest[name] for name in names}, shards=shard_proofs,
        extraction_source_sha256=extraction_closure[0], extraction_protocol_sha256=extraction_closure[1],
        inventory_files=len(rows), candidates=len(data["y"]), source_groups=len(set(data["group_id"])),
        labels=dict(Counter(map(str, data["y"]))), by_process={str(p): dict(Counter(map(str, data["y"][data["process"] == p]))) for p in np.unique(data["process"])},
        by_domain=dict(Counter(map(str, data["domain"]))), recovered_labels=int(recovered[keep].sum()),
        unknown_reasons=dict(Counter(map(str, data["label_reason"][data["y"] < 0]))),
        original_source_quarantine=preparation.get("omitted_cache_rows", []), aliases_removed=aliases,
        repeated_event_owners=replay_groups,
        ownership_precision=dict(Counter(map(str, data["owner_precision"]))), fold_readiness=ready,
        inventory_physical_points=sorted(point_census), current_BSM_point_census=point_census,
        exposure="No physical exposures pooled or normalized; native event weights retained unchanged.",
        excluded_recent_DP_cache=dict(rows=int(np.sum((old["process"] == "signal") & (old["domain"] == "recent"))),
                                      reason="Earlier recent-DP controls replaced by canonical current source closure."))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "data.npz", **data)
    receipt["data_sha256"] = sha256(output / "data.npz")
    (output / "cohort_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory", type=Path)
    mode.add_argument("--assemble", type=Path, help="Close a hash-pinned cached/sharded MC cohort")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--cache-dir", type=Path,
                        help="Stage one EOS input at a time with XRootD into this local directory")
    args = parser.parse_args()
    if args.assemble is not None:
        if args.frozen_source is not None or args.start != 0 or args.stop is not None or args.cache_dir is not None:
            parser.error("Extraction options cannot be supplied with --assemble")
        assemble_cohort(args.assemble, args.output)
    else:
        if args.frozen_source is None:
            parser.error("--inventory requires --frozen-source")
        extract_inventory(args.inventory, args.output, args.frozen_source, args.start, args.stop, args.cache_dir)


if __name__ == "__main__":
    main()
