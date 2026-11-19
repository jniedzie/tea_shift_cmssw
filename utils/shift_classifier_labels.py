# Code style: AI-code. AI-generated truth-label diagnosis and conservative recovery.
"""MC-only label recovery; never import from classifier prediction code.

The original label and reason come from the frozen label implementation. Known
labels stay unchanged. Missing GEN links can be recovered only through an exact
accepted hit-associated SimTrack identity, with actual same-event simulation
vertices when two distinct tracks are involved. Reconstructed kinematics,
sample/process names and the quality of the fitted vertex are never read.
"""

import math


def leg_diagnosis(event, index):
    """Separate rejected hit associations from accepted tracks lacking GEN links."""
    track = int(event["ShiftMuon_hitSimTrackId"][index])
    layers = int(event["ShiftMuon_hitTruthMatchedLayers"][index])
    purity = float(event["ShiftMuon_hitTruthPurity"][index])
    if not math.isfinite(purity):
        association = "nonfinite_purity"
    elif layers < 3:
        association = "too_few_layers"
    elif purity < 0.75:
        association = "low_purity"
    elif track < 0:
        association = "rejected_association"
    else:
        association = "accepted"

    gen = int(event["ShiftMuon_hitGenPartIdx"][index])
    if not 0 <= gen < len(event["GenPart_pdgId"]):
        gen_state = "missing_gen_link"
    elif abs(int(event["GenPart_pdgId"][gen])) != 13:
        gen_state = "non_muon_gen"
    elif int(event["GenPart_status"][gen]) != 1:
        gen_state = "non_status1_gen"
    else:
        gen_state = "status1_muon"
    return dict(sim_track_id=track, matched_layers=layers, purity=purity,
                association=association, gen_index=gen, gen_state=gen_state)


def _simulation_tracks(event, sim_truth):
    expected_request = event.get("original_request_sha256")
    actual_request = sim_truth.get("original_request_sha256")
    if (not isinstance(expected_request, str) or len(expected_request) != 64
            or any(value not in "0123456789abcdef" for value in expected_request)
            or actual_request != expected_request):
        raise ValueError("Simulation truth does not match the exact original detector request")
    expected = tuple(int(event[key]) for key in ("run", "luminosityBlock", "event"))
    if tuple(int(value) for value in sim_truth["event_id"]) != expected:
        raise ValueError("Simulation truth belongs to a different run/lumi/event")
    tracks = {}
    for record in sim_truth["tracks"]:
        identity = int(record["track_id"])
        if identity < 0 or identity in tracks:
            raise ValueError("Simulation track identities must be unique and nonnegative")
        tracks[identity] = record
    return tracks


def recover_pair_label(event, first, second, original_label, original_reason,
                       sim_truth=None, common_tolerance_cm=0.01,
                       separate_tolerance_cm=0.1):
    """Return ``(label, reason, diagnosis)`` without weakening matching quality.

    Optional ``sim_truth`` is an independently read EDM event record containing
    ``event_id=(run,lumi,event)``, ``original_request_sha256`` and ``tracks``.
    The target event must carry the same independently verified original request
    hash; matching event numbers alone is insufficient across replays. Each track record contains
    ``track_id``, ``pdg_id``, ``vertex_index`` and ``vertex_cm=(x,y,z)``. It must
    come from the exact original detector response, never GEN extrapolation,
    an angular reco match, or a newly sampled event. Missing records stay unknown.
    Each recovered track must additionally certify
    ``origin_semantics='validated_physical_muon_birth'`` and a nonempty string
    ``physical_muon_id``. Raw transport/restart vertices do not certify birth;
    validated primary GEN/HepMC or secondary-creation ancestry must establish
    the physical origin and identity before the caller supplies that certificate.
    """
    if original_label not in (-1, 0, 1):
        raise ValueError("Original label must be unknown, fake or genuine")
    if not (0 <= common_tolerance_cm < separate_tolerance_cm):
        raise ValueError("Invalid vertex-distance tolerance convention")
    legs = [leg_diagnosis(event, index) for index in (first, second)]
    diagnosis = dict(legs=legs, recovered=False, status="known_label_preserved")
    if original_label >= 0:
        return int(original_label), original_reason, diagnosis
    diagnosis["status"] = "hit_association_unresolved"
    if any(leg["association"] != "accepted" for leg in legs):
        return -1, original_reason, diagnosis

    identities = [leg["sim_track_id"] for leg in legs]
    if identities[0] == identities[1]:
        diagnosis.update(recovered=True, status="duplicate_sim_track")
        return 0, "same_hit_sim_track", diagnosis

    diagnosis["status"] = "simulation_vertex_unavailable"
    if sim_truth is None:
        return -1, original_reason, diagnosis
    tracks = _simulation_tracks(event, sim_truth)
    if any(identity not in tracks for identity in identities):
        return -1, original_reason, diagnosis
    records = [tracks[identity] for identity in identities]
    if any(abs(int(record["pdg_id"])) != 13 for record in records):
        diagnosis["status"] = "simulation_particle_not_muon"
        return -1, original_reason, diagnosis
    if any(record.get("origin_semantics") != "validated_physical_muon_birth" for record in records):
        diagnosis["status"] = "physical_birth_unqualified"
        return -1, original_reason, diagnosis
    physical_ids = [record.get("physical_muon_id") for record in records]
    if any(not isinstance(identity, str) or not identity for identity in physical_ids):
        diagnosis["status"] = "physical_muon_identity_unqualified"
        return -1, original_reason, diagnosis
    if physical_ids[0] == physical_ids[1]:
        diagnosis.update(recovered=True, status="duplicate_physical_muon")
        return 0, "same_physical_muon", diagnosis
    positions = [tuple(float(value) for value in record["vertex_cm"]) for record in records]
    if any(int(record["vertex_index"]) < 0 for record in records) or any(
            len(position) != 3 or not all(math.isfinite(value) for value in position)
            for position in positions):
        diagnosis["status"] = "simulation_vertex_invalid"
        return -1, original_reason, diagnosis
    distance = math.dist(*positions)
    diagnosis["vertex_distance_cm"] = distance
    if distance <= common_tolerance_cm:
        diagnosis.update(recovered=True, status="common_simulation_vertex")
        return 1, "common_hit_sim_vertex", diagnosis
    if distance >= separate_tolerance_cm:
        diagnosis.update(recovered=True, status="different_simulation_vertices")
        return 0, "different_hit_sim_vertices", diagnosis
    diagnosis["status"] = "simulation_vertex_tolerance_gap"
    return -1, original_reason, diagnosis


TASK_LABEL_SCHEMA = "shift-physical-muon-task-labels-v1"
TASK_NAMES = ("legacy-common-origin", "distinct-muons", "vertex-origin")
TASK_REASONS = {
    "distinct-muons": {
        "distinct_common_origin": 1, "distinct_different_origins": 1,
        "distinct_vertex_tolerance_gap": 1, "certified_distinct_physical_muons": 1,
        "duplicate_same_muon": 0, "duplicate_same_sim_track": 0,
        "duplicate_same_physical_muon": 0,
    },
    "vertex-origin": {"common_origin": 1, "different_origins": 0},
}
TASK_FIELDS = {"distinct-muons": "distinct_muons", "vertex-origin": "vertex_origin"}


def task_contract(task):
    if task not in TASK_NAMES:
        raise ValueError("Unknown classifier task: " + str(task))
    contract = dict(task=task, schema=TASK_LABEL_SCHEMA,
                    input_population="Retained selected OS dimuon pairs only",
                    inference="Reconstructed predictors only; truth/reasons/identities never enter X",
                    original_binary_labels_preserved=True)
    if task == "distinct-muons":
        contract.update(positive="Two distinct physical muons, regardless of birth position",
                        negative="Two reconstructed legs of the same physical muon",
                        application="Pair-veto development control; no muon leg removal or representative reranking",
                        missing_scope="Upstream muon deduplication needs all reconstructed-muon pairs including same-sign, qualified overlap graph and representative policy, and close/low-mass genuine controls")
    elif task == "vertex-origin":
        contract.update(positive="Distinct muons with a common physical birth; inaccurate fitted vertices remain positive",
                        negative="Distinct muons from different physical births",
                        duplicate_policy="Excluded from this task; task label remains unknown")
    else:
        contract.update(positive="Frozen common-origin binary label",
                        negative="Frozen mixture of duplicate legs and different births")
    return contract


def task_pair_label(original_label, original_reason, task):
    """Split an existing strict truth label by explicit reason, never reco shape."""
    task_contract(task)
    if original_label not in (-1, 0, 1):
        raise ValueError("Original label must be integer -1/0/1")
    reason = str(original_reason)
    if task == "legacy-common-origin":
        return int(original_label), reason
    common = reason in ("common_vertex", "common_hit_sim_vertex")
    different = reason in ("different_vertices", "different_hit_sim_vertices")
    duplicates = {"same_muon_twice": "duplicate_same_muon",
                  "same_hit_sim_track": "duplicate_same_sim_track",
                  "same_physical_muon": "duplicate_same_physical_muon"}
    if (common and original_label != 1) or ((different or reason in duplicates) and original_label != 0):
        raise ValueError("Original binary label contradicts its explicit truth reason")
    if original_label == 1 and not common and reason != "frozen_old_label":
        raise ValueError("Original positive lacks a recognized strict common-origin reason")
    if original_label == 1:
        # A frozen positive already certifies distinct common-origin muons.
        return 1, "distinct_common_origin" if task == "distinct-muons" else "common_origin"
    if different:
        return (1, "distinct_different_origins") if task == "distinct-muons" else (0, "different_origins")
    if reason in duplicates:
        return (0, duplicates[reason]) if task == "distinct-muons" else (-1, "unknown_excluded_duplicate_muons")
    if reason == "vertex_tolerance_gap" and original_label == -1 and task == "distinct-muons":
        # The frozen labeler establishes two distinct accepted truth muons
        # before testing this explicitly identified birth-position gap.
        return 1, "distinct_vertex_tolerance_gap"
    return -1, "unknown_" + reason


def task_labels_for_data(data, task):
    """Return task labels and reasons while leaving the original arrays intact.

    Optional certified task arrays may add independent physical-identity truth
    unavailable in the old binary label. Their explicit schema and reason are
    required; the frozen cohort's reader/proof must qualify the truth source.
    They cannot contradict a known label derived from the existing strict truth.
    """
    import numpy as np
    task_contract(task)
    original = np.asarray(data["y"])
    reasons = data.get("label_reason")
    if reasons is None:
        if task != "legacy-common-origin":
            raise ValueError("Separate classifier tasks require explicit label_reason metadata")
        reasons = np.full(len(original), "legacy_binary_label")
    if np.asarray(reasons).shape != original.shape:
        raise ValueError("Truth reasons must align with original labels")
    derived = [task_pair_label(int(y), reason, task) for y, reason in zip(original, reasons)]
    y = np.array([value for value, _ in derived], dtype=np.int8)
    why = np.array([reason for _, reason in derived], dtype="U128")
    if task == "legacy-common-origin":
        return y, why
    field = TASK_FIELDS[task]
    if field + "_y" in data:
        supplied = np.asarray(data[field + "_y"]); supplied_reason = np.asarray(data[field + "_reason"])
        schema = np.asarray(data.get("task_label_schema", []))
        if (supplied.shape != y.shape or supplied_reason.shape != y.shape or schema.shape != y.shape
                or not np.all(schema == TASK_LABEL_SCHEMA) or not np.isin(supplied, [-1, 0, 1]).all()):
            raise ValueError("Explicit task labels require aligned labels, reasons and the qualified schema")
        for i, (value, reason) in enumerate(zip(supplied, supplied_reason)):
            expected = TASK_REASONS[task].get(str(reason), -1 if str(reason).startswith("unknown_") else None)
            excluded_duplicate = why[i] == "unknown_excluded_duplicate_muons"
            if (expected is None or int(value) != expected or (y[i] >= 0 and int(value) != y[i])
                    or (excluded_duplicate and (int(value) != -1 or str(reason) != why[i]))):
                raise ValueError("Explicit task label contradicts its truth reason or known original truth")
        y, why = supplied.astype(np.int8, copy=True), supplied_reason.astype("U128", copy=True)
    return y, why


def pair_task_truth(event, first, second, original_label, original_reason, sim_truth=None):
    """Derive the two truth tasks; raw SimTrack numbers are not distinct identities."""
    result = {}
    for task, field in TASK_FIELDS.items():
        result[field + "_y"], result[field + "_reason"] = task_pair_label(original_label, original_reason, task)
    result["task_label_schema"] = TASK_LABEL_SCHEMA
    legs = [leg_diagnosis(event, index) for index in (first, second)]
    if any(leg["association"] != "accepted" for leg in legs):
        return result
    if legs[0]["sim_track_id"] == legs[1]["sim_track_id"]:
        if result["distinct_muons_y"] == 1:
            raise ValueError("Accepted shared hit track contradicts known distinct physical muons")
        result.update(distinct_muons_y=0, distinct_muons_reason="duplicate_same_sim_track",
                      vertex_origin_y=-1, vertex_origin_reason="unknown_excluded_duplicate_muons")
        return result
    if sim_truth is None:
        return result
    tracks = _simulation_tracks(event, sim_truth)
    identities = [leg["sim_track_id"] for leg in legs]
    if any(identity not in tracks for identity in identities):
        return result
    records = [tracks[identity] for identity in identities]
    if any(abs(int(record["pdg_id"])) != 13 for record in records):
        return result
    certified = all(record.get("identity_semantics") == "validated_physical_muon_identity"
                    or record.get("origin_semantics") == "validated_physical_muon_birth" for record in records)
    physical = [record.get("physical_muon_id") for record in records]
    if not certified or any(not isinstance(value, str) or not value for value in physical):
        return result
    distinct = physical[0] != physical[1]
    if result["distinct_muons_y"] >= 0 and result["distinct_muons_y"] != int(distinct):
        raise ValueError("Certified physical identities contradict known original truth")
    result["distinct_muons_y"] = int(distinct)
    result["distinct_muons_reason"] = "certified_distinct_physical_muons" if distinct else "duplicate_same_physical_muon"
    if not distinct:
        result["vertex_origin_y"], result["vertex_origin_reason"] = -1, "unknown_excluded_duplicate_muons"
    elif result["vertex_origin_y"] < 0:
        label, reason, _ = recover_pair_label(event, first, second, -1, original_reason, sim_truth)
        if label >= 0:
            result["vertex_origin_y"], result["vertex_origin_reason"] = task_pair_label(label, reason, "vertex-origin")
    return result


def refine_frozen_label_reasons(data, cache_directory):
    """Recover reason metadata using exact archived source/RECO119/label parity."""
    from collections import Counter
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    cache = Path(cache_directory); manifest_path = cache / "inputs.json"
    manifest = json.loads(manifest_path.read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name in ("reco", "labels"):
        if sha(cache / (name + ".npz")) != manifest[name + "_sha256"]:
            raise ValueError("Reference cache hash differs: " + name)
    reco = dict(np.load(cache / "reco.npz", allow_pickle=False))
    labels = dict(np.load(cache / "labels.npz", allow_pickle=False))
    names = tuple(map(str, reco["feature_names"]))
    if len(names) != 119 or tuple(map(str, data["feature_names"][:119])) != names:
        raise ValueError("Reference must have the exact ordered RECO119 prefix")
    canonical = lambda path: str(path).replace("/eos/home-j/jniedzie/", "/eos/user/j/jniedzie/", 1)
    paths = {int(row.get("index", row.get("source_index", i))): canonical(row.get("path", row.get("nano", {}).get("path", "")))
             for i, row in enumerate(manifest["inputs"])}
    lookup = {}
    for i, (source, process, ids, pair) in enumerate(zip(reco["source_index"], reco["process"], reco["ids"], reco["pair_index"])):
        path = paths[int(source)]
        if not path:
            raise ValueError("Reference source lacks its explicit Nano path")
        key = (path, str(process), *map(int, ids), int(pair))
        if key in lookup:
            raise ValueError("Duplicate source/event/pair identity in reference cache")
        lookup[key] = i
    refined = dict(data); reasons = np.asarray(data["label_reason"]).astype("U128", copy=True); matched = []
    muon_indices_checked = 0; muon_indices_unavailable = 0
    for i in np.flatnonzero(reasons == "frozen_old_label"):
        key = (canonical(data["nano_path"][i]), str(data["process"][i]), *map(int, data["ids"][i]), int(data["pair_index"][i]))
        j = lookup.get(key)
        if j is None:
            continue
        if int(data["y"][i]) != int(labels["y"][j]) or not np.array_equal(data["X"][i, :119], reco["X"][j], equal_nan=True):
            raise ValueError("Archived truth/RECO119 parity failed for an explicitly matched source pair")
        if "muon_indices" in data and np.all(np.asarray(data["muon_indices"][i]) >= 0):
            if not np.array_equal(data["muon_indices"][i], reco["muon_indices"][j]):
                raise ValueError("Archived muon-index parity failed")
            muon_indices_checked += 1
        else:
            # The sealed cohort uses [-1,-1] for unavailable historical row
            # metadata. Source/event/pair and all119 features remain mandatory.
            muon_indices_unavailable += 1
        reasons[i] = str(labels["reason"][j]); matched.append([int(i), int(j)])
    refined["label_reason"] = reasons
    proof = dict(schema="shift-frozen-label-reason-refinement-v1", reference_manifest=str(manifest_path.resolve()),
                 reference_manifest_sha256=sha(manifest_path), reference_reco_sha256=manifest["reco_sha256"],
                 reference_labels_sha256=manifest["labels_sha256"], refined_rows=len(matched), matched_row_map=matched,
                 original_binary_y_unchanged=np.array_equal(refined["y"], data["y"]),
                 explicit_Nano_source_and_event_pair_matched=True, ordered_RECO119_exact=True,
                 known_muon_indices_checked=muon_indices_checked, unavailable_muon_index_rows=muon_indices_unavailable,
                 refined_reason_counts=dict(Counter(str(reasons[i]) for i, _ in matched)))
    return refined, proof
