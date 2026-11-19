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
