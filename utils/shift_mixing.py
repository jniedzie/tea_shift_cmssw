"""Reusable event-mixing calculation selected by shift_histogrammer_config.

This backend extracts audited, event-preserving inputs, reproduces the published
unconstrained pair fit, and estimates independent accidental-pair shapes.
The main SHIFT histogram/plot configurations own how the study runs and looks.
Same-event fit closure is a required gate. Smooth plots do not validate the
mixed-event background rate. Row augmentation remains a shape-only study.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "configs"))
from shift_paths import cross_sections
from shift_sample_paths import full_gen_weight_sums
from shift_storage_paths import resolve_storage_path

MUON_MASS_GEV = 0.105658

FIT_IMPLEMENTATION_PROVENANCE = {
    "producer_source": "PhysicsTools/ShiftMuonSegments/plugins/ShiftMuonTableProducer.cc",
    "producer_source_sha256": "4821f95d103488f9880f018826c0405d83d913c806d4ac229b164ecb770bb7a0",
    "frozen_v10_cfi_sha256": "4aeb85ed7d1f7a569420bc97369170de48fe6a99271c21f95d612889ce5df771",
    "frozen_v10_runtime_sha256": "f535b1cf57021e0bd6e35e7b0b057842f5823476620b475e6b06bfecbbe9e295",
    "source_functions": ["straightLineApproach", "commonLineVertex", "PairChoice"],
    "cmssw_compiler_float_semantics": "SCRAM -O3 -march=x86-64-v3: scalar fit/mass dots use FMA; standalone DCA uses SIMD multiply then reduction and fused closest points",
    "hypothesis": "published unconstrained analytic pair",
    # The binary bundle preserves cfi, not the source .cc. Same-event closure
    # is therefore required to verify agreement with the deployed producer.
    "deployed_producer_source_in_bundle": False,
}

# These are current source/cfi defaults, not an implicit production contract.
# The caller should pass the resolved frozen production values and record them.
DEFAULT_CONFIG = {
    "originTransverseResolution": 100.0,
    "originZResolution": 2000.0,
    "commonVertexLineResolution": 100.0,
    "commonVertexBeamLineResolution": 100.0,
    "maxPairOriginNormalizedChi2": 9.0,
    "maxPairDca": 500.0,
    "maxDimuonVertices": 1,
    "requireOppositeSign": True,
    # CMSSW GlobalPoint/GlobalVector store float components and return float
    # dot/mag/unit operations. This matters for nearly parallel, boosted pairs.
    "floatStateMath": True,
}

FIT_MUON_FIELDS = ("vx", "vy", "vz", "px", "py", "pz", "charge", "topology")


def resolved_config(config: Optional[Mapping] = None) -> dict:
    """Validate analytic-fit settings; reject unavailable pair propagation."""
    result = dict(DEFAULT_CONFIG)
    result.update(config or {})
    # The published unconstrained hypothesis is reset after optional material
    # transport in the deployed source. Experimental transported/refitted
    # branches are separate hypotheses and require CMSSW detector states.
    for name in ("usePairVertexPropagation", "useCanonicalPairStatesPropagation",
                 "useRefittedPairStates", "useConstrainedPairStates"):
        if result.get(name, False):
            raise ValueError(f"{name} requires unavailable CMSSW pair states")
    for name in ("originTransverseResolution", "originZResolution",
                 "commonVertexLineResolution", "commonVertexBeamLineResolution"):
        value = float(result[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        result[name] = value
    for name in ("maxPairDca", "maxPairOriginNormalizedChi2"):
        value = float(result[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        result[name] = value
    count = result["maxDimuonVertices"]
    if isinstance(count, bool) or int(count) != count or count < 0:
        raise ValueError("maxDimuonVertices must be a nonnegative integer")
    result["maxDimuonVertices"] = int(count)
    return result


def _dot(first, second, float_math=False, fused=True):
    if not float_math:
        return float(np.asarray(first) @ np.asarray(second))
    # ExtVec::dot accumulates component products in T=float, left to right.
    # SCRAM's -O3 -march=x86-64-v3 contracts each accumulate to float FMA.
    # A product of two floats is exactly representable in double; performing
    # the add before the one float cast reproduces fmaf for these states.
    value = np.float32(0.0)
    for a, b in zip(first, second):
        if fused:
            value = np.float32(float(np.float32(a)) * float(np.float32(b)) + float(value))
        else:
            value = np.float32(value + np.float32(np.float32(a) * np.float32(b)))
    return float(value)


def _mag(momentum, float_math=False, fused=True):
    value = math.sqrt(_dot(momentum, momentum, float_math, fused))
    return float(np.float32(value)) if float_math else value


def _unit(momentum, float_math=False, fused=True):
    magnitude = _mag(momentum, float_math, fused)
    if float_math:
        return momentum * np.float32(np.float32(1.0) / np.float32(magnitude))
    return momentum / magnitude


def _muon_state(muon: Mapping, float_math=False) -> tuple:
    missing = [name for name in FIT_MUON_FIELDS if name not in muon]
    if missing:
        raise ValueError("Missing persisted ShiftMuon fields: " + ", ".join(missing))
    dtype = np.float32 if float_math else float
    point = np.asarray([muon[name] for name in ("vx", "vy", "vz")], dtype=dtype)
    momentum = np.asarray([muon[name] for name in ("px", "py", "pz")], dtype=dtype)
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(momentum)):
        raise ValueError("Nonfinite persisted ShiftMuon state")
    magnitude = _mag(momentum, float_math)
    if magnitude == 0.0 or not math.isfinite(magnitude):
        raise ValueError("Zero or nonfinite persisted ShiftMuon momentum")
    charge = int(muon["charge"])
    if charge not in (-1, 1) or charge != muon["charge"]:
        raise ValueError("Persisted ShiftMuon charge must be -1 or +1")
    direction = _unit(momentum, float_math)
    return point, momentum, direction


def straight_line_approach(first_point, first_direction,
                           second_point, second_direction, float_math=False) -> Optional[dict]:
    """CMSSW straightLineApproach, including its near-parallel convention.

    Directions must be unit vectors. The near-parallel branch deliberately
    uses full point separation, as the producer does; it is not the usual
    perpendicular distance between infinite parallel lines.
    """
    dtype = np.float32 if float_math else float
    first_point, second_point = np.asarray(first_point, dtype=dtype), np.asarray(second_point, dtype=dtype)
    first_direction, second_direction = np.asarray(first_direction, dtype=dtype), np.asarray(second_direction, dtype=dtype)
    separation = first_point - second_point
    # Unlike the inlined scalar fit/mass path, this standalone compiled
    # function preserves SIMD products, followed by separate float reduction.
    dot = _dot(first_direction, second_direction, float_math, fused=False)
    denominator = 1.0 - dot * dot
    if denominator < 1.0e-10:
        midpoint = np.asarray(0.5 * (first_point + second_point), dtype=dtype)
        distance = _mag(separation, float_math, fused=False)
    else:
        first_dot = _dot(first_direction, separation, float_math, fused=False)
        second_dot = _dot(second_direction, separation, float_math, fused=False)
        first_scale = (dot * second_dot - first_dot) / denominator
        second_scale = (second_dot - dot * first_dot) / denominator
        if float_math:
            # Inlined GlobalPoint + scale*GlobalVector contracts to float
            # FMA under the production compiler flags. Round scale to float
            # first (Basic3DVector scalar multiplication), then round only
            # once after multiply+add. Near-parallel tracks can otherwise
            # acquire a visibly different DCA from cancellation at large z.
            first_closest = np.asarray(first_point.astype(float) +
                                       float(np.float32(first_scale)) * first_direction.astype(float),
                                       dtype=np.float32)
            second_closest = np.asarray(second_point.astype(float) +
                                        float(np.float32(second_scale)) * second_direction.astype(float),
                                        dtype=np.float32)
        else:
            first_closest = first_point + first_scale * first_direction
            second_closest = second_point + second_scale * second_direction
        midpoint = np.asarray(0.5 * (first_closest + second_closest), dtype=dtype)
        distance = _mag(first_closest - second_closest, float_math, fused=False)
    if not math.isfinite(distance) or not np.all(np.isfinite(midpoint)):
        return None
    return {"distance": distance, "midpoint": midpoint}


def _inverse_symmetric_3x3(matrix):
    """Same cofactor formula and determinant threshold as the producer."""
    a, b, c = matrix[0]
    d, e, f = matrix[1, 1], matrix[1, 2], matrix[2, 2]
    determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
    if not math.isfinite(determinant) or abs(determinant) < 1.0e-18:
        return None
    return np.asarray([
        [d * f - e * e, c * e - b * f, b * e - c * d],
        [c * e - b * f, a * f - c * c, b * c - a * e],
        [b * e - c * d, b * c - a * e, a * d - b * b],
    ]) / determinant


def common_line_vertex(first_point, first_direction, second_point,
                       second_direction, line_resolution=100.0,
                       beam_line_resolution=100.0, float_math=False) -> Optional[dict]:
    """Analytic common-line fit with an x=y=0 prior and three fit dof."""
    normal = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    line_weight = 1.0 / (line_resolution * line_resolution)
    beam_weight = 1.0 / (beam_line_resolution * beam_line_resolution)
    for point, direction in ((first_point, first_direction), (second_point, second_direction)):
        direction = np.asarray(direction, dtype=float)
        projector = np.eye(3) - np.outer(direction, direction)
        normal += line_weight * projector
        rhs += line_weight * (projector @ point)
    normal[0, 0] += beam_weight
    normal[1, 1] += beam_weight
    covariance = _inverse_symmetric_3x3(normal)
    if covariance is None:
        return None
    fitted = covariance @ rhs
    if not np.all(np.isfinite(fitted)):
        return None
    chi2 = (fitted[0] * fitted[0] + fitted[1] * fitted[1]) * beam_weight
    for point, direction in ((first_point, first_direction), (second_point, second_direction)):
        dtype = np.float32 if float_math else float
        delta = np.asarray(fitted - point, dtype=dtype)
        residual = np.cross(delta, np.asarray(direction, dtype=dtype))
        chi2 += _mag(residual, float_math) ** 2 * line_weight
    position = fitted.astype(np.float32) if float_math else fitted
    return {"position": position, "error": np.sqrt(np.maximum(0.0, np.diag(covariance))),
            "covariance": covariance, "chi2": float(chi2), "ndof": 3.0}


def _eta(momentum) -> float:
    pt = math.hypot(momentum[0], momentum[1])
    return math.asinh(momentum[2] / pt) if pt > 0.0 else math.copysign(math.inf, momentum[2])


def pair_observables(first: Mapping, second: Mapping,
                     config: Optional[Mapping] = None) -> Optional[dict]:
    """Return analytic pair and shape observables, or None for a rejected pair.

    Missing state/schema fields raise ValueError. Nonfinite/zero states cannot
    establish closure and are not silently assigned a placeholder hypothesis.
    ``requireOppositeSign`` controls selection; set it false for a separate
    like-sign control. Index fields are supplied by ``select_pairs``.
    """
    cfg = resolved_config(config)
    float_math = cfg["floatStateMath"]
    x1, p1, u1 = _muon_state(first, float_math)
    x2, p2, u2 = _muon_state(second, float_math)
    approach = straight_line_approach(x1, _unit(p1, float_math, fused=False),
                                      x2, _unit(p2, float_math, fused=False), float_math)
    if approach is None or approach["distance"] > cfg["maxPairDca"]:
        return None
    is_os = int(first["charge"] != second["charge"])
    if cfg["requireOppositeSign"] and not is_os:
        return None
    if x1[2] * x2[2] <= 0.0:
        return None
    # GlobalPoint subtraction happens in float before assignment to doubles.
    delta = (x1 - x2).astype(float)
    origin_chi2 = ((delta[0] ** 2 + delta[1] ** 2) /
                   (2.0 * cfg["originTransverseResolution"] ** 2) +
                   delta[2] ** 2 / (2.0 * cfg["originZResolution"] ** 2))
    if origin_chi2 / 3.0 > cfg["maxPairOriginNormalizedChi2"]:
        return None
    fit = common_line_vertex(x1, u1, x2, u2, cfg["commonVertexLineResolution"],
                             cfg["commonVertexBeamLineResolution"], float_math)
    if fit is None:
        return None
    pair_p = p1.astype(float) + p2.astype(float)
    energy = math.sqrt(_dot(p1, p1, float_math) + MUON_MASS_GEV ** 2) + math.sqrt(_dot(p2, p2, float_math) + MUON_MASS_GEV ** 2)
    mass2 = energy * energy - pair_p[0] ** 2 - pair_p[1] ** 2 - pair_p[2] ** 2
    mass = math.sqrt(max(0.0, mass2))
    energy_double = math.sqrt(_dot(p1.astype(float), p1.astype(float)) + MUON_MASS_GEV ** 2) + math.sqrt(_dot(p2.astype(float), p2.astype(float)) + MUON_MASS_GEV ** 2)
    mass_double = math.sqrt(max(0.0, energy_double ** 2 - _dot(pair_p, pair_p)))
    pt = math.hypot(pair_p[0], pair_p[1])
    pt1 = float(first.get("pt", math.hypot(p1[0], p1[1])))
    pt2 = float(second.get("pt", math.hypot(p2[0], p2[1])))
    phi1 = float(first.get("phi", math.atan2(p1[1], p1[0])))
    phi2 = float(second.get("phi", math.atan2(p2[1], p2[0])))
    eta1, eta2 = float(first.get("eta", _eta(p1))), float(second.get("eta", _eta(p2)))
    delta_phi = abs(math.remainder(phi1 - phi2, 2.0 * math.pi))
    delta_eta = abs(eta1 - eta2)
    # Derived histogram shapes rebuild TVector3 in double from stored pt,
    # phi,pz. They do not call the producer's float GlobalVector methods.
    shape_p1 = np.array((pt1 * math.cos(phi1), pt1 * math.sin(phi1), float(p1[2])))
    shape_p2 = np.array((pt2 * math.cos(phi2), pt2 * math.sin(phi2), float(p2[2])))
    momentum1, momentum2 = float(np.linalg.norm(shape_p1)), float(np.linalg.norm(shape_p2))
    score = origin_chi2 / 3.0 + (approach["distance"] / cfg["commonVertexLineResolution"]) ** 2 + fit["chi2"] / fit["ndof"]
    half_chi2 = fit["chi2"] / 2.0
    # Survival probability of chi-square(df=3), without scipy/ROOT.
    probability = (math.erfc(math.sqrt(half_chi2)) +
                   2.0 * math.sqrt(half_chi2 / math.pi) * math.exp(-half_chi2))
    result = {
        "isOS": is_os, "topologyMin": min(int(first["topology"]), int(second["topology"])),
        "topologyMax": max(int(first["topology"]), int(second["topology"])),
        "vx": float(fit["position"][0]), "vy": float(fit["position"][1]),
        "vz": float(fit["position"][2]), "vxErr": float(fit["error"][0]),
        "vyErr": float(fit["error"][1]), "vzErr": float(fit["error"][2]),
        "chi2": fit["chi2"], "ndof": fit["ndof"], "normalizedChi2": fit["chi2"] / fit["ndof"],
        "probability": probability, "vertexFitProbability": probability,
        "dca": approach["distance"], "dcaStatus": 1,
        "dcaX": float(approach["midpoint"][0]), "dcaY": float(approach["midpoint"][1]),
        "dcaZ": float(approach["midpoint"][2]),
        "originCompatibilityChi2": float(origin_chi2),
        "originCompatibilityNormalizedChi2": float(origin_chi2 / 3.0), "score": float(score),
        "mass": mass, "massDouble": mass_double, "pt": pt, "pz": float(pair_p[2]), "eta": _eta(pair_p),
        "phi": math.atan2(pair_p[1], pair_p[0]), "p": float(np.linalg.norm(pair_p)),
        "openingAngle": math.atan2(float(np.linalg.norm(np.cross(shape_p1, shape_p2))), float(shape_p1 @ shape_p2)),
        "deltaPhi": delta_phi, "deltaEta": delta_eta, "deltaR": math.hypot(delta_eta, delta_phi),
        "ptAsymmetry": abs(pt1 - pt2) / (pt1 + pt2) if pt1 + pt2 > 0.0 else math.nan,
        "momentumAsymmetry": abs(momentum1 - momentum2) / (momentum1 + momentum2),
        "minMuonPt": min(pt1, pt2), "maxMuonPt": max(pt1, pt2),
        "vertexR": math.hypot(fit["position"][0], fit["position"][1]),
        "rapidity": math.asinh(pair_p[2] / math.hypot(mass, pt)) if math.hypot(mass, pt) > 0.0 else math.nan,
        "kalmanAttempted": 0, "kalmanValid": 0, "usesLineFallback": 1,
    }
    return result


def select_pairs(muons: Sequence[Mapping], config: Optional[Mapping] = None,
                 require_os: bool = True) -> list:
    """Producer pair ranking, stable ties, non-reuse and max vertex count.

    ``require_os=False`` permits both charge combinations; callers wanting an
    SS-only control must select SS before ranking rather than filter afterward.
    """
    cfg = resolved_config(config)
    cfg["requireOppositeSign"] = bool(require_os)
    choices = []
    for first in range(len(muons)):
        for second in range(first + 1, len(muons)):
            pair = pair_observables(muons[first], muons[second], cfg)
            if pair is not None:
                pair.update(muonIdx1=first, muonIdx2=second)
                choices.append(pair)
    choices.sort(key=lambda pair: pair["score"])
    used = set()
    result = []
    for pair in choices:
        first, second = pair["muonIdx1"], pair["muonIdx2"]
        if first in used or second in used:
            continue
        if cfg["maxDimuonVertices"] > 0 and len(result) >= cfg["maxDimuonVertices"]:
            break
        used.update((first, second))
        result.append(pair)
    return result


CAMPAIGN = "shift_detector_representative_20261007_v10"
SCHEMA = "shift-event-mixing-input-v1"
MUON_FIELDS = (
    "px py pz pt p eta phi mass vx vy vz charge topology recoAlgorithm quality "
    "orientationValid directionFlipped inferredSourceSide genPartIdx hitGenPartIdx "
    "hitSimTrackId hitTruthPurity hitTruthMatchedLayers hitTruthAmbiguousLayers "
    "simTruthMatched simTrackId constrainedValid constrainedStatus constrainedHits "
    "constrainedPt constrainedPz constrainedEta constrainedPhi constrainedPx "
    "constrainedPy constrainedVx constrainedVy constrainedVz constrainedChi2 "
    "constrainedNdof normalizedChi2 chi2 ndof nValidHits nValidMuonHits nMuonStations "
    "nLostHits ptErr etaErr phiErr "
    "nHitsPlusEndcap nHitsMinusEndcap nHitsBarrel nStationsNearEndcap "
    "nStationsBarrel nStationsFarEndcap trackVx trackVy trackVz dz dxy "
    "linePcaR linePcaZ directionalRefitValid directionalRefitUsedPrecisionHits "
    "vertexRefitStatus vertexRefitIdx"
).split()
REQUIRED_MUON_FIELDS = (
    "px py pz pt eta phi vx vy vz charge topology genPartIdx hitGenPartIdx "
    "normalizedChi2 nValidHits nValidMuonHits nMuonStations nLostHits "
    "recoAlgorithm ptErr etaErr phiErr"
).split()
FINITE_MUON_FIELDS = "px py pz pt eta phi vx vy vz charge topology genPartIdx hitGenPartIdx".split()
GEN_FIELDS = "pdgId status genPartIdxMother pt pz eta phi mass vx vy vz".split()
ACTIVITY_FIELDS = (
    "nDTSegments nCSCSegments nDTRecHits nCSCRecHits nRPCRecHits nGEMRecHits "
    "nGEMSegments nGeneralTracks nPixelRecHits nTrackerSelectedTracks"
).split()
WEIGHT_FIELDS = ("shiftSamplingGenWeight", "shiftOriginalGenWeight",
                 "shiftSamplingProbability", "shiftSamplingWeight")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def event_identity(campaign, stratum, event_id):
    """An identity independent of file names, grouping, and extraction order."""
    if len(event_id) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in event_id):
        raise ValueError("Event identity requires three nonnegative integers")
    return f"{campaign}:{stratum}:{event_id[0]}:{event_id[1]}:{event_id[2]}"


def event_partition(identity):
    bucket = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % 100
    return "train" if bucket < 60 else "validation" if bucket < 80 else "test"


def validate_weights(values, ledger):
    expected = (ledger["corrected_weight"], ledger["native_weight"],
                ledger["sampling_probability"], ledger["inverse_sampling_probability"])
    for branch, reference in zip(WEIGHT_FIELDS, expected):
        value = values[branch]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or
                not math.isclose(value, reference, rel_tol=1e-10, abs_tol=1e-300)):
            raise RuntimeError(f"Nano weight differs from frozen sampling ledger: {branch}")
    if not 0. < values["shiftSamplingProbability"] <= 1.:
        raise RuntimeError("Invalid event inclusion probability")
    if not math.isclose(values["shiftSamplingGenWeight"],
                        values["shiftOriginalGenWeight"] / values["shiftSamplingProbability"],
                        rel_tol=1e-10, abs_tol=1e-300):
        raise RuntimeError("Detector weight does not equal native GEN weight / inclusion probability")


def build_inventory(campaign, production_directory=None):
    """Admit complete validated receipts, recording every unavailable QCD job."""
    if campaign != CAMPAIGN:
        raise RuntimeError("Mixing pilot requires the audited V10 natural-decay production")
    root = Path(production_directory or PROJECT / "validation" / ("ntuple_production_" + campaign.split("representative_", 1)[1]))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "shift-gen-to-nano-plan-v1" or
            Path(manifest.get("eos_output", "")).name != campaign or
            manifest.get("complete_muon_decays") is not True or
            "natural branching ratios" not in manifest.get("correction", "")):
        raise RuntimeError("Mixing pilot requires the audited V10 natural-decay production")
    sums = full_gen_weight_sums(campaign, root)
    ledger_path = root / manifest["detector_sampling"]["ledger"]
    plan = json.loads((root / "sampling_plan.json").read_text())
    ledger_digest = sha256(ledger_path)
    if (plan.get("sampling_ledger", {}).get("path") != manifest["detector_sampling"]["ledger"] or
            plan["sampling_ledger"].get("sha256") != ledger_digest):
        raise RuntimeError("Sampling event ledger differs from the frozen sampling plan")
    ledger = defaultdict(list)
    identities = set()
    with gzip.open(ledger_path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if not row["stratum"].startswith("qcd_"):
                continue
            identity = event_identity(campaign, row["stratum"], row["event_id"])
            if identity in identities:
                raise RuntimeError(f"Repeated physics event in sampling ledger: {identity}")
            identities.add(identity)
            ledger[row["source"]].append(row)
    sources = {row["index"]: row for row in manifest["sources"]}
    job_path = root / "all_jobs.txt"
    jobs = [tuple(map(int, line.split())) for line in job_path.read_text().splitlines()]
    if (len(jobs) != manifest["jobs"] or any(len(row) != 4 for row in jobs) or
            {row[3] for row in jobs} != set(range(manifest["jobs"]))):
        raise RuntimeError("Invalid frozen Nano job inventory")
    files, missing = [], []
    planned = Counter()
    for source_index, skip, count, job in sorted(jobs, key=lambda row: row[3]):
        source = sources[source_index]
        stratum = source["stratum"]
        if not stratum.startswith("qcd_"):
            continue
        planned[stratum] += count
        if (skip < 0 or count <= 0 or skip + count > source["events"] or
                len(ledger[source_index]) != source["events"]):
            raise RuntimeError(f"Frozen source event counts differ for job {job}")
        receipt = root / "results" / f"status{job}.json"
        if not receipt.is_file():
            receipt = root / "results" / f"g{job // 500}" / f"status{job}.json"
        if not receipt.is_file():
            missing.append(dict(job=job, stratum=stratum, events=count, reason="missing receipt"))
            continue
        record = json.loads(receipt.read_text())
        if record.get("job") != job or record.get("events") != count or record.get("source_stratum") != stratum:
            raise RuntimeError(f"Receipt identity differs: {receipt}")
        if record.get("complete") is False:
            missing.append(dict(job=job, stratum=stratum, events=count, reason="failed or incomplete receipt"))
            continue
        original_path = str(Path(manifest["eos_output"]) / stratum / f"job{job:07d}" / "nano.root")
        if (record.get("complete") is not True or record.get("exit_code") != 0 or
                record.get("nano_path") != original_path or
                record.get("validated_tier_events") != {tier: count for tier in ("GEN", "SIM", "DIGIHLT", "RECO", "NANO")} or
                not isinstance(record.get("nano_bytes"), int) or record["nano_bytes"] <= 0 or
                not re.fullmatch(r"[0-9a-f]{64}", record.get("report_sha256") or "")):
            raise RuntimeError(f"Unvalidated Nano receipt: {receipt}")
        resolved = resolve_storage_path(original_path)
        if Path(resolved).stat().st_size != record["nano_bytes"]:
            raise RuntimeError(f"Published Nano bytes differ from receipt: {resolved}")
        files.append(dict(job=job, stratum=stratum, events=count, source_index=source_index,
                          source_skip=skip, original_path=original_path, resolved_path=resolved,
                          receipt=str(receipt), receipt_sha256=sha256(receipt),
                          report_sha256=record["report_sha256"], nano_bytes=record["nano_bytes"],
                          expected_events=ledger[source_index][skip:skip + count]))
    if dict(planned) != {name: count for name, count in manifest["strata"].items() if name.startswith("qcd_")}:
        raise RuntimeError("QCD source inventory differs from manifest strata")
    inventory = dict(schema="shift-event-mixing-inventory-v1", campaign=campaign,
                     checked_at_utc=datetime.now(timezone.utc).isoformat(),
                     source_input_complete=not missing, physics_valid=False, normalization_ready=False,
                     manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
                     sampling_plan_sha256=manifest["detector_sampling"]["plan_sha256"],
                     sampling_ledger=str(ledger_path), sampling_ledger_sha256=ledger_digest,
                     job_inventory_sha256=sha256(job_path), planned_events=dict(planned),
                     full_gen_sumw={name: value for name, value in sums.items() if name.startswith("qcd_")},
                     cross_sections_pb=dict(cross_sections["qcd"]),
                     missing_jobs=missing, files=files,
                     notes=["Validated partial Nano inventory; missing jobs remain explicitly excluded.",
                            "Event yield weights use full parent GEN denominators; no mixed-pair rate is assigned.",
                            "Truth fields are separate diagnostics and must not define pools or training features."])
    return inventory


def _scalar(value):
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def truth_chain(index, gen):
    """Keep actual ancestors, skipping no physical resonance or decay link."""
    result, visited = [], set()
    while 0 <= index < len(gen["pdgId"]):
        if index in visited:
            raise RuntimeError("Cyclic GenPart mother chain")
        visited.add(index)
        particle = {name: _scalar(values[index]) for name, values in gen.items()}
        particle["index"] = index
        result.append(particle)
        index = int(particle["genPartIdxMother"])
    return result


def read_nano(row, inventory):
    import awkward as ak
    import uproot
    # Explicit threaded file source avoids mmap-backed EOS stalls.
    with uproot.open(row["resolved_path"], handler=uproot.source.file.MultithreadedFileSource,
                     num_workers=1) as file:
        tree = file["Events"]
        if tree.num_entries != row["events"]:
            raise RuntimeError(f"Published Nano event count differs: {row['resolved_path']}")
        available = set(tree.keys())
        required = {"run", "luminosityBlock", "event", "nShiftMuon", "nShiftDimuonVertex", *WEIGHT_FIELDS}
        required.update("ShiftMuon_" + name for name in REQUIRED_MUON_FIELDS)
        required.update("GenPart_" + name for name in GEN_FIELDS)
        required.update("ShiftDimuonVertex_" + name for name in ("muonIdx1", "muonIdx2", "mass", "chi2", "probability", "dca"))
        if required - available:
            raise RuntimeError(f"Required mixing branches missing: {sorted(required - available)}")
        muon_fields = [name for name in MUON_FIELDS if "ShiftMuon_" + name in available]
        pair_fields = sorted(name[len("ShiftDimuonVertex_"):] for name in available if name.startswith("ShiftDimuonVertex_"))
        activity_fields = [name for name in ACTIVITY_FIELDS if "ShiftRecoDiag_" + name in available]
        names = (list(required) + ["ShiftMuon_" + name for name in muon_fields] +
                 ["ShiftDimuonVertex_" + name for name in pair_fields] +
                 ["ShiftRecoDiag_" + name for name in activity_fields])
        arrays = tree.arrays(sorted(set(names)), library="ak", how=dict)
        events = []
        campaign, stratum = inventory["campaign"], row["stratum"]
        scale = inventory["cross_sections_pb"][stratum.split("_", 1)[1]] / inventory["full_gen_sumw"][stratum]
        for entry, expected in enumerate(row["expected_events"]):
            event_id = [int(arrays[name][entry]) for name in ("run", "luminosityBlock", "event")]
            if event_id != expected["event_id"] or expected["stratum"] != stratum:
                raise RuntimeError(f"Published event identity differs from ledger at job {row['job']}, entry {entry}")
            weights = {name: float(arrays[name][entry]) for name in WEIGHT_FIELDS}
            validate_weights(weights, expected)
            identity = event_identity(campaign, stratum, event_id)
            nmuons, npairs = int(arrays["nShiftMuon"][entry]), int(arrays["nShiftDimuonVertex"][entry])
            reco = {name: ak.to_list(arrays["ShiftMuon_" + name][entry]) for name in muon_fields}
            gen = {name: ak.to_list(arrays["GenPart_" + name][entry]) for name in GEN_FIELDS}
            saved = {name: ak.to_list(arrays["ShiftDimuonVertex_" + name][entry]) for name in pair_fields}
            if any(len(v) != nmuons for v in reco.values()) or any(len(v) != npairs for v in saved.values()):
                raise RuntimeError("Flat-table object counts differ from collection lengths")
            muons = []
            for index in range(nmuons):
                muon = {name: _scalar(v[index]) for name, v in reco.items()}
                muon["index"] = index
                if any(muon[name] is None for name in FINITE_MUON_FIELDS):
                    raise RuntimeError("Nonfinite required muon fit input")
                muon["truth"] = {"gen_match": truth_chain(int(muon["genPartIdx"]), gen),
                                  "hit_match": truth_chain(int(muon["hitGenPartIdx"]), gen)}
                muons.append(muon)
            pairs = [{name: _scalar(v[index]) for name, v in saved.items()} for index in range(npairs)]
            for pair in pairs:
                if any(not 0 <= int(pair[name]) < nmuons for name in ("muonIdx1", "muonIdx2")):
                    raise RuntimeError("Stored dimuon references an unavailable muon")
            events.append(dict(schema=SCHEMA, event_key=identity, event_id=event_id,
                               stratum=stratum, partition=event_partition(identity),
                               source_index=row["source_index"], source_raw_index=expected["raw_index"],
                               nano_job=row["job"], nano_entry=entry,
                               original_gen_weight=weights["shiftOriginalGenWeight"],
                               sampling_probability=weights["shiftSamplingProbability"],
                               sampling_inverse_probability=weights["shiftSamplingWeight"],
                               detector_weight=weights["shiftSamplingGenWeight"],
                               physical_weight_pb=weights["shiftSamplingGenWeight"] * scale,
                               activity={name: _scalar(arrays["ShiftRecoDiag_" + name][entry]) for name in activity_fields},
                               nShiftMuon=nmuons, muons=muons, stored_pairs=pairs))
    return events, dict(muon_fields=muon_fields, pair_fields=pair_fields, activity_fields=activity_fields)


def _read_nano_task(task):
    """Picklable worker with only the normalization metadata it needs."""
    row, normalization = task
    return read_nano(row, normalization)


def iter_pool(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("schema") != SCHEMA:
                raise RuntimeError("Unknown event mixing input schema")
            yield row


def first_noncopy_parent(chain):
    """Return the first different-PDG ancestor of a matched generator muon."""
    if not chain or abs(chain[0]["pdgId"]) != 13:
        return None
    pdg_id = chain[0]["pdgId"]
    return next((row for row in chain[1:] if row["pdgId"] != pdg_id), None)


def truth_census(events):
    """Raw parent-PDG census of reconstructed objects, separate from features.

    Exact generator indices identify a common mother within each event. Equal
    parent PDG codes alone do not establish that two muons share a parent.
    """
    result = {match: {"muon_first_noncopy_parent_abs_pdg": Counter(),
                      "muon_ancestor_abs_pdg": Counter(), "inclusive_pairs": Counter(),
                      "both_both_pairs": Counter()} for match in ("gen_match", "hit_match", "origin_match")}
    def chain_for(muon, match):
        if match == "origin_match":
            # Reproduce the existing MC-only origin histogram's explicit
            # angular fallback; hit truth remains separate for closure.
            match = "hit_match" if muon.get("hitGenPartIdx", -1) >= 0 else "gen_match"
        return muon["truth"][match]
    event_count = 0
    for event in events:
        event_count += 1
        for match, counts in result.items():
            for muon in event["muons"]:
                chain = chain_for(muon, match)
                parent = first_noncopy_parent(chain)
                counts["muon_first_noncopy_parent_abs_pdg"][str(abs(parent["pdgId"])) if parent else "unmatched_or_no_parent"] += 1
                for pdg_id in {abs(row["pdgId"]) for row in chain}:
                    counts["muon_ancestor_abs_pdg"][str(pdg_id)] += 1
            for pair in event["stored_pairs"]:
                first = chain_for(event["muons"][pair["muonIdx1"]], match)
                second = chain_for(event["muons"][pair["muonIdx2"]], match)
                a, b = first_noncopy_parent(first), first_noncopy_parent(second)
                if first and second and first[0]["index"] == second[0]["index"]:
                    category = "same_matched_gen_particle"
                elif a is None or b is None:
                    category = "unmatched_or_no_parent"
                elif a["index"] == b["index"]:
                    category = "common_parent_abs_pdg_" + str(abs(a["pdgId"]))
                else:
                    category = "different_parents"
                counts["inclusive_pairs"][category] += 1
                if pair["topologyMin"] == 2 and pair["topologyMax"] == 2:
                    counts["both_both_pairs"][category] += 1
    return dict(schema="shift-reconstructed-qcd-parent-pdg-census-v1", events=event_count,
                counts={match: {key: dict(value) for key, value in counts.items()} for match, counts in result.items()},
                notes=["Raw reconstructed-object counts, not parent production rates or branching fractions.",
                       "Ancestry and exact parent indices are truth diagnostics only.",
                       "origin_match reproduces the MC origin histogram's hit-first, angular-fallback convention.",
                       "Different parents do not establish statistical independence.",
                       "Muon ancestor counts count each PDG code at most once per muon."])


def extract_pool(output_directory, campaign=CAMPAIGN, production_directory=None, workers=8, executor_kind="process"):
    inventory = build_inventory(campaign, production_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pool = output / "qcd_events.jsonl.gz"
    partial = pool.with_name(pool.name + ".partial")
    statistics = defaultdict(Counter)
    seen, schema = set(), None
    rows = inventory["files"]
    normalization = {key: inventory[key] for key in ("campaign", "cross_sections_pb", "full_gen_sumw")}
    executor_class = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
    with gzip.open(partial, "wt", compresslevel=6) as stream, executor_class(max_workers=workers) as executor:
        # Independent processes avoid the Python GIL during ROOT metadata
        # decoding. Never pickle the full input/receipt inventory per task.
        results = executor.map(_read_nano_task, ((row, normalization) for row in rows), chunksize=4)
        for index, (events, file_schema) in enumerate(results):
            if schema is not None and file_schema != schema:
                raise RuntimeError("Nano object schemas differ between admitted files")
            schema = file_schema
            for event in events:
                if event["event_key"] in seen:
                    raise RuntimeError(f"Repeated physics event in Nano pool: {event['event_key']}")
                seen.add(event["event_key"])
                stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
                counts = statistics[event["stratum"]]
                counts["events"] += 1
                counts["empty_events"] += event["nShiftMuon"] == 0
                counts["one_muon_events"] += event["nShiftMuon"] == 1
                counts["muons"] += event["nShiftMuon"]
                counts["both_endcaps_muons"] += sum(muon["topology"] == 2 for muon in event["muons"])
                counts["stored_pairs"] += len(event["stored_pairs"])
                counts["both_both_stored_pairs"] += sum(pair["topologyMin"] == 2 and pair["topologyMax"] == 2 for pair in event["stored_pairs"])
                counts["detector_sumw"] += event["detector_weight"]
                counts["physical_sumw_pb"] += event["physical_weight_pb"]
                counts["partition_" + event["partition"] + "_events"] += 1
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(f"Read {index + 1}/{len(rows)} Nano files; {len(seen)} events", flush=True)
    partial.replace(pool)
    inventory["statistics"] = {name: dict(value) for name, value in statistics.items()}
    inventory["branch_schema"] = schema
    inventory["pool"] = str(pool)
    inventory["pool_sha256"] = sha256(pool)
    inventory["partition_prescription"] = "sha256(campaign:stratum:run:lumi:event) first 8 bytes modulo 100; train<60, validation<80, otherwise test"
    inventory["reader_execution"] = dict(executor=executor_kind, workers=workers, ordered=True)
    # The frozen expected ledger is checked during extraction and remains an
    # independent authority; duplicating it in the final receipt adds no value.
    for row in inventory["files"]:
        del row["expected_events"]
    (output / "input_inventory.json").write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    print(json.dumps(inventory["statistics"], indent=2), flush=True)
    return inventory



CLASSIFIER = PROJECT / "CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier"
MASS_EDGES = np.asarray(
    list(np.arange(0., 5.001, .1)) + list(np.arange(5.5, 20.001, .5))
    + list(np.arange(22., 60.001, 2.)) + list(np.arange(65., 120.001, 5.))
)
MASS_BANDS = ((0., 2.), (2., 4.), (4., 10.), (10., 30.), (30., 70.), (70., 110.))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported audit value: {type(value).__name__}")
    path.write_text(json.dumps(record, indent=2, allow_nan=False, default=scalar) + "\n")


def validate_inventory(pool, inventory_path, events):
    """Tie the exact input bytes, identities and counts to extraction receipts."""
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema") != "shift-event-mixing-inventory-v1":
        raise ValueError("Unknown mixing input inventory schema")
    pool_hash = digest(pool)
    if inventory.get("pool_sha256") != pool_hash:
        raise ValueError("Input pool hash differs from extraction inventory")
    observed = defaultdict(Counter)
    for event in events:
        identity = event_identity(inventory["campaign"], event["stratum"], event["event_id"])
        if event["event_key"] != identity or event["partition"] != event_partition(identity):
            raise ValueError("Input event identity or partition differs from extraction prescription")
        observed[event["stratum"]]["events"] += 1
        observed[event["stratum"]]["muons"] += len(event["muons"])
    expected = inventory["statistics"]
    if set(observed) != set(expected) or any(
            counts[key] != expected[stratum][key]
            for stratum, counts in observed.items() for key in ("events", "muons")):
        raise ValueError("Input event or muon counts differ from extraction inventory")
    return dict(path=str(inventory_path.resolve()), sha256=digest(inventory_path),
                pool_sha256=pool_hash, campaign=inventory["campaign"],
                source_input_complete=inventory["source_input_complete"],
                missing_jobs=inventory["missing_jobs"],
                sampling_plan_sha256=inventory["sampling_plan_sha256"],
                sampling_ledger_sha256=inventory["sampling_ledger_sha256"],
                full_gen_sumw=inventory["full_gen_sumw"],
                cross_sections_pb=inventory["cross_sections_pb"])


def activity_bin(event):
    """Fixed bins of reconstructed CSC occupancy; no truth or muon-count cut."""
    value = event["activity"].get("nCSCSegments", -1)
    if value is None or value < 0:
        return "unavailable"
    for upper in (0, 4, 9, 19, 49):
        if value <= upper:
            return str(upper)
    return "50+"


def pools_for(events):
    pools = defaultdict(list)
    seen = set()
    for index, event in enumerate(events):
        key = event["event_key"]
        if key in seen:
            raise ValueError("Duplicate source event: " + key)
        seen.add(key)
        if event["partition"] not in ("train", "validation", "test"):
            raise ValueError("Unknown event partition")
        if not math.isfinite(event["detector_weight"]) or event["detector_weight"] < 0:
            raise ValueError("This positive-weight PoC requires finite nonnegative weights")
        campaign = key.rsplit(":", 4)[0]
        pools[(campaign, event["stratum"], event["partition"], activity_bin(event))].append(index)
    return dict(pools)


def hit_truth(muon):
    truth = muon.get("truth", {}).get("hit_match", [])
    purity = muon.get("hitTruthPurity", 0.)
    if (not truth or abs(truth[0]["pdgId"]) != 13 or truth[0]["status"] != 1
            or purity is None or not math.isfinite(purity) or purity < .75
            or muon.get("hitTruthMatchedLayers", 0) < 3):
        return None
    return truth[0]


def label_pair(first, second, same_event):
    """MC annotations only. Cross-event coincident prompt muons stay unknown."""
    a, b = hit_truth(first), hit_truth(second)
    if a is None or b is None:
        return -1, "unknown_match"
    if same_event and (a["index"] == b["index"] or (
            first.get("hitSimTrackId", -1) >= 0
            and first["hitSimTrackId"] == second.get("hitSimTrackId", -2))):
        return 0, "same_muon_twice"
    if any(a[k] is None or b[k] is None for k in ("vx", "vy", "vz")):
        return -1, "unknown_vertex"
    separation = math.sqrt(sum((a[k] - b[k]) ** 2 for k in ("vx", "vy", "vz")))
    if not math.isfinite(separation):
        return -1, "unknown_vertex"
    if separation >= .1:
        return 0, "different_vertices"
    if separation <= .01:
        return (1, "common_vertex") if same_event else (-1, "mixed_coincident_vertices")
    return -1, "vertex_tolerance_gap"


def common_parent(first, second):
    """Exact first non-muon ancestor, not the coarse plotting category."""
    first_match = first.get("truth", {}).get("hit_match", [])
    second_match = second.get("truth", {}).get("hit_match", [])
    if first_match and second_match and first_match[0]["index"] == second_match[0]["index"]:
        return None  # Two reconstructions of one true muon are not a decay pair.
    if (first.get("hitSimTrackId", -1) >= 0
            and first["hitSimTrackId"] == second.get("hitSimTrackId", -2)):
        return None
    ancestors = []
    for muon in (first, second):
        chain = muon.get("truth", {}).get("hit_match", [])
        ancestors.append(next((a for a in chain if abs(a["pdgId"]) != 13), None))
    a, b = ancestors
    return int(a["pdgId"]) if a and b and a["index"] == b["index"] else None


def select_ss(muons, config):
    """Separate like-sign control: same analytic cuts and best-pair convention."""
    candidates = []
    setting = dict(config, requireOppositeSign=False)
    for first in range(len(muons)):
        for second in range(first + 1, len(muons)):
            if muons[first]["charge"] != muons[second]["charge"]:
                continue
            pair = pair_observables(muons[first], muons[second], setting)
            if pair is not None:
                candidates.append(dict(pair, muonIdx1=first, muonIdx2=second))
    candidates.sort(key=lambda pair: pair["score"])
    retained, used = [], set()
    for pair in candidates:
        indices = {pair["muonIdx1"], pair["muonIdx2"]}
        if indices & used:
            continue
        if config["maxDimuonVertices"] and len(retained) >= config["maxDimuonVertices"]:
            break
        retained.append(pair)
        used.update(indices)
    return retained


def same_event_closure(events, config):
    """Check candidate membership and all fit/kinematic classifier inputs."""
    fields = ("mass", "pt", "pz", "eta", "phi", "vx", "vy", "vz",
              "vxErr", "vyErr", "vzErr", "chi2", "normalizedChi2", "probability",
              "dca", "originCompatibilityNormalizedChi2")
    maximum = defaultdict(float)
    failures = []
    mismatched_events = mismatched_fields = checked_pairs = 0
    for event in events:
        calculated = select_pairs(event["muons"], config)
        stored = event["stored_pairs"]
        indices = lambda rows: [(int(p["muonIdx1"]), int(p["muonIdx2"])) for p in rows]
        if indices(calculated) != indices(stored):
            mismatched_events += 1
            if len(failures) < 30:
                failures.append(dict(event=event["event_key"], stored=indices(stored), calculated=indices(calculated)))
            continue
        for calculated_pair, stored_pair in zip(calculated, stored):
            checked_pairs += 1
            for field in fields:
                if field not in stored_pair:
                    raise ValueError("Missing stored closure field: " + field)
                a, b = float(calculated_pair[field]), float(stored_pair[field])
                difference = abs(a - b)
                maximum[field] = max(maximum[field], difference)
                # Persisted input floats and Nano field precision bound closure.
                absolute = .05 if field.startswith("v") else .002
                if not math.isclose(a, b, rel_tol=.002, abs_tol=absolute):
                    mismatched_fields += 1
                    if len(failures) < 30:
                        failures.append(dict(event=event["event_key"], field=field, stored=b, calculated=a))
    return dict(passed=checked_pairs > 0 and mismatched_events == 0 and mismatched_fields == 0,
                events=len(events), checked_pairs=checked_pairs,
                membership_mismatches=mismatched_events, field_mismatches=mismatched_fields,
                max_absolute_difference=dict(maximum), examples=failures,
                tolerances=dict(relative=.002, vertex_absolute_cm=.05, other_absolute=.002))


def pair_record(pair, first, second, anchor, donor, anchor_index, donor_index, weight, kind):
    label, reason = label_pair(first, second, anchor_index == donor_index)
    return dict(pair=pair, muons=[first, second], anchor_event=anchor_index,
                donor_event=donor_index, source_events=[anchor["event_key"], donor["event_key"]],
                source_muons=[first["index"], second["index"]], stratum=anchor["stratum"],
                partition=anchor["partition"], activity_bin=activity_bin(anchor),
                weight=weight, kind=kind, truth_label=label, truth_reason=reason,
                common_parent_pdg=common_parent(first, second) if anchor_index == donor_index else None,
                both_endcaps=int(first["topology"]) == 2 and int(second["topology"]) == 2)


def same_event_rows(events, config):
    rows = []
    for index, event in enumerate(events):
        for kind, pairs in (("same_os", select_pairs(event["muons"], config)),
                            ("same_ss", select_ss(event["muons"], config))):
            for pair in pairs:
                first, second = (event["muons"][int(pair[k])] for k in ("muonIdx1", "muonIdx2"))
                rows.append(pair_record(pair, first, second, event, event, index, index,
                                        event["physical_weight_pb"], kind))
    return rows


def event_replacement(events, config, partners, rng):
    """Positive legs of event i + negative legs of a distinct comparable event j.

    For uniform donor events, W_j/mean(W_donors) is the importance correction.
    Multiply one parent event's sigma/S_GEN, never two cross sections. The
    estimated rate remains conditional on factorization and control closure.
    """
    rows = []
    accounting = Counter()
    for indices in pools_for(events).values():
        total_weight = sum(events[i]["detector_weight"] for i in indices)
        for position, index in enumerate(indices):
            anchor = events[index]
            if len(indices) < 2:
                accounting["events_without_donor_pool"] += 1
                accounting["excluded_physical_weight_pb/" + anchor["stratum"]] += anchor["physical_weight_pb"]
                continue
            mean_weight = (total_weight - anchor["detector_weight"]) / (len(indices) - 1)
            if mean_weight <= 0:
                raise ValueError("Empty donor weight normalization")
            positive = [muon for muon in anchor["muons"] if muon["charge"] > 0]
            donors = rng.integers(0, len(indices) - 1, size=partners)
            for choice in donors:
                donor_index = indices[int(choice) + int(choice >= position)]
                donor = events[donor_index]
                accounting["mock_events"] += 1
                negative = [muon for muon in donor["muons"] if muon["charge"] < 0]
                if not positive or not negative:
                    accounting["mock_events_with_no_os_candidate"] += 1
                    continue
                muons = positive + negative
                weight = anchor["physical_weight_pb"] * donor["detector_weight"] / mean_weight / partners
                for pair in select_pairs(muons, config):
                    first, second = (muons[int(pair[k])] for k in ("muonIdx1", "muonIdx2"))
                    rows.append(pair_record(pair, first, second, anchor, donor, index, donor_index,
                                            weight, "event_os"))
    return rows, dict(accounting)


def row_augmentation(events, config, partners, rng):
    """Uniform partner rows, weighted shapes only; no absolute rate assigned."""
    rows = []
    accounting = Counter()
    setting = dict(config, requireOppositeSign=False)
    for indices in pools_for(events).values():
        bank = [(index, muon) for index in indices for muon in events[index]["muons"]]
        for anchor_index, first in bank:
            eligible = [(index, second) for index, second in bank if index != anchor_index]
            if not eligible:
                accounting["muons_without_partners"] += 1
                continue
            for choice in rng.integers(0, len(eligible), size=partners):
                donor_index, second = eligible[int(choice)]
                accounting["trial_pairs"] += 1
                pair = pair_observables(first, second, setting)
                if pair is None:
                    continue
                pair = dict(pair, muonIdx1=0, muonIdx2=1)
                anchor, donor = events[anchor_index], events[donor_index]
                weight = anchor["detector_weight"] * donor["detector_weight"] * len(eligible) / partners
                kind = "row_os" if first["charge"] != second["charge"] else "row_ss"
                rows.append(pair_record(pair, first, second, anchor, donor, anchor_index, donor_index, weight, kind))
    return rows, dict(accounting)


def histogram(rows, weight_name="weight"):
    masses = np.asarray([r["pair"]["mass"] for r in rows])
    weights = np.asarray([r[weight_name] for r in rows])
    return np.histogram(masses, MASS_EDGES, weights=weights)[0]


def event_bootstrap(rows, event_count, repetitions, rng, events=None):
    """Source-event Poisson multipliers, optionally resampling donor normalization.

    Event-replacement rows use the empirical weighted donor distribution. Its
    sum of weights is resampled along with source events on the fixed graph.
    Random-partner Monte Carlo variation still requires separate seed trials.
    """
    if not rows or repetitions < 2:
        return np.zeros(len(MASS_EDGES) - 1)
    first = np.asarray([r["anchor_event"] for r in rows], dtype=int)
    second = np.asarray([r["donor_event"] for r in rows], dtype=int)
    mass = np.asarray([r["pair"]["mass"] for r in rows])
    weights = np.asarray([r["weight"] for r in rows])
    same = first == second
    if events is not None:
        groups = np.zeros(event_count, dtype=int)
        pools = pools_for(events)
        for group, indices in enumerate(pools.values()):
            groups[indices] = group
        event_weights = np.asarray([e["detector_weight"] for e in events])
        totals = np.bincount(groups, weights=event_weights, minlength=len(pools))
        original_donor_sum = totals[groups[first]] - event_weights[first]
    samples = []
    for _ in range(repetitions):
        counts = rng.poisson(1., event_count)
        multiplier = counts[first] * np.where(same, 1, counts[second])
        if events is not None:
            totals = np.bincount(groups, weights=counts * event_weights, minlength=len(pools))
            donor_sum = totals[groups[first]] - counts[first] * event_weights[first]
            ratio = np.divide(original_donor_sum, donor_sum, out=np.zeros_like(donor_sum), where=donor_sum > 0)
            multiplier = multiplier * np.where(same, 1., ratio)
        samples.append(np.histogram(mass, MASS_EDGES, weights=weights * multiplier)[0])
    return np.std(samples, axis=0, ddof=1)


def coverage(rows):
    result = []
    for low, high in MASS_BANDS:
        selected = [r for r in rows if low <= r["pair"]["mass"] < high]
        contributions = Counter()
        for row in selected:
            for key in set(row["source_events"]):
                contributions[key] += abs(row["weight"])
        result.append(dict(mass_interval_GeV=[low, high], pairs=len(selected),
                           distinct_source_events=len(contributions),
                           largest_event_fraction=max(contributions.values(), default=0.) / sum(abs(r["weight"]) for r in selected)
                           if contributions else None))
    return result


def feature_table(rows):
    """Use the existing data-only classifier extractor; labels are separate."""
    sys.path.insert(0, str(CLASSIFIER))
    from features import MUON_QUALITY, MUON_KINEMATICS, PAIR_QUALITY, PAIR_GEOMETRY, extract_reco
    arrays = defaultdict(list)
    for row in rows:
        arrays["run"].append(0)
        arrays["luminosityBlock"].append(0)
        arrays["event"].append(len(arrays["event"]))
        arrays["nShiftMuon"].append(2)
        arrays["nShiftDimuonVertex"].append(1)
        for field in MUON_QUALITY + MUON_KINEMATICS:
            arrays["ShiftMuon_" + field].append([
                np.nan if muon[field] is None else muon[field] for muon in row["muons"]
            ])
        for field in ("muonIdx1", "muonIdx2", "mass") + PAIR_QUALITY + PAIR_GEOMETRY:
            value = 0 if field == "muonIdx1" else 1 if field == "muonIdx2" else row["pair"][field]
            arrays["ShiftDimuonVertex_" + field].append([value])
    # Empty arrays still have the allowlisted schema.
    from features import RECO_BRANCHES
    for name in RECO_BRANCHES:
        arrays.setdefault(name, [])
    return extract_reco(arrays)


def like_sign_transfer(same, augmented, both=False):
    """MC control diagnostic, not proof of an OS/SS background-rate relation.

    Within each original stratum/fold/occupancy pool, normalize the mixed SS
    shape to the real SS control yield, then transfer its mixed OS/SS ratio.
    Pair-sum shape weights grow quadratically with pool exposure, whereas the
    real event yield grows linearly; a shared stratum scale would distort the
    relative occupancy contributions. OS is never used to choose this scale.
    Pools lacking either control remain explicitly uncovered. Event-level
    competition still differs for row augmentation and needs closure.
    """
    reference = [r for r in same if not both or r["both_endcaps"]]
    extra = [r for r in augmented if not both or r["both_endcaps"]]
    result, predicted = {}, []
    pool_key = lambda row: (row["stratum"], row["partition"], row["activity_bin"])
    pools = sorted({pool_key(row) for row in reference + extra})
    for pool in pools:
        stratum, partition, activity = pool
        selected = [r for r in reference if pool_key(r) == pool]
        mock = [r for r in extra if pool_key(r) == pool]
        real_os = sum(r["weight"] for r in selected if r["kind"] == "same_os")
        real_ss = sum(r["weight"] for r in selected if r["kind"] == "same_ss")
        mixed_os = sum(r["weight"] for r in mock if r["kind"] == "row_os")
        mixed_ss = sum(r["weight"] for r in mock if r["kind"] == "row_ss")
        scale = real_ss / mixed_ss if mixed_ss > 0 and real_ss > 0 else None
        label = "/".join(pool)
        result[label] = dict(stratum=stratum, partition=partition, activity_bin=activity,
                              same_os_pb=real_os, same_ss_pb=real_ss,
                              mixed_os_shape_weight=mixed_os, mixed_ss_shape_weight=mixed_ss,
                              control_available=scale is not None,
                              control_missing_reason=(None if scale is not None else
                                  "no_same_event_ss_yield" if real_ss <= 0 else "no_mixed_ss_support"),
                              uncovered_same_os_pb=real_os if scale is None else 0.,
                              predicted_os_pb=scale * mixed_os if scale is not None else None,
                              prediction_over_same_os=scale * mixed_os / real_os
                              if scale is not None and real_os > 0 else None)
        if scale is not None:
            predicted.extend(dict(r, weight=r["weight"] * scale, like_sign_control_pool=label)
                             for r in mock if r["kind"] == "row_os")
    return result, predicted


def write_rows(path, rows):
    with gzip.open(path, "wt", encoding="utf8") as output:
        for row in rows:
            # Keep truth outside X; muon truth remains only in the MC audit pool.
            record = {key: value for key, value in row.items() if key != "muons"}
            output.write(json.dumps(record, allow_nan=False) + "\n")


def plot_results(output, same, mixed, augmented, events, bootstrap, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    width = np.diff(MASS_EDGES)
    result = {}
    with PdfPages(output / "mixing_diagnostics.pdf") as pdf:
        for both in (False, True):
            select = lambda rows, kind: [r for r in rows if r["kind"] == kind and (not both or r["both_endcaps"])]
            reference = select(same, "same_os")
            factorized = select(mixed, "event_os")
            extra = select(augmented, "row_os")
            key = "Both-Both" if both else "inclusive"
            result[key] = dict(same_event=coverage(reference), event_replacement=coverage(factorized),
                               row_augmentation=coverage(extra))
            transfer, ss_prediction = like_sign_transfer(same, augmented, both=both)
            result[key]["like_sign_transfer_diagnostic"] = transfer
            a, b = histogram(reference), histogram(factorized)
            uncertainty = event_bootstrap(factorized, len(events), bootstrap, rng, events=events)
            centers = (MASS_EDGES[:-1] + MASS_EDGES[1:]) / 2
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for axis, limits in zip(axes[0], ((0, 12), (0, 120))):
                axis.step(centers, a / width, where="mid", label="same event")
                axis.step(centers, b / width, where="mid", label="mixed: independence model")
                if ss_prediction:
                    axis.step(centers, histogram(ss_prediction) / width, where="mid",
                              label="mixed: MC like-sign normalization", alpha=.8)
                axis.fill_between(centers, np.maximum(0, b - uncertainty) / width,
                                  (b + uncertainty) / width, step="mid", alpha=.2,
                                  label="source-event bootstrap only")
                axis.set(xlim=limits, yscale="log", ylabel="model cross section / GeV [pb/GeV]",
                         xlabel="reconstructed dimuon mass [GeV]")
                axis.legend(fontsize=8)
            for kind, label in (("row_os", "mixed OS"), ("row_ss", "mixed SS")):
                rows = select(augmented, kind)
                # Separate strata: native GEN weight scales cannot be multiplied
                # into an inter-stratum physical mixture. Show raw shape coverage.
                counts = np.histogram([r["pair"]["mass"] for r in rows], MASS_EDGES)[0]
                axes[1, 0].step(centers, counts / width, where="mid", label=label)
            axes[1, 0].set(xlim=(0, 120), yscale="log", ylabel="trial pairs / GeV (shape study)",
                           xlabel="reconstructed dimuon mass [GeV]")
            axes[1, 0].legend(fontsize=8)
            bands = result[key]["row_augmentation"]
            axes[1, 1].bar(np.arange(len(bands)), [r["distinct_source_events"] for r in bands])
            axes[1, 1].set_xticks(np.arange(len(bands)))
            axes[1, 1].set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in MASS_BANDS])
            axes[1, 1].set(ylabel="distinct original events", xlabel="mass interval [GeV]")
            fig.suptitle(f"QCD mixing proof of concept: {key}; normalization/closure unvalidated")
            fig.tight_layout(rect=(0, 0, 1, .96))
            pdf.savefig(fig)
            plt.close(fig)
    return result


def write_mixing_histograms(output, same, mixed, augmented, events, bootstrap, rng):
    """Publish ROOT inputs for the main SHIFT plotter, retaining source errors.

    Only event replacement has a source-event bootstrap uncertainty here.
    Other model/shape curves have no assigned error: pair reuse and the SS
    control scale must not be disguised as independent trial statistics.
    """
    from array import array
    import ROOT

    output = Path(output)
    diagnostics, selections = {}, defaultdict(dict)
    errors = {}
    for both in (False, True):
        category = "Both-Both" if both else ""
        selected = lambda rows, kind: [row for row in rows if row["kind"] == kind
                                      and (not both or row["both_endcaps"])]
        reference = selected(same, "same_os")
        factorized = selected(mixed, "event_os")
        extra = selected(augmented, "row_os")
        transfer, control = like_sign_transfer(same, augmented, both=both)
        diagnostics[category or "inclusive"] = dict(
            same_event=coverage(reference), event_replacement=coverage(factorized),
            row_augmentation=coverage(extra), like_sign_transfer_diagnostic=transfer)
        selections["same_event"][category] = reference
        selections["event_replacement"][category] = factorized
        selections["like_sign_transfer"][category] = control
        selections["row_os"][category] = extra
        selections["row_ss"][category] = selected(augmented, "row_ss")
        errors[category] = event_bootstrap(
            factorized, len(events), bootstrap, rng, events=events)

    descriptions = (
        ("same_event", "same event", "model"),
        ("event_replacement", "mixed: independence model", "model"),
        ("like_sign_transfer", "mixed: MC like-sign control", "model"),
        ("row_os", "mixed OS trials", "trials"),
        ("row_ss", "mixed SS trials", "trials"),
    )
    artifacts = []
    for name, label, kind in descriptions:
        path = output / (name + ".root")
        partial = path.with_name(path.name + ".partial")
        file = ROOT.TFile.Open(str(partial), "RECREATE")
        if not file or file.IsZombie():
            raise RuntimeError(f"Cannot create mixing histograms: {partial}")
        try:
            directory = file.mkdir("mixing")
            directory.cd()
            for category, rows in selections[name].items():
                hname = f"ShiftDimuonVertex{category}_mass_{kind}"
                hist = ROOT.TH1D(hname, "normalization/physics closure unvalidated",
                                 len(MASS_EDGES) - 1, array("d", MASS_EDGES))
                masses = np.asarray([row["pair"]["mass"] for row in rows])
                weights = (np.ones(len(rows)) if kind == "trials" else
                           np.asarray([row["weight"] for row in rows]))
                values = np.histogram(masses, MASS_EDGES, weights=weights)[0]
                for index, value in enumerate(values, 1):
                    hist.SetBinContent(index, float(value))
                    hist.SetBinError(index, float(errors[category][index - 1])
                                     if name == "event_replacement" else 0.)
                # Match NumPy's closed final edge, used by the bootstrap above.
                hist.SetBinContent(0, float(weights[masses < MASS_EDGES[0]].sum()))
                hist.SetBinContent(len(MASS_EDGES), float(weights[masses > MASS_EDGES[-1]].sum()))
                hist.SetEntries(len(rows))
                hist.Write()
            file.Write()
        finally:
            file.Close()
        partial.replace(path)
        artifacts.append(dict(
            name=name, label=label, kind=kind, histogram_path=str(path.resolve()),
            sha256=digest(path), uncertainty_scope=(
                "source-event bootstrap on fixed graph; partner variation separate"
                if name == "event_replacement" else "not evaluated")))
    return diagnostics, artifacts


def run_event_mixing(settings):
    """Run the optional main histogrammer stage from its settings dictionary."""
    options = dict(settings)
    defaults = dict(pool=None, inventory=None, output=None, config_json=None,
                    fit=None, partners=20, bootstrap=100, seed=20261009,
                    extract=False, campaign=CAMPAIGN, production_directory=None,
                    input_output=None, workers=8, executor="process", diagnostics=False)
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown eventMixing settings: {sorted(unknown)}")
    defaults.update(options)
    args = SimpleNamespace(**defaults)
    if not isinstance(args.diagnostics, bool):
        raise ValueError("eventMixing diagnostics must be True or False")
    for name, minimum, maximum in (("partners", 1, 100), ("bootstrap", 2, 1000)):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"eventMixing {name} must be {minimum}--{maximum}")
    if args.output is None:
        raise ValueError("eventMixing output must be a new directory")
    if args.config_json is not None and args.fit is not None:
        raise ValueError("Choose eventMixing config_json or fit, not both")
    args.output = Path(args.output)
    fit_settings = (json.loads(Path(args.config_json).read_text())
                    if args.config_json is not None else args.fit)
    config = resolved_config(fit_settings)
    if args.extract:
        if (isinstance(args.workers, bool) or not isinstance(args.workers, int)
                or args.workers < 1 or args.executor not in ("process", "thread")):
            raise ValueError("eventMixing extraction needs positive workers and process/thread executor")
        if args.pool is not None or args.inventory is not None:
            raise ValueError("Choose eventMixing extraction or an existing pool/inventory")
    elif args.pool is None or args.inventory is None:
        raise ValueError("eventMixing needs both pool and inventory, or extract=True")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.extract:
        input_output = Path(args.input_output or args.output / "inputs")
        if input_output != args.output and input_output.exists():
            raise ValueError("eventMixing input_output must be a new directory")
        extracted = extract_pool(input_output, args.campaign, args.production_directory,
                                 args.workers, args.executor)
        args.pool = Path(extracted["pool"])
        args.inventory = input_output / "input_inventory.json"
    else:
        args.pool, args.inventory = Path(args.pool), Path(args.inventory)
    events = list(iter_pool(args.pool))
    if not events:
        raise ValueError("Empty input pool")
    input_provenance = validate_inventory(args.pool, args.inventory, events)
    pools_for(events)  # Identity, fold and weight gates before any mixing.
    sources = (Path(__file__), CLASSIFIER / "features.py",
               Path(__file__).resolve().parents[1] / "configs/shift_histogrammer_config.py",
               Path(__file__).resolve().parents[1] / "configs/shift_sample_paths.py")
    report = dict(schema="shift-event-mixing-poc-v1", physics_valid=False, rate_validated=False,
                  pool=str(args.pool.resolve()), pool_sha256=digest(args.pool), seed=args.seed,
                  partners=args.partners, bootstrap_repetitions=args.bootstrap, config=config,
                  events=len(events), muons=sum(len(e["muons"]) for e in events),
                  input_provenance=input_provenance,
                  pool_definition="campaign/stratum + preassigned event partition + fixed CSC-segment occupancy bins",
                  source_sha256={path.name: digest(path) for path in sources},
                  fit_implementation_provenance=FIT_IMPLEMENTATION_PROVENANCE)
    report["event_multiplicity"] = dict(Counter(str(len(e["muons"])) for e in events))
    report["pool_sizes"] = {"/".join(key): len(indices) for key, indices in pools_for(events).items()}
    report["fit_reproduction_validated"] = False
    report["mixing_physics_closure_validated"] = False
    report["same_event_closure"] = same_event_closure(events, config)
    report["fit_reproduction_validated"] = report["same_event_closure"]["passed"]
    write_json(args.output / "closure.json", report)
    if not report["same_event_closure"]["passed"]:
        raise RuntimeError("Same-event closure failed; no mixed predictions produced. See closure.json")
    rng = np.random.default_rng(args.seed)
    same = same_event_rows(events, config)
    mixed, report["event_replacement_accounting"] = event_replacement(events, config, args.partners, rng)
    augmented, report["augmentation_accounting"] = row_augmentation(events, config, args.partners, rng)
    for name, rows in (("same_event", same), ("event_replacement", mixed), ("row_augmentation", augmented)):
        write_rows(args.output / (name + ".jsonl.gz"), rows)
        report[name] = dict(rows=len(rows), categories=dict(Counter(r["kind"] for r in rows)),
                           truth_reasons=dict(Counter(r["truth_reason"] for r in rows)),
                           partitions=dict(Counter(r["partition"] for r in rows)),
                           mass_min_GeV=min((r["pair"]["mass"] for r in rows), default=None),
                           mass_max_GeV=max((r["pair"]["mass"] for r in rows), default=None),
                           mass_above_plot_range=int(sum(r["pair"]["mass"] >= MASS_EDGES[-1] for r in rows)))
    features = feature_table(augmented)
    np.savez_compressed(args.output / "mixed_reco_features.npz", X=features["X"],
                        feature_names=features["feature_names"], nuisance=features["nuisance"],
                        partitions=np.asarray([r["partition"] for r in augmented]),
                        strata=np.asarray([r["stratum"] for r in augmented]),
                        source_events=np.asarray([r["source_events"] for r in augmented]),
                        source_muons=np.asarray([r["source_muons"] for r in augmented]))
    np.savez_compressed(args.output / "mixed_mc_annotations.npz",
                        label=np.asarray([r["truth_label"] for r in augmented]),
                        reason=np.asarray([r["truth_reason"] for r in augmented]),
                        strata=np.asarray([r["stratum"] for r in augmented]),
                        weight_scope=np.asarray("within_stratum_only_shape_weights"),
                        shape_weight=np.asarray([r["weight"] for r in augmented]))
    bootstrap_state = deepcopy(rng.bit_generator.state)
    report["mass_coverage"], artifacts = write_mixing_histograms(
        args.output, same, mixed, augmented, events, args.bootstrap, rng)
    report["same_event_exact_parent_pdg"] = dict(Counter(
        str(r["common_parent_pdg"]) for r in same if r["kind"] == "same_os" and r["common_parent_pdg"] is not None))
    report["histogram_artifacts"] = artifacts
    if args.diagnostics:
        diagnostic_output = args.output / "diagnostics"
        diagnostic_output.mkdir()
        diagnostic_rng = np.random.default_rng(args.seed)
        diagnostic_rng.bit_generator.state = bootstrap_state
        diagnostic_coverage = plot_results(
            diagnostic_output, same, mixed, augmented, events, args.bootstrap, diagnostic_rng)
        if diagnostic_coverage != report["mass_coverage"]:
            raise RuntimeError("Specialized mixing diagnostics changed coverage accounting")
        report["diagnostics"] = dict(
            enabled=True, source_event_bootstrap_error_band=True,
            uncertainty_scope="source-event bootstrap on fixed graph; partner variation separate",
            artifacts=[dict(path=str(path.resolve()), sha256=digest(path))
                       for path in sorted(diagnostic_output.iterdir()) if path.is_file()])
    report["limitations"] = [
        "Event replacement rate assumes conditional independence; it is not validated by producing smooth plots.",
        "Correlated same-event backgrounds, true resonances, DY and detector/pattern failures remain separate components.",
        "Row augmentation is shape-only; pairs sharing source events are not independent statistics.",
        "Event bootstrap resamples source reuse and donor normalization on the fixed graph; random-partner variation is separate.",
        "Row shape weights apply within each stratum; their scales do not define a physical mixture across bins.",
        "Like-sign transfer is an MC control diagnostic using SS only; finite-control error, correlations and pair competition need validation.",
        "Mixed coincident prompt vertices are unknown training labels; truth never enters the feature matrix.",
        "Mixing cannot reproduce same-event shared-hit failures or reconstruction competition between its two tracks.",
        "No joint-refitted or target-constrained pair result is synthesized.",
        "Published input subset and incomplete upstream production do not establish final background normalization.",
    ]
    report["elapsed_seconds"] = time.monotonic() - started
    report_path = args.output / "report.json"
    write_json(report_path, report)
    write_json(args.output / "mixing_plot_inputs.json", dict(
        schema="shift-event-mixing-plot-inputs-v1", complete=True,
        physics_valid=False, rate_validated=False, fit_reproduction_validated=True,
        report=str(report_path.resolve()), report_sha256=digest(report_path), samples=artifacts,
        normalization="already weighted model pb; row trials are shape-only",
        categories=["", "Both-Both"], mass_edges_GeV=MASS_EDGES.tolist()))
    print(json.dumps({key: report[key] for key in (
        "events", "muons", "same_event", "event_replacement", "row_augmentation", "elapsed_seconds")}))
    return report
