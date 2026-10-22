# Code style: AI-code. AI-generated validation of separate detector and decay weights.
"""Keep native GEN weights, detector inclusion and decay likelihood separate.

This interface permits one predeclared decay draw per original hard event.
It annotates a caller-owned Nano working copy, without changing GEN exposure
or claiming that the sample is ready for physical normalization.
"""
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct


DETECTOR_BRANCHES = ("shiftOriginalGenWeight", "shiftSamplingProbability",
                     "shiftSamplingWeight", "shiftSamplingGenWeight")
PROPOSAL_BRANCHES = ("shiftDecayProposalDensity", "shiftDecayProposalWeight",
                     "shiftProposalGenWeight")
METADATA_KEY = "ShiftDecayProposalWeights"
WEIGHT_POLICY = "native_gen_weight*decay_proposal_weight/reco_inclusion_probability"


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Need a finite numeric " + label)
    return float(value)


def require_close(value, expected, label):
    value = finite_number(value, label)
    if not math.isclose(value, expected, rel_tol=2.e-12, abs_tol=0.):
        raise ValueError("Incorrect " + label)


def original_identity(values):
    if (not isinstance(values, (list, tuple)) or len(values) != 3 or
            any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values)):
        raise ValueError("Original event identity needs three nonnegative integers")
    return tuple(values)


def source_identity(row):
    name, digest = row.get("source_id"), row.get("original_source_sha256")
    if (not isinstance(name, str) or not name or not isinstance(digest, str) or
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Original source ID and lowercase SHA-256 are required")
    return name, digest


def validate_proposal_rows(rows, source_exposures=None):
    """Validate the complete per-file authority; return explicit Double fields.

    ``sampling_probability`` is the cumulative event inclusion probability.
    ``proposal_density_ratio`` is the full-event q/f, and its declared inverse
    is not a probability: it can exceed one. Rows bind these numbers to the
    original source, event and single draw. Repeated ordinary/biased scouts
    must be kept in separate diagnostic samples.
    Source exposures, when supplied, come from the frozen full cohort receipt.
    They may include zero-admitted sources and must not be inferred from rows.
    """
    if not isinstance(rows, (list, tuple)) or not rows and not source_exposures:
        raise ValueError("Exact event authority and explicit exposure for zero-admission files are required")
    identities, sources, result = set(), {}, []
    for row in rows:
        if "admitted" in row and row["admitted"] is not True:
            raise ValueError("Only admitted originals may receive histogram weights")
        identity = original_identity(row["event_id"])
        if identity in identities:
            raise ValueError("Repeated original event; only one proposal draw is supported")
        identities.add(identity)
        source, digest = source_identity(row)
        if source in sources and sources[source] != digest:
            raise ValueError("One source ID is bound to different original GEN files")
        sources[source] = digest
        draw = row.get("proposal_draw_id")
        if (isinstance(draw, bool) or not isinstance(draw, (str, int)) or
                isinstance(draw, str) and not draw or isinstance(draw, int) and draw < 0):
            raise ValueError("An explicit predeclared proposal draw ID is required")
        native = finite_number(row["native_weight"], "native GEN weight")
        probability = finite_number(row["sampling_probability"], "inclusion probability")
        density = finite_number(row["proposal_density_ratio"], "decay proposal q/f")
        correction = finite_number(row["proposal_correction"], "decay proposal f/q")
        if not 0. < probability <= 1. or density <= 0. or correction <= 0.:
            raise ValueError("Need inclusion probability in (0,1] and positive decay densities")
        if row.get("proposal_kind") == "ordinary":
            require_close(density, 1., "ordinary full-event proposal density")
            require_close(correction, 1., "ordinary full-event proposal correction")
        inverse_probability, detector = 1. / probability, native / probability
        if not all(math.isfinite(v) for v in (inverse_probability, detector, 1. / density)):
            raise ValueError("Detector or decay correction exceeds Double support")
        require_close(correction, 1. / density, "full-event decay likelihood correction")
        require_close(row["sampling_gen_weight"], detector, "unchanged native W/p")
        final = detector * correction
        if not math.isfinite(final) or native != 0. and final == 0.:
            raise ValueError("Final proposal weight overflows or underflows Double support")
        require_close(row["proposal_gen_weight"], final, "single application of decay correction")
        result.append(dict(event_id=list(identity), source_id=source,
            original_source_sha256=digest, proposal_draw_id=draw,
            shiftOriginalGenWeight=native, shiftSamplingProbability=probability,
            shiftSamplingWeight=inverse_probability, shiftSamplingGenWeight=detector,
            shiftDecayProposalDensity=density, shiftDecayProposalWeight=correction,
            shiftProposalGenWeight=final))
    if len(set(sources.values())) != len(sources):
        raise ValueError("An original GEN source is repeated under multiple source IDs")
    if source_exposures is not None:
        if not isinstance(source_exposures, (list, tuple)):
            raise ValueError("Source exposures must be a distinct-source list")
        exposed = {}
        for exposure in source_exposures:
            source, digest = source_identity(exposure)
            kind = exposure.get("denominator_kind", "fixed_source_trials")
            if kind == "fixed_source_trials":
                count = exposure.get("full_source_trials")
                ambiguous = "full_source_events" in exposure
            elif kind == "accepted_gen_events":
                count = exposure.get("full_source_events")
                ambiguous = "full_source_trials" in exposure
            else:
                raise ValueError("Unknown original GEN denominator kind")
            if ambiguous or isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("Full original GEN exposure needs one positive integer denominator")
            if source in exposed or digest in exposed.values():
                raise ValueError("Repeated original source exposure")
            exposed[source] = digest
        if any(exposed.get(source) != digest for source, digest in sources.items()):
            raise ValueError("Full source exposures do not cover the exact selected source identities")
    return result


def _key_payloads(path, root_file):
    """Pin persisted non-Events objects, including Run/Lumi and CMSSW metadata."""
    hashes = {}
    with Path(path).open("rb") as stream:
        for key in root_file.GetListOfKeys():
            if key.GetName() == "Events":
                continue
            stream.seek(int(key.GetSeekKey()))
            payload = stream.read(int(key.GetNbytes()))
            hashes[(key.GetName(), int(key.GetCycle()))] = hashlib.sha256(payload).hexdigest()
    return hashes


def annotate_proposal_weights(path, rows, source_exposures=None, *, expected_nano_sha256):
    """Append validated Double weights to a caller-owned Nano working copy.

    The existing native ``genWeight``, original Double W and detector W/p stay
    unchanged. Missing detector fields require the explicit row authority;
    partially annotated inputs and any existing proposal fields are rejected.
    No material efficiency is applied. Full source exposure is preserved as
    supplied, never obtained by summing copied selected-event Run/Lumi rows.
    The expected Nano hash must come from the upstream receipt that binds this
    exact Nano to its completed GEN graph and original/draw authority. Event
    IDs and native weights alone cannot distinguish ordinary/biased siblings.
    """
    records = validate_proposal_rows(rows, source_exposures)
    if (not isinstance(expected_nano_sha256, str) or len(expected_nano_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in expected_nano_sha256)):
        raise ValueError("The upstream exact Nano SHA-256 is required")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected_nano_sha256:
        raise ValueError("Nano differs from the upstream graph/draw receipt SHA-256")
    import ROOT
    selected = {tuple(row["event_id"]): row for row in records}
    root_file = ROOT.TFile.Open(str(path), "UPDATE")
    if not root_file or root_file.IsZombie() or root_file.TestBit(ROOT.TFile.kRecovered):
        raise ValueError("Cannot annotate an invalid Nano file")
    try:
        tree = root_file.Get("Events")
        if not tree or int(tree.GetEntries()) != len(records):
            raise ValueError("Nano row count differs from the exact event authority")
        if root_file.Get(METADATA_KEY) or any(tree.GetBranch(name) for name in PROPOSAL_BRANCHES):
            raise ValueError("Decay proposal fields already exist; refusing a second correction")
        existing = [bool(tree.GetBranch(name)) for name in DETECTOR_BRANCHES]
        if any(existing) and not all(existing):
            raise ValueError("Nano has only part of the existing detector weight contract")
        if all(existing):
            for name in DETECTOR_BRANCHES:
                if tree.GetLeaf(name).GetTypeName() != "Double_t":
                    raise ValueError("Existing detector weights must be Double: " + name)
        native_leaf = tree.GetLeaf("genWeight")
        if not native_leaf or native_leaf.GetTypeName() not in ("Float_t", "Double_t"):
            raise ValueError("Native genWeight must remain a Float or Double scalar")
        native_type = native_leaf.GetTypeName()
        ordered, seen, native_values = [], set(), []
        for event in tree:
            identity = tuple(int(getattr(event, name)) for name in ("run", "luminosityBlock", "event"))
            if identity not in selected or identity in seen:
                raise ValueError("Nano identities differ from the single-draw original authority")
            seen.add(identity)
            record = selected[identity]
            native = record["shiftOriginalGenWeight"]
            expected_native = struct.unpack("f", struct.pack("f", native))[0] if native_type == "Float_t" else native
            require_close(float(event.genWeight), expected_native, "unchanged native genWeight")
            native_values.append(float(event.genWeight))
            if all(existing):
                for name in DETECTOR_BRANCHES:
                    require_close(float(getattr(event, name)), record[name], "existing " + name)
            ordered.append(record)
        if seen != set(selected):
            raise ValueError("Nano does not cover exactly the original event authority")
        untouched = _key_payloads(path, root_file)
        names = PROPOSAL_BRANCHES if all(existing) else DETECTOR_BRANCHES + PROPOSAL_BRANCHES
        buffers = {name: array("d", [0.]) for name in names}
        branches = {name: tree.Branch(name, buffers[name], name + "/D") for name in names}
        for record in ordered:
            for name in names:
                buffers[name][0] = record[name]
                if branches[name].Fill() < 0:
                    raise ValueError("Could not append proposal weight branch: " + name)
        root_file.cd()
        if tree.Write("", ROOT.TObject.kOverwrite) <= 0:
            raise ValueError("Could not persist proposal weight branches")
        metadata = dict(schema="shift-decay-proposal-weight-annotation-v1", events=len(records),
            weight_policy=WEIGHT_POLICY, weight_branch="shiftProposalGenWeight",
            input_nano_sha256=expected_nano_sha256,
            original_event_ids=[row["event_id"] for row in ordered],
            event_authority_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            source_exposures=source_exposures, one_draw_per_original=True,
            native_weights_unchanged=True, detector_probability_unchanged=True,
            decay_correction_application_count=1, material_survival_correction_applied=False,
            normalization_ready=False, weight_annotation_alone_does_not_establish_physics_normalization=True)
        ROOT.TObjString(json.dumps(metadata, sort_keys=True, allow_nan=False)).Write(METADATA_KEY)
    finally:
        root_file.Close()
    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise ValueError("Cannot read back proposal weights")
    try:
        # The newly added annotation is the sole permitted non-Events key.
        current = {key: digest for key, digest in _key_payloads(path, root_file).items()
                   if key[0] != METADATA_KEY}
        if current != untouched:
            raise ValueError("Persisted Run/Lumi or other native metadata changed")
        tree = root_file.Get("Events")
        for name in DETECTOR_BRANCHES + PROPOSAL_BRANCHES:
            if not tree.GetLeaf(name) or tree.GetLeaf(name).GetTypeName() != "Double_t":
                raise ValueError("Proposal and detector fields must persist as Double")
        for index, event in enumerate(tree):
            if float(event.genWeight) != native_values[index]:
                raise ValueError("Native genWeight changed during annotation")
            for name in DETECTOR_BRANCHES + PROPOSAL_BRANCHES:
                require_close(float(getattr(event, name)), ordered[index][name], "persisted " + name)
    finally:
        root_file.Close()
    return metadata
