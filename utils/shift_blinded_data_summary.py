#!/usr/bin/env python3
# Code style: AI-code. AI-generated blinded data handling utility.
"""Release only aggregate canonical-mass counts inside an explicit J/psi policy.

This is deliberately independent of the ordinary, unblinded MC histogrammer.
Restricted input ROOT files are never rewritten and no candidate rows, event
lists, constituent muons, alternate masses or classifier values are exported.
The input-kind declaration is provenance, not access control; both modes use
the same mass gate. Collision-data qualification additionally pins this source.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile


MASS_BRANCH = "ShiftDimuonVertex_mass"
COUNT_BRANCH = "nShiftDimuonVertex"
SUMMARY_SCHEMA = "shift-blinded-data-summary-v2"
MANIFEST_SCHEMA = "shift-data-input-provenance-v2"
POLICY_KEYS = {
    "schema", "version", "blinded", "unit", "mass_branch", "mass_min_gev",
    "mass_max_gev", "histogram_bins", "endpoints",
}
SUMMARY_KEYS = {
    "schema", "input_kind", "policy", "policy_sha256", "qualification_sha256",
    "unique_processed_events", "events_with_allowed_pair", "allowed_pairs",
    "mass_bin_edges_gev", "mass_bin_counts", "processed_identity_sha256",
    "input_provenance", "input_provenance_sha256", "duplicate_events_skipped",
}


class BlindingError(ValueError):
    """A controlled failure whose message contains no event or candidate data."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def executable_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def check_reference(reference):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str)
            or not isinstance(reference["sha256"], str) or len(reference["sha256"]) != 64):
        raise BlindingError("Missing commissioning evidence reference")
    try:
        payload = Path(reference["path"]).read_bytes()
        measured = hashlib.sha256(payload).hexdigest()
    except OSError:
        raise BlindingError("Commissioning evidence is unavailable") from None
    if measured != reference["sha256"]:
        raise BlindingError("Commissioning evidence differs from its qualified bytes")
    return payload


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != POLICY_KEYS:
        raise BlindingError("Missing or unsupported blinding policy fields")
    if (policy["schema"] != "shift-data-blinding-policy-v1"
            or not isinstance(policy["version"], str) or not policy["version"].strip()
            or policy["blinded"] is not True or policy["unit"] != "candidate"
            or policy["mass_branch"] != MASS_BRANCH or policy["endpoints"] != "inclusive"):
        raise BlindingError("Unsupported blinding policy")
    low, high = policy["mass_min_gev"], policy["mass_max_gev"]
    if (type(low) not in (int, float) or type(high) not in (int, float)
            or not math.isfinite(low) or not math.isfinite(high) or not 0 < low < high):
        raise BlindingError("Blinding mass bounds must be finite, positive and ordered")
    bins = policy["histogram_bins"]
    if type(bins) is not int or not 1 <= bins <= 10000:
        raise BlindingError("Invalid blinded histogram bin count")
    return dict(policy)


def validate_qualification(policy, input_kind, qualification, provenance_sha256):
    if input_kind not in ("mc-as-data", "collision-data"):
        raise BlindingError("An explicit supported input kind is required")
    if input_kind == "mc-as-data":
        return
    required = {
        "schema", "policy_sha256", "mc_as_data_passed", "independent_review_passed",
        "executable_sha256", "input_provenance_sha256", "purpose",
        "approved_mass_min_gev", "approved_mass_max_gev", "authorization", "mc_response_evidence",
    }
    if (not isinstance(qualification, dict) or set(qualification) != required
            or qualification["schema"] != "shift-data-blinding-qualification-v2"
            or qualification["purpose"] != "jpsi-control-commissioning"
            or type(qualification["approved_mass_min_gev"]) not in (int, float)
            or type(qualification["approved_mass_max_gev"]) not in (int, float)
            or qualification["approved_mass_min_gev"] != policy["mass_min_gev"]
            or qualification["approved_mass_max_gev"] != policy["mass_max_gev"]
            or qualification["policy_sha256"] != digest_json(policy)
            or qualification["executable_sha256"] != executable_sha256()
            or qualification["input_provenance_sha256"] != provenance_sha256
            or not isinstance(provenance_sha256, str) or len(provenance_sha256) != 64
            or any(qualification[key] is not True for key in (
                "mc_as_data_passed", "independent_review_passed"))):
        raise BlindingError("Collision data require qualification of this exact blinding policy")
    # Commissioning authorizes this band only; it does not claim adequate
    # J/psi tail acceptance, a yield fit, or physical normalization.
    try:
        authorization = json.loads(check_reference(qualification["authorization"]))
    except ValueError:
        raise BlindingError("Commissioning authorization is not valid JSON") from None
    if (not isinstance(authorization, dict)
            or authorization.get("schema") != "shift-data-commissioning-authorization-v1"
            or authorization.get("purpose") != qualification["purpose"]
            or authorization.get("mass_branch") != policy["mass_branch"]
            or authorization.get("endpoints") != policy["endpoints"]
            or type(authorization.get("approved_mass_min_gev")) not in (int, float)
            or type(authorization.get("approved_mass_max_gev")) not in (int, float)
            or authorization["approved_mass_min_gev"] != policy["mass_min_gev"]
            or authorization["approved_mass_max_gev"] != policy["mass_max_gev"]):
        raise BlindingError("Commissioning authorization does not approve this exact mass band")
    check_reference(qualification["mc_response_evidence"])


def identity_key(identity):
    if not isinstance(identity, (tuple, list)) or len(identity) != 3:
        raise BlindingError("Malformed event identity")
    for value, bits in zip(identity, (32, 32, 64)):
        if type(value) is not int or not 0 <= value < 2 ** bits:
            raise BlindingError("Event identity is not an unsigned integer of the required width")
    return tuple(identity)


class BlindedSummary:
    """Keep event identities privately and accumulate only independently gated pairs."""

    def __init__(self, policy, input_kind, qualification=None, provenance_sha256=None):
        self.policy = validate_policy(policy)
        validate_qualification(self.policy, input_kind, qualification, provenance_sha256)
        self._policy_sha256 = digest_json(self.policy)
        self.provenance_sha256 = provenance_sha256
        self.input_kind = input_kind
        self.qualification = json.loads(canonical_json(qualification)) if qualification is not None else None
        self._qualification_sha256 = digest_json(self.qualification) if self.qualification else None
        self.identities = set()
        self.duplicates = 0
        self.events_with_allowed_pair = 0
        self.counts = [0] * self.policy["histogram_bins"]

    def add_event(self, identity, count, masses):
        self.check_policy(check_executable=False)
        key = identity_key(identity)
        if key in self.identities:
            self.duplicates += 1
            return
        if type(count) is not int or count < 0 or count >= 2 ** 32:
            raise BlindingError("Invalid canonical dimuon count")
        if len(masses) != count:
            raise BlindingError("Canonical mass array does not match its count")
        low, high = self.policy["mass_min_gev"], self.policy["mass_max_gev"]
        accepted_bins = []
        for mass in masses:
            if type(mass) not in (int, float) or not math.isfinite(mass) or mass < 0:
                raise BlindingError("Invalid canonical dimuon mass")
            if low <= mass <= high:
                index = min(len(self.counts) - 1, int((mass - low) / (high - low) * len(self.counts)))
                accepted_bins.append(index)
        self.identities.add(key)
        if accepted_bins:
            self.events_with_allowed_pair += 1
        for index in accepted_bins:
            self.counts[index] += 1

    def check_policy(self, check_executable=True):
        validate_policy(self.policy)
        if digest_json(self.policy) != self._policy_sha256:
            raise BlindingError("Blinding policy changed after validation")
        if (digest_json(self.qualification) if self.qualification else None) != self._qualification_sha256:
            raise BlindingError("Blinding qualification changed after validation")
        if check_executable:
            validate_qualification(self.policy, self.input_kind, self.qualification, self.provenance_sha256)

    def result(self, input_provenance):
        self.check_policy()
        low, high = self.policy["mass_min_gev"], self.policy["mass_max_gev"]
        ledger = hashlib.sha256()
        for key in sorted(self.identities):
            ledger.update(struct.pack(">IIQ", *key))
        result = dict(
            schema=SUMMARY_SCHEMA, input_kind=self.input_kind,
            policy=dict(self.policy), policy_sha256=self._policy_sha256,
            qualification_sha256=self._qualification_sha256,
            unique_processed_events=len(self.identities),
            events_with_allowed_pair=self.events_with_allowed_pair,
            allowed_pairs=sum(self.counts), duplicate_events_skipped=self.duplicates,
            mass_bin_edges_gev=[low + (high - low) * i / len(self.counts)
                                for i in range(len(self.counts) + 1)],
            mass_bin_counts=list(self.counts), processed_identity_sha256=ledger.hexdigest(),
            input_provenance=input_provenance, input_provenance_sha256=self.provenance_sha256,
        )
        validate_summary(result)
        return result


def validate_summary(result):
    """Validate nested allowlists as well as the top-level output keys."""
    if not isinstance(result, dict) or set(result) != SUMMARY_KEYS:
        raise BlindingError("Unsupported blinded output schema")
    policy = validate_policy(result["policy"])
    if (result["schema"] != SUMMARY_SCHEMA
            or result["input_kind"] not in ("mc-as-data", "collision-data")
            or result["policy_sha256"] != digest_json(policy)):
        raise BlindingError("Inconsistent blinded output provenance")
    provenance = result["input_provenance"]
    if not isinstance(provenance, list):
        raise BlindingError("Unsupported input provenance schema")
    for source in provenance:
        if (not isinstance(source, dict) or set(source) != {"path", "root_uuid", "entries", "sha256", "bytes"}
                or not isinstance(source["path"], str) or not isinstance(source["root_uuid"], str)
                or type(source["entries"]) is not int or source["entries"] < 0
                or type(source["bytes"]) is not int or source["bytes"] <= 0
                or not isinstance(source["sha256"], str) or len(source["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in source["sha256"])):
            raise BlindingError("Unsupported input provenance schema")
    for name in ("unique_processed_events", "events_with_allowed_pair", "allowed_pairs",
                 "duplicate_events_skipped"):
        if type(result[name]) is not int or result[name] < 0:
            raise BlindingError("Invalid blinded aggregate counter")
    counts = result["mass_bin_counts"]
    if (not isinstance(counts, list) or len(counts) != policy["histogram_bins"]
            or any(type(value) is not int or value < 0 for value in counts)
            or sum(counts) != result["allowed_pairs"]
            or result["events_with_allowed_pair"] > min(result["unique_processed_events"], sum(counts))):
        raise BlindingError("Invalid blinded aggregate histogram")
    low, high = policy["mass_min_gev"], policy["mass_max_gev"]
    expected_edges = [low + (high - low) * i / len(counts) for i in range(len(counts) + 1)]
    if result["mass_bin_edges_gev"] != expected_edges:
        raise BlindingError("Blinded histogram edges differ from the qualified policy")
    for name in ("policy_sha256", "processed_identity_sha256", "qualification_sha256", "input_provenance_sha256"):
        value = result[name]
        if name in ("qualification_sha256", "input_provenance_sha256") and value is None and result["input_kind"] == "mc-as-data":
            continue
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise BlindingError("Invalid blinded output digest")


def checked_leaf(tree, name, types, count_name=None):
    leaf = tree.GetLeaf(name)
    if not leaf or not tree.GetBranch(name) or leaf.GetTypeName() not in types:
        raise BlindingError("Required Nano branch is missing or has an unsupported type")
    count = leaf.GetLeafCount()
    if count_name is None:
        if count or leaf.GetLenStatic() != 1:
            raise BlindingError("Event identity or count branch is not scalar")
    elif not count or count.GetName() != count_name:
        raise BlindingError("Canonical mass branch has an unsupported array count")
    return leaf


def validate_manifest(manifest, paths, input_kind):
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "inputs"}
            or manifest["schema"] != MANIFEST_SCHEMA
            or not isinstance(manifest["inputs"], list)):
        raise BlindingError("An explicit input provenance manifest is required")
    expected = {}
    for source in manifest["inputs"]:
        if (not isinstance(source, dict) or set(source) != {"path", "root_uuid", "origin", "sha256", "bytes"}
                or not isinstance(source["path"], str) or not source["path"]
                or not isinstance(source["root_uuid"], str) or not source["root_uuid"]
                or type(source["bytes"]) is not int or source["bytes"] <= 0
                or not isinstance(source["sha256"], str) or len(source["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in source["sha256"])
                or source["origin"] != input_kind or source["path"] in expected):
            raise BlindingError("Input provenance origin or schema does not match the requested mode")
        expected[source["path"]] = dict(source)
    if set(expected) != set(paths):
        raise BlindingError("Input paths differ from the provenance manifest")
    return expected


def verify_input_bytes(source):
    """Verify staged immutable bytes; a ROOT UUID does not pin file content."""
    digest = hashlib.sha256()
    total = 0
    try:
        with Path(source["path"]).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(chunk)
                digest.update(chunk)
    except OSError:
        raise BlindingError("A staged input cannot be checked against its frozen bytes") from None
    if total != source["bytes"] or digest.hexdigest() != source["sha256"]:
        raise BlindingError("A staged input differs from its frozen bytes")


def root_summary(paths, policy, input_kind, qualification=None, provenance_manifest=None):
    """Preflight every file and read only five allowlisted Nano branches with ROOT."""
    paths = sorted(set(str(path) for path in paths))
    if not paths:
        raise BlindingError("At least one exact input file is required")
    expected = validate_manifest(provenance_manifest, paths, input_kind)
    summary = BlindedSummary(policy, input_kind, qualification, digest_json(provenance_manifest))
    for source in expected.values():
        verify_input_bytes(source)
    try:
        import ROOT
    except ImportError:
        raise BlindingError("The ROOT runtime is unavailable") from None
    ROOT.gROOT.SetBatch(True)
    opened = []
    provenance = []
    try:
        for path in paths:
            source = ROOT.TFile.Open(path, "READ")
            if not source or source.IsZombie() or source.TestBit(ROOT.TFile.kRecovered):
                raise BlindingError("An input ROOT file is unreadable or recovered")
            opened.append(source)
            if source.GetUUID().AsString() != expected[path]["root_uuid"]:
                raise BlindingError("An input ROOT UUID differs from its provenance manifest")
            tree = source.Get("Events")
            if not tree or not tree.InheritsFrom("TTree"):
                raise BlindingError("An input has no supported Events tree")
            checked_leaf(tree, "run", ("UInt_t",))
            checked_leaf(tree, "luminosityBlock", ("UInt_t",))
            checked_leaf(tree, "event", ("ULong64_t",))
            checked_leaf(tree, COUNT_BRANCH, ("UInt_t", "Int_t"))
            checked_leaf(tree, MASS_BRANCH, ("Float_t", "Double_t"), COUNT_BRANCH)
            tree.SetBranchStatus("*", 0)
            for name in ("run", "luminosityBlock", "event", COUNT_BRANCH, MASS_BRANCH):
                tree.SetBranchStatus(name, 1)
            provenance.append(dict(path=path, root_uuid=source.GetUUID().AsString(),
                                   entries=int(tree.GetEntries()), sha256=expected[path]["sha256"],
                                   bytes=expected[path]["bytes"]))
        for source in opened:
            tree = source.Get("Events")
            for index in range(tree.GetEntries()):
                if tree.GetEntry(index) <= 0:
                    raise BlindingError("A required input event could not be read")
                identity = (int(tree.run), int(tree.luminosityBlock), int(tree.event))
                count = int(getattr(tree, COUNT_BRANCH))
                # Duplicates have fixed first-file ownership; no candidate content
                # from a later representation changes the already counted event.
                if identity in summary.identities:
                    summary.add_event(identity, 0, ())
                    continue
                if not 0 <= count < 2 ** 32:
                    raise BlindingError("Invalid canonical dimuon count")
                values = getattr(tree, MASS_BRANCH)
                if len(values) != count:
                    raise BlindingError("Canonical mass array does not match its count")
                summary.add_event(identity, count, [float(value) for value in values])
        for source in expected.values():
            verify_input_bytes(source)
        return summary.result(provenance)
    except BlindingError:
        raise
    except Exception:
        # ROOT/reader exception text may include arbitrary input content.
        raise BlindingError("The restricted ROOT reader failed") from None
    finally:
        for source in opened:
            source.Close()


def write_summary(path, result):
    """Publish one complete allowlisted JSON atomically without replacing evidence."""
    validate_summary(result)
    path = Path(path)
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--input-kind", choices=("mc-as-data", "collision-data"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        policy = json.loads(args.policy.read_text())
        manifest = json.loads(args.provenance_manifest.read_text())
        qualification = json.loads(args.qualification.read_text()) if args.qualification else None
        result = root_summary(args.input, policy, args.input_kind, qualification, manifest)
        write_summary(args.output, result)
    except BlindingError as error:
        print("Blinded summary failed: " + str(error), file=sys.stderr)
        return 1
    except Exception:
        print("Blinded summary failed before publication", file=sys.stderr)
        return 1
    print("Blinded aggregate summary published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
