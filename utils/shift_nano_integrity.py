# Code style: AI-code. AI-generated complete Nano merge validation.
"""Check every persisted Nano value before retiring the original files.

The inventory follows the reviewed grouped-Nano retirement validator in
``shift_cmssw_workflow/scripts/retire_grouped_dark_photon_chunks.py``.
EDM object metadata is compared as exact uncompressed baskets. A merge that
repackages those objects is conservatively rejected, even if it might be
equivalent with native CMSSW dictionaries. Nothing is deleted by this module.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


REQUIRED_TREES = ("Events", "Runs", "LuminosityBlocks", "MetaData", "ParameterSets")
EDM_METADATA_TREES = ("MetaData", "ParameterSets")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def semantic_inventory(paths):
    """Hash ordered typed values across input file boundaries, including EDM."""
    import awkward as ak
    import uproot

    inventory, hashes, classes = {}, {}, None
    for path in paths:
        with uproot.open(path, handler=uproot.source.file.MemmapSource) as source:
            observed = source.classnames(cycle=False)
            if classes is None:
                classes = observed
            require(classes == observed, "Nano ROOT key classes differ")
            for name, classname in observed.items():
                if classname == "TObjString":
                    value = str(source[name])
                    if name in inventory:
                        require(inventory[name]["value"] == value, "Nano string metadata changed")
                    inventory[name] = dict(classname=classname, value=value)
                    continue
                require(classname == "TTree", "Unsupported Nano object class: " + classname)
                tree = source[name]
                types = tree.typenames()
                if name not in inventory:
                    inventory[name] = dict(classname=classname, entries=0, branches={})
                    for branch, typename in types.items():
                        mode = ("ordered-uncompressed-baskets" if name in EDM_METADATA_TREES
                                else "ordered-values")
                        inventory[name]["branches"][branch] = dict(typename=typename, mode=mode)
                        hashes[name, branch] = hashlib.sha256()
                require(types == {key: value["typename"]
                                  for key, value in inventory[name]["branches"].items()},
                        "Nano branch schema changed: " + name)
                inventory[name]["entries"] += tree.num_entries
                if name in EDM_METADATA_TREES:
                    for branch in types:
                        node = tree[branch]
                        for index in range(node.num_baskets):
                            basket = node.basket(index)
                            offsets = basket.byte_offsets
                            record = dict(entries=basket.num_entries,
                                payload_sha256=hashlib.sha256(basket.data.tobytes()).hexdigest(),
                                offsets_sha256=(hashlib.sha256(offsets.tobytes()).hexdigest()
                                                if offsets is not None else None))
                            hashes[name, branch].update(json.dumps(record, sort_keys=True).encode() + b"\n")
                    continue
                # Direct branch reads avoid repeated parsing of thousands of
                # branch names and keep one branch in memory at a time.
                nodes = dict(tree.iteritems(recursive=True))
                require(set(nodes) == set(types), "Nano branch objects differ from the typed schema")
                for branch, node in nodes.items():
                    values = node.array(library="ak", array_cache=None)
                    for value in ak.to_list(values):
                        encoded = json.dumps(value, separators=(",", ":"), allow_nan=True).encode()
                        hashes[name, branch].update(encoded + b"\n")
    require(inventory, "No Nano objects")
    for name in REQUIRED_TREES:
        require(name in inventory and inventory[name]["classname"] == "TTree",
                "Missing retained Nano tree: " + name)
    for (name, branch), digest in hashes.items():
        inventory[name]["branches"][branch]["sha256"] = digest.hexdigest()
    return inventory


def compare_all_branches(inputs, merged):
    """Return a compact proof tied to the exact input and replacement bytes."""
    expected, actual = semantic_inventory(inputs), semantic_inventory([merged])
    require(expected == actual, "Merged Nano changed branch values, rows, schema or EDM metadata baskets")
    compact = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    return dict(schema="shift-nano-all-branch-equivalence-v1", complete=True,
        events=actual["Events"]["entries"],
        event_branches=len(actual["Events"]["branches"]),
        trees={name: dict(entries=value["entries"], branches=len(value["branches"]))
               for name, value in actual.items() if value["classname"] == "TTree"},
        all_branch_inventory_sha256=hashlib.sha256(compact).hexdigest(),
        input_sha256=[sha256(path) for path in inputs], merged_sha256=sha256(merged),
        edm_metadata_baskets_preserved=True, edm_metadata_values_preserved=True)


def compare_replay_values(baseline, replay, *, weight_annotations=None):
    """Require identical ordered data values, allowing fresh EDM provenance.

    A replay creates a new process/output history. Its MetaData/ParameterSets
    baskets must exist, but are not a physics-value equivalence criterion.
    Every other object remains exact. With explicit original worker annotations,
    only their pre-annotation file hashes may differ: fresh EDM provenance changes
    those bytes too. All normalization fields and typed data still match exactly.
    Callers must bind the worker reports to the published files independently.
    This proof cannot authorize retiring either original file.
    """
    expected = semantic_inventory([baseline])
    actual = semantic_inventory([replay])
    for inventory in (expected, actual):
        for name in EDM_METADATA_TREES:
            require(inventory[name]["entries"] > 0 and inventory[name]["branches"],
                    "Replay Nano has empty EDM metadata: " + name)
    expected_data = {name: value for name, value in expected.items()
                     if name not in EDM_METADATA_TREES}
    actual_data = {name: value for name, value in actual.items()
                   if name not in EDM_METADATA_TREES}
    annotation_provenance = None
    strings_identical = all(expected.get(name) == value for name, value in actual.items()
                            if value["classname"] == "TObjString")
    if weight_annotations is not None:
        require(isinstance(weight_annotations, dict)
                and set(weight_annotations) == {"baseline", "replay"},
                "Replay needs both independently bound worker weight annotations")
        key = "ShiftDecayProposalWeights"
        stripped, records = [], {}
        for side, inventory in (("baseline", expected_data), ("replay", actual_data)):
            require(key in inventory and inventory[key]["classname"] == "TObjString",
                    "Replay Nano is missing its weight annotation")
            annotation = json.loads(inventory[key]["value"])
            require(isinstance(annotation, dict) and annotation == weight_annotations[side]
                    and annotation.get("schema") == "shift-decay-proposal-weight-annotation-v1",
                    "Replay weight annotation differs from its bound worker report")
            digest = annotation.get("input_nano_sha256")
            require(isinstance(digest, str) and len(digest) == 64
                    and all(value in "0123456789abcdef" for value in digest),
                    "Replay annotation needs a valid pre-annotation Nano hash")
            records[side] = dict(input_nano_sha256=digest,
                annotation_string_sha256=hashlib.sha256(inventory[key]["value"].encode()).hexdigest())
            body = {name: value for name, value in annotation.items() if name != "input_nano_sha256"}
            stripped.append(body)
            inventory[key] = dict(classname="TObjString", value=json.dumps(
                body, sort_keys=True, separators=(",", ":"), allow_nan=False))
        require(stripped[0] == stripped[1],
                "Replay changed weight policy, event authority or source normalization")
        annotation_provenance = dict(records, independently_bound_worker_annotations=True,
            only_excluded_field="ShiftDecayProposalWeights.input_nano_sha256",
            all_other_annotation_fields_preserved=True)
    require(expected_data == actual_data,
            "Replay Nano changed ordered typed branch values, schema, rows or string metadata")
    compact = json.dumps(actual_data, sort_keys=True, separators=(",", ":")).encode()
    provenance = {}
    for name in EDM_METADATA_TREES:
        before, after = expected[name], actual[name]
        provenance[name] = dict(
            baseline_entries=before["entries"], replay_entries=after["entries"],
            baseline_branches=len(before["branches"]), replay_branches=len(after["branches"]),
            baseline_inventory_sha256=hashlib.sha256(
                json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            replay_inventory_sha256=hashlib.sha256(
                json.dumps(after, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            basket_inventory_equal=before == after)
    return dict(schema="shift-nano-replay-value-equivalence-v1", complete=True,
        baseline_sha256=sha256(baseline), replay_sha256=sha256(replay),
        events=actual["Events"]["entries"], ordered_typed_values_preserved=True,
        trees={name: dict(entries=value["entries"], branches=len(value["branches"]))
               for name, value in actual_data.items() if value["classname"] == "TTree"},
        string_metadata_objects=sorted(name for name, value in actual_data.items()
                                       if value["classname"] == "TObjString"),
        string_metadata_preserved=strings_identical,
        normalization_metadata_preserved=True,
        weight_annotation_provenance=annotation_provenance,
        data_inventory_sha256=hashlib.sha256(compact).hexdigest(),
        edm_metadata_present=True, edm_metadata=provenance,
        edm_metadata_comparison="Presence and inventory only; new process/output provenance may differ.",
        edm_metadata_byte_equivalence_claimed=False,
        retirement_authorized=False)


def compare_in_runtime(inputs, merged, cmssw_base=None):
    """Use the stock CVMFS Python dependencies, without reading shared AFS."""
    if cmssw_base is None:
        return compare_all_branches(inputs, merged)
    cmssw_base = Path(cmssw_base)
    require(str(cmssw_base).startswith("/cvmfs/cms.cern.ch/")
            and (cmssw_base / "src").is_dir(),
            "Nano integrity runtime must be an existing stock CMS CVMFS release")
    with tempfile.TemporaryDirectory(prefix="nano-integrity-", dir=Path(merged).parent) as temporary:
        request, result = Path(temporary) / "request.json", Path(temporary) / "result.json"
        request.write_text(json.dumps(dict(inputs=list(map(str, inputs)), merged=str(merged))))
        environment = dict(os.environ)
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "ROOTSYS",
                     "CONDA_PREFIX", "CMSSW_BASE", "CMSSW_RELEASE_BASE", "CMSSW_SEARCH_PATH"):
            environment.pop(name, None)
        environment.update(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1")
        command = ["/bin/bash", "-c",
            'set -eo pipefail; source /cvmfs/cms.cern.ch/cmsset_default.sh; '
            'cd "$1/src"; eval "$(scram runtime -sh)"; '
            'exec python3 "$2" --request "$3" --result "$4"',
            "nano-integrity", str(cmssw_base), str(Path(__file__).resolve()), str(request), str(result)]
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=1200)
        require(completed.returncode == 0 and result.is_file(),
                "Complete Nano merge validation failed: " + (completed.stdout + completed.stderr)[-4000:])
        proof = json.loads(result.read_text())
        proof["integrity_runtime"] = str(cmssw_base)
        return proof


def validate_retirement_proof(receipt):
    """An IDs-only or unrelated merge check cannot authorize Nano retirement."""
    proof = receipt.get("merged_nano_integrity", {})
    require(proof.get("schema") == "shift-nano-all-branch-equivalence-v1"
            and proof.get("complete") is True
            and proof.get("edm_metadata_baskets_preserved") is True
            and proof.get("edm_metadata_values_preserved") is True,
            "Nano retirement requires complete all-branch and EDM metadata equivalence")
    require(isinstance(proof.get("merged_sha256"), str) and len(proof["merged_sha256"]) == 64
            and proof["merged_sha256"] == receipt["merged_nano"].get("sha256")
            and proof.get("input_sha256") == [row["sha256"] for row in receipt["nano_inputs"]],
            "Nano retirement equivalence refers to different file bytes")
    require(proof.get("events") == sum(row["events"] for row in receipt["nano_inputs"]),
            "Nano retirement equivalence has the wrong original-event count")
    require(isinstance(proof.get("event_branches"), int) and proof["event_branches"] > 0
            and len(proof.get("all_branch_inventory_sha256", "")) == 64,
            "Nano retirement equivalence lacks its complete branch inventory")
    for name in REQUIRED_TREES:
        row = proof.get("trees", {}).get(name, {})
        require(isinstance(row.get("entries"), int) and row["entries"] >= 0
                and isinstance(row.get("branches"), int) and row["branches"] > 0,
                "Nano retirement equivalence omits " + name)
    require(proof["trees"]["Events"]["entries"] == proof["events"]
            and proof["trees"]["Events"]["branches"] == proof["event_branches"],
            "Nano retirement equivalence has an inconsistent Events inventory")
    return proof


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text())
    proof = compare_all_branches(request["inputs"], request["merged"])
    with arguments.result.open("x") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")
