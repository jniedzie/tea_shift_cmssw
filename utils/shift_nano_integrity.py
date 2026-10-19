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
