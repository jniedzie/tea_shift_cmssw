#!/usr/bin/env python3
# Code style: AI-code. AI-generated finite data commissioning publisher.
"""Publish approved aggregate data counts after exact producer and review checks.

The caller stages sealed ROOT inputs and verifies EOS/native worker handoff.
This module owns no worker, transfer, controller, reconstruction or retry loop.
Only the independently gated 2.6--3.6 GeV aggregate reaches the public delivery.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

try:
    from . import shift_blinded_data_summary as blind
    from .shift_blinded_data_plot_inputs import (CONTEXT_NAME, read_plot_context, read_summary,
                                                validate_plot_context, verify_histogram)
except ImportError:
    import shift_blinded_data_summary as blind
    from shift_blinded_data_plot_inputs import (CONTEXT_NAME, read_plot_context, read_summary,
                                               validate_plot_context, verify_histogram)


PROJECT = Path(__file__).resolve().parents[1]
FILES = ("blinded_summary.json", "blinded_histograms.root", "ShiftDimuonVertex_mass.pdf")
BAND = (2.6, 3.6)


def require(condition, message):
    if not condition:
        raise blind.BlindingError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def check_pin(reference, path):
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}
            and Path(reference["path"]).resolve() == Path(path).resolve()
            and reference["sha256"] == sha256(path), "A required evidence pin differs")


def plotting_helpers():
    paths = [PROJECT / "tea/apps/examples/plotter.py"]
    for directory in ("tea/pylibs/plotting", "tea/pylibs/logger"):
        paths.extend(sorted((PROJECT / directory).rglob("*.py")))
    return [dict(path=str(path.relative_to(PROJECT)), sha256=sha256(path)) for path in paths]


def verify_plotting_environment(review):
    require(review.get("plotting_helper_closure") == plotting_helpers(),
            "The ordinary plotting helper closure differs from its qualified bytes")
    runtime = review.get("plotting_runtime")
    import ROOT
    require(isinstance(runtime, dict) and set(runtime) == {"root_version", "python_executable", "lcg_view"}
            and runtime["root_version"] == ROOT.gROOT.GetVersion()
            and Path(runtime["python_executable"]).resolve() == Path(sys.executable).resolve(),
            "The plotting interpreter or ROOT version differs from the qualified runtime")
    if runtime["lcg_view"] is not None:
        view = Path(runtime["lcg_view"])
        require(str(view).startswith("/cvmfs/sft.cern.ch/lcg/views/LCG_")
                and (view / "setup.sh").is_file()
                and (view / "bin/python3").resolve() == Path(sys.executable).resolve(),
                "The publisher is not using its qualified immutable LCG view")


def save_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(load(path) == value, "An immutable publication record already has different content")
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def validate_authority(policy_path, authorization_path, mc_proof_path, mc_response_path, review_path):
    policy = blind.validate_policy(load(policy_path))
    require((policy["mass_min_gev"], policy["mass_max_gev"]) == BAND,
            "This publisher is limited to the approved commissioning band")
    code = {
        "reader_sha256": sha256(PROJECT / "utils/shift_blinded_data_summary.py"),
        "converter_sha256": sha256(PROJECT / "utils/shift_blinded_data_plot_inputs.py"),
        "plot_config_sha256": sha256(PROJECT / "configs/shift_data_plotter_config.py"),
        "publisher_sha256": sha256(__file__),
    }
    review, proof = load(review_path), load(mc_proof_path)
    require(review.get("schema") == "shift-data-blinding-review-v2" and review.get("passed") is True
            and review.get("policy_sha256") == blind.digest_json(policy)
            and all(review.get(key) == value for key, value in code.items())
            and isinstance(review.get("review_roles"), list) and len(review["review_roles"]) >= 2,
            "Independent review does not qualify this exact publication closure")
    verify_plotting_environment(review)
    for key, path in (("mc_as_data_validation", mc_proof_path), ("authorization", authorization_path),
                      ("mc_response_evidence", mc_response_path)):
        check_pin(review.get(key), path)
    require(proof.get("schema") == "shift-data-mc-as-data-validation-v2"
            and proof.get("reader_sha256") == code["reader_sha256"]
            and proof.get("policy_sha256") == blind.digest_json(policy)
            and proof.get("all_permitted_aggregates_equal") is True
            and proof.get("truth_branches_present") is False
            and proof.get("collision_data_processed") is False
            and proof.get("runtime") == review["plotting_runtime"]
            and type(proof.get("allowed_raw_mc_pairs")) is int and proof["allowed_raw_mc_pairs"] > 0,
            "The exact reader lacks a positive truth-stripped MC proof")
    return policy, code, review


def validate_prior_producer(producer):
    """Verify the one qualified manual pilot from its unchanged original proofs."""
    references = producer.get("source_proofs")
    require(isinstance(references, dict) and set(references) == {
        "execution", "identity_audit", "retention", "profile_comparison"},
        "The prior-producer adapter lacks its original proof closure")
    evidence = {}
    for key, reference in references.items():
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "Malformed prior producer pin")
        check_pin(reference, reference["path"])
        evidence[key] = load(reference["path"])
    execution, audit = evidence["execution"], evidence["identity_audit"]
    retention, profile = evidence["retention"], evidence["profile_comparison"]
    output = producer["output"]
    retained = [row for row in retention.get("retained", []) if row.get("name") == "restricted_nano.root"]
    require(producer.get("original_events") == 10
            and execution.get("stage") == "cmsRun_finished" and execution.get("returncode") == 0
            and execution.get("timed_out") is False and execution.get("input_kind") == "collision-data"
            and execution.get("closure_unchanged") is True and execution.get("private_payload_closure_unchanged") is True
            and audit.get("events") == 10 and audit.get("unique_events") == 10
            and audit.get("root_uuid") == output["root_uuid"] and audit.get("ordered_original_ids_match") is True
            and audit.get("all_events_certified") is True and audit.get("all_required_shift_schema_present") is True
            and audit.get("candidate_branches_read") is False
            and audit.get("branches_read") == ["run", "luminosityBlock", "event"]
            and retention.get("current_root_sha256") == output["sha256"]
            and retention.get("input_identity_sha256") == audit.get("identity_sha256")
            and retention.get("candidate_branches_read") is False
            and len(retained) == 1 and retained[0].get("sha256") == output["sha256"]
            and retained[0].get("bytes") == output["bytes"]
            and profile.get("mismatches") == [] and profile.get("gdml_matches_frozen_contract") is True
            and profile.get("physics_settings_retuned") is False and profile.get("data_truth_diagnostics_disabled") is True
            and profile.get("runtime_members_unchanged_from_qualified_capsule") == 1333
            and profile.get("private_payload_closure_sha256") == execution.get("private_payload_closure_sha256")
            and set(profile.get("comparison", {})) == {"flags", "canonical_table_sha256", "lss_transport",
                                                       "lss_contract_sha256", "es_component_hashes"}
            and all(value is True for value in profile["comparison"].values()),
            "Original native proofs do not qualify the prior pilot adapter")


def validated_inputs(inventory):
    legacy = inventory.get("schema") == "shift-data-staged-publication-inputs-v1"
    if legacy:
        require(inventory.get("dataset") == "/ZeroBias/Run2023D-v1/RAW"
                and inventory.get("era") == "Run2023D" and inventory.get("run") == 369943,
                "Unsupported commissioning inventory")
    else:
        require(inventory.get("schema") == "shift-data-staged-publication-inputs-v2",
                "Unsupported commissioning inventory")
        require(type(inventory.get("event_budget")) is int and 1 <= inventory["event_budget"] <= 1000000,
                "A scaled data cohort requires an explicit bounded event budget")
        validate_plot_context(dict(schema="shift-data-control-plot-context-v1",
                                   era=inventory.get("era"), datasets=inventory.get("datasets"),
                                   runs=inventory.get("runs"), summary_sha256="0" * 64))
    count, ledger = inventory.get("expected_unique_events"), inventory.get("expected_identity_sha256")
    require(type(count) is int and 1 <= count <= (1000 if legacy else inventory["event_budget"])
            and isinstance(ledger, str) and len(ledger) == 64
            and all(c in "0123456789abcdef" for c in ledger), "Missing exact completed-event ownership")
    rows = inventory.get("inputs")
    require(isinstance(rows, list) and rows, "No completed canonical inputs were supplied")
    inputs, mappings, expected_total = [], [], 0
    datasets, runs = set(), set()
    for row in rows:
        keys = {"path", "origin_uri", "root_uuid", "sha256", "bytes", "producer_receipt", "expected_event_count"}
        require(isinstance(row, dict) and set(row) == keys and isinstance(row["origin_uri"], str)
                and row["origin_uri"] and type(row["expected_event_count"]) is int
                and row["expected_event_count"] > 0, "Malformed canonical input registration")
        reference = row["producer_receipt"]
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
                "Missing native producer receipt pin")
        check_pin(reference, reference["path"])
        producer = load(reference["path"])
        if producer.get("schema") == "shift-data-qualified-prior-producer-v1":
            validate_prior_producer(producer)
        else:
            require(producer.get("schema") == "shift-data-worker-result-v1"
                    and producer.get("input_origin") == "collision-data",
                    "A native input is not a known collision-data producer")
        if not legacy:
            dataset, source_url = producer.get("dataset"), producer.get("source_url")
            original_runs = producer.get("original_runs")
            require(isinstance(dataset, str) and re.fullmatch(
                        r"/[A-Za-z0-9_-]+/Run2023D-v[0-9]+/RAW", dataset)
                    and producer.get("era") == inventory["era"]
                    and isinstance(original_runs, list) and original_runs
                    and all(type(value) is int and 0 < value < 2**32 for value in original_runs)
                    and original_runs == sorted(set(original_runs))
                    and isinstance(source_url, str) and row["origin_uri"] == source_url,
                    "A native producer lacks exact recorded-data source metadata")
            primary, processing = dataset.split("/")[1:3]
            version = processing.split("-v")[1]
            require(re.fullmatch(r"root://[^/]+//store/data/Run2023D/" + re.escape(primary)
                                 + r"/RAW/v" + re.escape(version) + r"/.+\.root", source_url),
                    "The native RAW source differs from its declared dataset")
            datasets.add(dataset)
            runs.update(original_runs)
        native, output, audit = producer.get("native", {}), producer.get("output", {}), producer.get("identity_audit", {})
        require(producer.get("complete") is True and type(native.get("exit_code")) is int
                and native["exit_code"] == 0 and producer.get("candidate_branches_read") is False
                and audit.get("ordered_original_ids_match") is True and audit.get("candidate_branches_read") is False
                and type(producer.get("original_events")) is int
                and producer.get("original_events") == row["expected_event_count"]
                and all(output.get(key) == row[key] for key in ("sha256", "bytes", "root_uuid")),
                "A canonical input is not a verified successful native producer output")
        inputs.append({key: row[key] for key in ("path", "root_uuid", "sha256", "bytes")})
        inputs[-1]["origin"] = "collision-data"
        mappings.append(dict(origin_uri=row["origin_uri"], staged_path=row["path"],
                             producer_receipt=reference, sha256=row["sha256"], bytes=row["bytes"],
                             root_uuid=row["root_uuid"], expected_event_count=row["expected_event_count"]))
        expected_total += row["expected_event_count"]
    require(expected_total == count, "Completed native ownership differs from the declared subset")
    if not legacy:
        require(sorted(datasets) == inventory["datasets"] and sorted(runs) == inventory["runs"],
                "The completed native sources differ from the declared data cohort")
    manifest = dict(schema=blind.MANIFEST_SCHEMA, inputs=inputs)
    blind.validate_manifest(manifest, [row["path"] for row in inputs], "collision-data")
    return manifest, mappings


def render_plot(stage, summary_path):
    configuration = stage / "_publication_config.py"
    settings = dict(summary_path=str(summary_path), output_path=str(stage))
    if read_plot_context(summary_path) is None:
        settings.update(era="Run2023D", stream="ZeroBias", run="369943")
    configuration.write_text(
        "import runpy\n" + "globals().update(runpy.run_path(" +
        repr(str(PROJECT / "configs/shift_data_plotter_config.py")) + ", init_globals=" +
        repr(settings) + "))\n")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    paths = [stage, PROJECT, PROJECT / "configs", PROJECT / "tea/pylibs/plotting", PROJECT / "tea/pylibs/logger"]
    environment["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-B", str(PROJECT / "tea/apps/examples/plotter.py"),
                             "--config", "_publication_config"], cwd=stage, env=environment,
                            capture_output=True, text=True, timeout=180)
    require(result.returncode == 0, "The ordinary data plotter failed before publication")
    configuration.unlink()


def bundle_files(directory):
    return FILES + ((CONTEXT_NAME,) if (directory / CONTEXT_NAME).exists() else ())


def check_bundle(directory, exact=True):
    files = bundle_files(directory)
    require(all((directory / name).is_file() for name in files)
            and (not exact or set(path.name for path in directory.iterdir()) == set(files)),
            "Incomplete publication bundle")
    summary_path = directory / FILES[0]
    summary, _ = read_summary(summary_path)
    read_plot_context(summary_path)
    verify_histogram(summary_path, directory / FILES[1], BAND)
    pdf = (directory / FILES[2]).read_bytes()
    require(len(pdf) > 1000 and pdf.startswith(b"%PDF-") and b"%%EOF" in pdf[-100:],
            "The rendered data PDF is incomplete")
    return summary, {name: dict(sha256=sha256(directory / name), bytes=(directory / name).stat().st_size)
                     for name in files}


def replace_delivery(output, stage, expected_hashes):
    """Replace complete staged files, restoring the old delivery on a write failure."""
    files = tuple(expected_hashes)
    existing = [name for name in bundle_files(output) if (output / name).exists()]
    require(len(existing) in (0, len(files)) and (not existing or tuple(existing) == files)
            and all(not (output / name).is_symlink() for name in files),
            "An existing public delivery needs reconciliation")
    backup = Path(tempfile.mkdtemp(prefix=".previous-", dir=output))
    previous_hashes = None
    if existing:
        for name in files:
            shutil.copy2(output / name, backup / name)
        _, previous_hashes = check_bundle(backup)
    replaced = []
    try:
        for name in files:
            temporary = output / ("." + name + ".next")
            require(not temporary.exists(), "An earlier public-file transition needs reconciliation")
            shutil.copy2(stage / name, temporary)
            os.replace(temporary, output / name)
            replaced.append(name)
        _, measured = check_bundle(output, exact=False)
        require(measured == expected_hashes, "Published delivery byte readback failed")
    except Exception:
        for name in replaced:
            if existing:
                os.replace(backup / name, output / name)
            else:
                (output / name).unlink()
        for name in files:
            (output / ("." + name + ".next")).unlink(missing_ok=True)
        if existing:
            _, restored = check_bundle(output, exact=False)
            require(restored == previous_hashes, "Previous ready delivery could not be restored")
        shutil.rmtree(backup)
        raise
    return backup


def completed_publication(receipt_dir, output, inventory_path, review_path, code, manifest):
    """Adopt an exact completed producer attempt without rerunning its data reader."""
    receipt_path = receipt_dir / "publication.json"
    if not receipt_path.exists():
        return None
    receipt = load(receipt_path)
    require(receipt.get("complete") is True and receipt.get("code") == code
            and receipt.get("inventory") == pin(inventory_path) and receipt.get("review") == pin(review_path),
            "A previous publication receipt needs reconciliation")
    for source in manifest["inputs"]:
        blind.verify_input_bytes(source)
    _, hashes = check_bundle(output, exact=False)
    # An ordinary replot may regenerate the derivative PDF; frozen aggregate
    # JSON and ROOT input must still be byte-identical to their producer receipt.
    retained = FILES[:2] + ((CONTEXT_NAME,) if CONTEXT_NAME in hashes else ())
    require(set(hashes) == set(receipt.get("outputs", {}))
            and all(hashes[name] == receipt["outputs"].get(name) for name in retained),
            "Previously committed aggregate inputs differ from their receipt")
    return receipt


def restore_previous_delivery(output, backup):
    files = bundle_files(backup) if (backup / FILES[0]).exists() else bundle_files(output)
    existing = [name for name in files if (backup / name).is_file()]
    require(len(existing) in (0, len(files)), "Previous-delivery backup needs reconciliation")
    expected = check_bundle(backup)[1] if existing else None
    for name in files:
        if existing:
            os.replace(backup / name, output / name)
        else:
            (output / name).unlink(missing_ok=True)
    if existing:
        require(check_bundle(output, exact=False)[1] == expected, "Previous ready delivery could not be restored")
    shutil.rmtree(backup)


def publish(inventory_path, policy_path, authorization_path, mc_proof_path, mc_response_path,
            review_path, output, receipt_dir):
    output, receipt_dir = Path(output).resolve(), Path(receipt_dir).resolve()
    policy, code, review = validate_authority(policy_path, authorization_path, mc_proof_path, mc_response_path, review_path)
    inventory = load(inventory_path)
    manifest, mappings = validated_inputs(inventory)
    qualification = dict(
        schema="shift-data-blinding-qualification-v2", purpose="jpsi-control-commissioning",
        approved_mass_min_gev=BAND[0], approved_mass_max_gev=BAND[1],
        policy_sha256=blind.digest_json(policy), executable_sha256=code["reader_sha256"],
        input_provenance_sha256=blind.digest_json(manifest), mc_as_data_passed=True,
        independent_review_passed=True, authorization=pin(authorization_path),
        mc_response_evidence=pin(mc_response_path),
    )
    blind.validate_qualification(policy, "collision-data", qualification, blind.digest_json(manifest))
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = completed_publication(receipt_dir, output, inventory_path, review_path, code, manifest)
        if existing is not None:
            return existing
        save_new_json(receipt_dir / "provenance.json", manifest)
        save_new_json(receipt_dir / "qualification.json", qualification)
        summary = blind.root_summary([row["path"] for row in manifest["inputs"]], policy,
                                     "collision-data", qualification, manifest)
        require(summary["unique_processed_events"] == inventory["expected_unique_events"]
                and summary["processed_identity_sha256"] == inventory["expected_identity_sha256"]
                and summary["duplicate_events_skipped"] == 0,
                "The actual completed input union differs from frozen unique event ownership")
        save_new_json(receipt_dir / "blinded_summary.json", summary)
        stage = Path(tempfile.mkdtemp(prefix=".incoming-", dir=output))
        try:
            shutil.copy2(receipt_dir / "blinded_summary.json", stage / FILES[0])
            if inventory["schema"] == "shift-data-staged-publication-inputs-v2":
                save_new_json(stage / CONTEXT_NAME, dict(
                    schema="shift-data-control-plot-context-v1", era=inventory["era"],
                    datasets=inventory["datasets"], runs=inventory["runs"],
                    summary_sha256=sha256(stage / FILES[0])))
            verify_plotting_environment(review)
            render_plot(stage, stage / FILES[0])
            verify_plotting_environment(review)
            require(code == {key: sha256(PROJECT / path) for key, path in (
                ("reader_sha256", "utils/shift_blinded_data_summary.py"),
                ("converter_sha256", "utils/shift_blinded_data_plot_inputs.py"),
                ("plot_config_sha256", "configs/shift_data_plotter_config.py"),
                ("publisher_sha256", "utils/publish_shift_blinded_data.py"))},
                "The publication code changed during processing")
            _, hashes = check_bundle(stage)
            save_new_json(receipt_dir / "prepared_publication.json", dict(
                schema="shift-data-prepared-publication-v1", outputs=hashes, code=code,
                inventory=pin(inventory_path), review=pin(review_path)))
            backup = replace_delivery(output, stage, hashes)
            receipt = dict(schema="shift-data-control-publication-v2", complete=True,
                           inventory=pin(inventory_path), review=pin(review_path), code=code,
                           input_mapping=mappings, output_directory=str(output), outputs=hashes,
                           expected_unique_events=inventory["expected_unique_events"],
                           processed_identity_sha256=summary["processed_identity_sha256"],
                           allowed_pairs=summary["allowed_pairs"], events_with_allowed_pair=summary["events_with_allowed_pair"],
                           normalization="none", output_readback_verified=True)
            receipt_path = receipt_dir / "publication.json"
            try:
                save_new_json(receipt_path, receipt)
                require(load(receipt_path) == receipt, "Publication receipt readback failed")
            except Exception:
                if receipt_path.exists():
                    expected = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
                    require(sha256(receipt_path) == hashlib.sha256(expected).hexdigest(),
                            "An uncertain receipt requires reconciliation before restoring the previous delivery")
                    receipt_path.unlink()
                restore_previous_delivery(output, backup)
                raise
            shutil.rmtree(backup)
            return receipt
        finally:
            if stage is not None and stage.exists():
                shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "policy", "authorization", "mc-proof", "mc-response-evidence", "review", "output", "receipt-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        publish(args.inventory, args.policy, args.authorization, args.mc_proof,
                args.mc_response_evidence, args.review, args.output, args.receipt_dir)
    except blind.BlindingError as error:
        print("Data control publication failed: " + str(error), file=sys.stderr)
        return 1
    except Exception:
        print("Data control publication failed; preserve native inputs and publication receipts", file=sys.stderr)
        return 1
    print("Approved data control summary and PDF published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
