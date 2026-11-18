# Code style: AI-code. AI-generated atomic publication and native handoff tests.
"""Synthetic ROOT inputs qualify the real bounded publisher without collision data."""

from array import array
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils import publish_shift_blinded_data as publisher
from utils import shift_blinded_data_summary as blind

try:
    import ROOT
except ImportError:
    ROOT = None


@unittest.skipIf(ROOT is None, "ROOT runtime is unavailable")
class BlindedPublicationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "current"
        self.policy = dict(schema="shift-data-blinding-policy-v1", version="synthetic-commissioning-v1",
                           blinded=True, unit="candidate", mass_branch=blind.MASS_BRANCH,
                           mass_min_gev=2.6, mass_max_gev=3.6, histogram_bins=20, endpoints="inclusive")
        self.policy_path = self.write("policy.json", self.policy)
        self.authorization = self.write("authorization.json", dict(
            schema="shift-data-commissioning-authorization-v1", purpose="jpsi-control-commissioning",
            mass_branch=blind.MASS_BRANCH, endpoints="inclusive", approved_mass_min_gev=2.6,
            approved_mass_max_gev=3.6))
        self.response = self.write("response.json", dict(synthetic=True, final_tail_coverage_qualified=False))
        self.runtime = dict(root_version=ROOT.gROOT.GetVersion(), python_executable=sys.executable, lcg_view=None)
        self.mc_proof = self.write("mc_proof.json", dict(
            schema="shift-data-mc-as-data-validation-v2", reader_sha256=publisher.sha256(PROJECT / "utils/shift_blinded_data_summary.py"),
            policy_sha256=blind.digest_json(self.policy), all_permitted_aggregates_equal=True,
            truth_branches_present=False, collision_data_processed=False, allowed_raw_mc_pairs=1, runtime=self.runtime))
        self.review = self.write("review.json", dict(
            schema="shift-data-blinding-review-v2", passed=True, review_roles=["synthetic root review", "synthetic independent review"],
            policy_sha256=blind.digest_json(self.policy), reader_sha256=publisher.sha256(PROJECT / "utils/shift_blinded_data_summary.py"),
            converter_sha256=publisher.sha256(PROJECT / "utils/shift_blinded_data_plot_inputs.py"),
            plot_config_sha256=publisher.sha256(PROJECT / "configs/shift_data_plotter_config.py"),
            publisher_sha256=publisher.sha256(PROJECT / "utils/publish_shift_blinded_data.py"),
            authorization=publisher.pin(self.authorization), mc_as_data_validation=publisher.pin(self.mc_proof),
            mc_response_evidence=publisher.pin(self.response), plotting_helper_closure=publisher.plotting_helpers(),
            plotting_runtime=self.runtime))

    def write(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        return path

    def input(self, name, identity, values):
        path = self.directory / (name + ".root")
        output = ROOT.TFile(str(path), "RECREATE")
        tree = ROOT.TTree("Events", "Synthetic commissioning handoff")
        run, lumi, event = array("I", [369943]), array("I", [356]), array("Q", [identity])
        count, mass = array("I", [len(values)]), array("d", list(values) or [0.])
        for field, buffer, spec in (("run", run, "run/i"), ("luminosityBlock", lumi, "luminosityBlock/i"),
                                     ("event", event, "event/l"), ("nShiftDimuonVertex", count, "nShiftDimuonVertex/i"),
                                     (blind.MASS_BRANCH, mass, blind.MASS_BRANCH + "[nShiftDimuonVertex]/D")):
            tree.Branch(field, buffer, spec)
        tree.Fill();tree.Write();output.Close()
        output = ROOT.TFile.Open(str(path))
        native_output = dict(root_uuid=output.GetUUID().AsString(), sha256=publisher.sha256(path), bytes=path.stat().st_size)
        output.Close()
        producer = self.write(name + "_producer.json", dict(
            schema="shift-data-worker-result-v1", input_origin="collision-data",
            complete=True, native=dict(exit_code=0), original_events=1, output=native_output,
            candidate_branches_read=False, identity_audit=dict(ordered_original_ids_match=True, candidate_branches_read=False)))
        return dict(path=str(path), origin_uri="root://synthetic.invalid/" + path.name,
                    **native_output, producer_receipt=publisher.pin(producer), expected_event_count=1)

    def inventory(self, name, rows, identities):
        ledger = hashlib.sha256()
        for identity in sorted(identities):
            ledger.update(struct.pack(">IIQ", 369943, 356, identity))
        return self.write(name, dict(schema="shift-data-staged-publication-inputs-v1",
                                    dataset="/ZeroBias/Run2023D-v1/RAW", era="Run2023D", run=369943,
                                    expected_unique_events=len(identities), expected_identity_sha256=ledger.hexdigest(), inputs=rows))

    def publish(self, inventory, receipt_name):
        return publisher.publish(inventory, self.policy_path, self.authorization, self.mc_proof,
                                 self.response, self.review, self.output, self.directory / receipt_name)

    def scaled_inventory(self, name, rows, identities):
        path = self.inventory(name, rows, identities)
        inventory = json.loads(path.read_text())
        inventory.pop("dataset");inventory.pop("run")
        inventory.update(schema="shift-data-staged-publication-inputs-v2",
                         datasets=["/Muon0/Run2023D-v1/RAW"], runs=[369943], event_budget=10000)
        for row in rows:
            receipt_path = Path(row["producer_receipt"]["path"])
            receipt = json.loads(receipt_path.read_text())
            source = "root://eoscms.cern.ch//store/data/Run2023D/Muon0/RAW/v1/000/369/943/fixture.root"
            receipt.update(dataset=inventory["datasets"][0], era="Run2023D",
                           source_url=source, original_runs=[369943])
            receipt_path.write_text(json.dumps(receipt))
            row.update(origin_uri=source, producer_receipt=publisher.pin(receipt_path))
        inventory["inputs"] = rows
        return self.write(name, inventory)

    def test_real_publication_and_exact_retry_preserve_current_bytes_and_no_outside_values(self):
        row = self.input("first", 1, [3.05, 666.123456])
        inventory = self.inventory("inventory.json", [row], [1])
        receipt = self.publish(inventory, "receipt")
        self.assertEqual(receipt["expected_unique_events"], 1)
        self.assertEqual(receipt["allowed_pairs"], 1)
        self.assertTrue(all((self.output / name).is_file() and not (self.output / name).is_symlink()
                            for name in publisher.FILES))
        before = {name: publisher.sha256(self.output / name) for name in publisher.FILES}
        adopted = self.publish(inventory, "receipt")
        self.assertEqual(adopted, receipt)
        self.assertEqual({name: publisher.sha256(self.output / name) for name in publisher.FILES}, before)
        summary = (self.output / publisher.FILES[0]).read_text()
        self.assertNotIn("666.123456", summary)
        # An ordinary replot may regenerate its PDF while preserving the exact
        # qualified aggregate JSON and ROOT input bytes.
        stage = self.directory / "replot"
        stage.mkdir()
        config = stage / "reuse_config.py"
        config.write_text("import runpy\nglobals().update(runpy.run_path(" +
                          repr(str(PROJECT / "configs/shift_data_plotter_config.py")) +
                          ", init_globals=" + repr(dict(output_path=str(self.output), era="Run2023D",
                                                        stream="ZeroBias", run="369943")) + "))\n")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        environment["PYTHONPATH"] = os.pathsep.join(map(str, [stage, PROJECT, PROJECT / "configs",
            PROJECT / "tea/pylibs/plotting", PROJECT / "tea/pylibs/logger"])) + os.pathsep + environment.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-B", str(PROJECT / "tea/apps/examples/plotter.py"),
                                 "--config", "reuse_config"], cwd=stage, env=environment,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({name: publisher.sha256(self.output / name) for name in publisher.FILES[:2]},
                         {name: before[name] for name in publisher.FILES[:2]})
        self.assertGreater((self.output / publisher.FILES[2]).stat().st_size, 1000)

    def test_plot_failure_preserves_old_ready_delivery_and_does_not_promote_partial_subset(self):
        first = self.input("first", 1, [3.05])
        first_inventory = self.inventory("first_inventory.json", [first], [1])
        self.publish(first_inventory, "receipt_first")
        before = {name: publisher.sha256(self.output / name) for name in publisher.FILES}
        second = self.input("second", 2, [777.234567])
        second_inventory = self.inventory("second_inventory.json", [first, second], [1, 2])
        with patch.object(publisher, "render_plot", side_effect=blind.BlindingError("Synthetic plot failure")):
            with self.assertRaises(blind.BlindingError):
                self.publish(second_inventory, "receipt_second")
        self.assertEqual({name: publisher.sha256(self.output / name) for name in publisher.FILES}, before)
        self.assertFalse((self.directory / "receipt_second/publication.json").exists())
        self.assertFalse(list(self.output.glob(".incoming-*")))
        self.assertTrue(Path(second["path"]).is_file())

    def test_failed_second_file_replace_restores_the_entire_old_ready_delivery(self):
        first = self.input("first", 1, [3.05])
        first_inventory = self.inventory("first_inventory.json", [first], [1])
        self.publish(first_inventory, "receipt_first")
        before = {name: publisher.sha256(self.output / name) for name in publisher.FILES}
        second = self.input("second", 2, [3.15])
        second_inventory = self.inventory("second_inventory.json", [first, second], [1, 2])
        replace = os.replace
        failed = False
        def fail_second_once(source, target):
            nonlocal failed
            if not failed and Path(source).name == ".blinded_histograms.root.next":
                failed = True
                raise OSError("Synthetic single promotion failure")
            return replace(source, target)
        with patch.object(publisher.os, "replace", side_effect=fail_second_once):
            with self.assertRaises(OSError):
                self.publish(second_inventory, "receipt_second")
        self.assertTrue(failed)
        self.assertEqual({name: publisher.sha256(self.output / name) for name in publisher.FILES}, before)
        self.assertFalse((self.directory / "receipt_second/publication.json").exists())

    def test_failed_native_producer_and_stale_review_reject_before_any_reader_access(self):
        row = self.input("first", 1, [3.05])
        producer_path = Path(row["producer_receipt"]["path"])
        producer = json.loads(producer_path.read_text())
        producer["native"]["exit_code"] = 1
        producer_path.write_text(json.dumps(producer))
        row["producer_receipt"] = publisher.pin(producer_path)
        inventory = self.inventory("inventory.json", [row], [1])
        with patch.object(publisher.blind, "root_summary", side_effect=AssertionError("Must not read events")):
            with self.assertRaises(blind.BlindingError):
                self.publish(inventory, "receipt")
        self.assertFalse(self.output.exists())
        review = json.loads(self.review.read_text());review["publisher_sha256"] = "0" * 64
        self.review.write_text(json.dumps(review))
        with self.assertRaises(blind.BlindingError):
            self.publish(inventory, "receipt")
        self.assertFalse(self.output.exists())

    def test_prior_manual_producer_requires_original_semantic_proofs_not_normalized_flags(self):
        output = dict(sha256="a" * 64, bytes=1234, root_uuid="synthetic-prior-uuid")
        execution = self.write("prior_execution.json", dict(
            stage="cmsRun_finished", returncode=0, timed_out=False, input_kind="collision-data",
            closure_unchanged=True, private_payload_closure_unchanged=True, private_payload_closure_sha256="b" * 64))
        audit_path = self.write("prior_audit.json", dict(
            events=10, unique_events=10, root_uuid=output["root_uuid"], ordered_original_ids_match=True,
            all_events_certified=True, all_required_shift_schema_present=True, candidate_branches_read=False,
            branches_read=["run", "luminosityBlock", "event"], identity_sha256="c" * 64))
        retention = self.write("prior_retention.json", dict(
            current_root_sha256=output["sha256"], input_identity_sha256="c" * 64, candidate_branches_read=False,
            retained=[dict(name="restricted_nano.root", sha256=output["sha256"], bytes=output["bytes"])]))
        profile = self.write("prior_profile.json", dict(
            mismatches=[], gdml_matches_frozen_contract=True, physics_settings_retuned=False,
            data_truth_diagnostics_disabled=True, runtime_members_unchanged_from_qualified_capsule=1333,
            private_payload_closure_sha256="b" * 64,
            comparison={key: True for key in ("flags", "canonical_table_sha256", "lss_transport",
                                              "lss_contract_sha256", "es_component_hashes")}))
        adapter = dict(schema="shift-data-qualified-prior-producer-v1", original_events=10, output=output,
                       source_proofs={key: publisher.pin(path) for key, path in (
                           ("execution", execution), ("identity_audit", audit_path),
                           ("retention", retention), ("profile_comparison", profile))})
        publisher.validate_prior_producer(adapter)
        audit = json.loads(audit_path.read_text())
        audit["ordered_original_ids_match"] = False
        audit_path.write_text(json.dumps(audit))
        adapter["source_proofs"]["identity_audit"] = publisher.pin(audit_path)
        with self.assertRaises(blind.BlindingError):
            publisher.validate_prior_producer(adapter)

    def test_changed_plotting_helper_pin_rejects_before_any_event_reader_access(self):
        row = self.input("first", 1, [3.05])
        inventory = self.inventory("inventory.json", [row], [1])
        review = json.loads(self.review.read_text())
        review["plotting_helper_closure"][0]["sha256"] = "0" * 64
        self.review.write_text(json.dumps(review))
        with patch.object(publisher.blind, "root_summary", side_effect=AssertionError("Must not read events")):
            with self.assertRaises(blind.BlindingError):
                self.publish(inventory, "receipt")
        self.assertFalse(self.output.exists())

    def test_mc_producer_cannot_be_relabelled_as_collision_data_by_the_inventory(self):
        row = self.input("first", 1, [3.05])
        producer_path = Path(row["producer_receipt"]["path"])
        producer = json.loads(producer_path.read_text())
        producer.update(schema="ordinary-MC-producer-v1", input_origin="mc-as-data")
        producer_path.write_text(json.dumps(producer))
        row["producer_receipt"] = publisher.pin(producer_path)
        inventory = self.inventory("inventory.json", [row], [1])
        with patch.object(publisher.blind, "root_summary", side_effect=AssertionError("Must not read events")):
            with self.assertRaises(blind.BlindingError):
                self.publish(inventory, "receipt")
        self.assertFalse(self.output.exists())

    def test_receipt_commit_failure_restores_old_ready_after_all_three_promotions(self):
        first = self.input("first", 1, [3.05])
        self.publish(self.inventory("first_inventory.json", [first], [1]), "receipt_first")
        before = {name: publisher.sha256(self.output / name) for name in publisher.FILES}
        second = self.input("second", 2, [3.15])
        inventory = self.inventory("second_inventory.json", [first, second], [1, 2])
        save = publisher.save_new_json
        def fail_commit(path, value):
            if Path(path).name == "publication.json":
                raise OSError("Synthetic receipt commit failure")
            return save(path, value)
        with patch.object(publisher, "save_new_json", side_effect=fail_commit):
            with self.assertRaises(OSError):
                self.publish(inventory, "receipt_second")
        self.assertEqual({name: publisher.sha256(self.output / name) for name in publisher.FILES}, before)
        self.assertFalse((self.directory / "receipt_second/publication.json").exists())

    def test_receipt_readback_failure_removes_uncommitted_receipt_and_restores_old_ready(self):
        first = self.input("first", 1, [3.05])
        self.publish(self.inventory("first_inventory.json", [first], [1]), "receipt_first")
        before = {name: publisher.sha256(self.output / name) for name in publisher.FILES}
        second = self.input("second", 2, [3.15])
        inventory = self.inventory("second_inventory.json", [first, second], [1, 2])
        load = publisher.load
        def fail_readback(path):
            if Path(path) == self.directory / "receipt_second/publication.json":
                raise OSError("Synthetic receipt readback failure")
            return load(path)
        with patch.object(publisher, "load", side_effect=fail_readback):
            with self.assertRaises(OSError):
                self.publish(inventory, "receipt_second")
        self.assertEqual({name: publisher.sha256(self.output / name) for name in publisher.FILES}, before)
        self.assertFalse((self.directory / "receipt_second/publication.json").exists())

    def test_scaled_publication_keeps_bound_muon_labels_for_ordinary_replot_and_retry(self):
        row = self.input("muon", 1, [3.05, 666.123456])
        inventory = self.scaled_inventory("muon_inventory.json", [row], [1])
        receipt = self.publish(inventory, "muon_receipt")
        files = publisher.FILES + (publisher.CONTEXT_NAME,)
        self.assertEqual(set(receipt["outputs"]), set(files))
        before = {name: publisher.sha256(self.output / name) for name in files}
        context = publisher.read_plot_context(self.output / publisher.FILES[0])
        self.assertEqual(context["datasets"], ["/Muon0/Run2023D-v1/RAW"])
        stage = self.directory / "replot"
        stage.mkdir()
        publisher.render_plot(stage, self.output / publisher.FILES[0])
        text = subprocess.run(["pdftotext", str(stage / publisher.FILES[2]), "-"],
                              capture_output=True, text=True, check=True).stdout
        for label in ("Muon0", "Run2023D", "369943", "1 unique processed events"):
            self.assertIn(label, text)
        self.assertNotIn("ZeroBias", text)
        self.assertEqual(self.publish(inventory, "muon_receipt"), receipt)
        self.assertEqual({name: publisher.sha256(self.output / name) for name in files}, before)
        context["datasets"] = ["/Muon1/Run2023D-v1/RAW"]
        (self.output / publisher.CONTEXT_NAME).write_text(json.dumps(context))
        with self.assertRaises(blind.BlindingError):
            self.publish(inventory, "muon_receipt")

    def test_scaled_source_relabelling_rejects_before_reader_access(self):
        row = self.input("muon", 1, [3.05])
        inventory = self.scaled_inventory("muon_inventory.json", [row], [1])
        source = json.loads(inventory.read_text())
        for changes in (dict(datasets=["/Muon1/Run2023D-v1/RAW"]), dict(runs=[369944]),
                        dict(event_budget=1000001), dict(expected_unique_events=10001)):
            with self.subTest(changes=changes):
                inventory.write_text(json.dumps(dict(source, **changes)))
                with patch.object(publisher.blind, "root_summary", side_effect=AssertionError("Must not read events")):
                    with self.assertRaises(blind.BlindingError):
                        self.publish(inventory, "muon_receipt")
        self.assertFalse(self.output.exists())

    def test_large_budget_keeps_exact_exposure_and_approved_window(self):
        row = self.input("large_budget", 1, [3.05, 666.123456])
        inventory = self.scaled_inventory("large_budget_inventory.json", [row], [1])
        source = json.loads(inventory.read_text())
        source["event_budget"] = 1000000
        inventory.write_text(json.dumps(source))
        receipt = self.publish(inventory, "large_budget_receipt")
        self.assertEqual(receipt["allowed_pairs"], 1)
        self.assertEqual(receipt["events_with_allowed_pair"], 1)
        summary = json.loads((self.output / publisher.FILES[0]).read_text())
        self.assertEqual(summary["unique_processed_events"], 1)
        self.assertNotIn("666.123456", json.dumps(summary))

    def test_scaled_receipt_failure_restores_all_four_previous_files(self):
        first = self.input("first", 1, [3.05])
        self.publish(self.scaled_inventory("first_inventory.json", [first], [1]), "receipt_first")
        files = publisher.FILES + (publisher.CONTEXT_NAME,)
        before = {name: publisher.sha256(self.output / name) for name in files}
        second = self.input("second", 2, [3.15])
        inventory = self.scaled_inventory("second_inventory.json", [first, second], [1, 2])
        save = publisher.save_new_json
        def fail_commit(path, value):
            if Path(path).name == "publication.json":
                raise OSError("Synthetic four-file receipt failure")
            return save(path, value)
        with patch.object(publisher, "save_new_json", side_effect=fail_commit):
            with self.assertRaises(OSError):
                self.publish(inventory, "receipt_second")
        self.assertEqual({name: publisher.sha256(self.output / name) for name in files}, before)


if __name__ == "__main__":
    unittest.main()
