# Code style: AI-code. AI-generated normalization and closure regression tests.
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "streaming", Path(__file__).resolve().parents[1] / "utils/shift_streaming_histograms.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
streaming = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(streaming)


def cohort():
    return dict(schema="shift-streaming-nano-cohort-v1", complete=True,
                cohort_id="qcd_0to1_source0", stratum="qcd_0to1",
                expected_reco_job_ids=[10, 11],
                verified_nano=[dict(job_id=10, events=1), dict(job_id=11, events=1)],
                selected_events=[dict(event_id=[1, 1, 1], native_weight=2., sampling_probability=.1),
                                 dict(event_id=[1, 1, 2], native_weight=1., sampling_probability=1.)],
                normalization=dict(schema="shift-stream-source-exposure-v1", stratum="qcd_0to1",
                                   source_id="GEN0", gen_sha256="a" * 64, sigma_nd_pb=7e10,
                                   full_source_trials=12345, no_second_filter_efficiency=True,
                                   histogram_weight=streaming.WEIGHT_POLICY))


def receipt(index=0, stratum="qcd_0to1"):
    record = cohort()
    return dict(schema="shift-streaming-histogram-cohort-v1", complete=True,
                cohort_id="cohort" + str(index), stratum=stratum,
                normalization=dict(record["normalization"], source_id="GEN" + str(index),
                                   gen_sha256=str(index) * 64, stratum=stratum))


class StreamingHistogramsTest(unittest.TestCase):
    def test_campaign_publication_is_available_before_any_plot_and_checks_immutable_outputs(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            source = scratch/'merged.root'
            source.write_bytes(b'verified-merged-histogram')
            rows = [dict(stratum='qcd_0to1', path=str(source))]
            with patch.dict(sys.modules, {'teaHelpers': types.SimpleNamespace(eos_xrootd_url=lambda path: None)}):
                result = streaming.publish_campaign_histograms(rows, [], str(scratch/'data'), '20261019_test',
                    '20261019T120000Z_abcdef123456', scratch)
                self.assertEqual(Path(result[0]['path']).read_bytes(), source.read_bytes())
                # A resumed publish accepts only a byte-identical existing version.
                streaming.publish_campaign_histograms(rows, [], str(scratch/'data'), '20261019_test',
                    '20261019T120000Z_abcdef123456', scratch)
                source.write_bytes(b'different-merged-histogram')
                with self.assertRaises(FileExistsError):
                    streaming.publish_campaign_histograms(rows, [], str(scratch/'data'), '20261019_test',
                        '20261019T120000Z_abcdef123456', scratch)

    def test_runtime_missing_any_log_mass_category_is_rejected(self):
        import ROOT
        names=[name for name in streaming.REQUIRED_HISTOGRAMS if name.endswith('_logMass')]
        self.assertEqual(len(names),7)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old_runtime.root';source=ROOT.TFile(str(path),'RECREATE')
            source.mkdir('dimuon').cd()
            for name in streaming.REQUIRED_HISTOGRAMS:
                if name==names[-1]:continue
                histogram=ROOT.TH1D(Path(name).name,'',160,-1.,3.);histogram.Write()
            source.Close()
            with self.assertRaisesRegex(ValueError,'Missing required histogram.*logMass'):
                streaming.histogram_summary(path)

    def test_resume_summary_does_not_rewrite_published_histograms(self):
        import ROOT
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'histograms.root'
            output = ROOT.TFile(str(path), 'RECREATE')
            directory = output.mkdir('dimuon')
            directory.cd()
            for name in streaming.REQUIRED_HISTOGRAMS:
                histogram = ROOT.TH1D(Path(name).name, '', 20, 0., 200.)
                histogram.Fill(25.)
                histogram.Write()
            output.Close()
            before = streaming.sha256(path)
            result = streaming.histogram_summary(path, materialize_errors=False)
            self.assertEqual(streaming.sha256(path), before)
            self.assertEqual(result[streaming.REQUIRED_HISTOGRAMS[0]]['sumw2'], 1.)

    def test_valid_cohort_retains_double_weight_and_full_trial_exposure(self):
        record = cohort()
        self.assertEqual(len(streaming.validate_cohort(record)), 2)
        self.assertEqual(record["normalization"]["full_source_trials"], 12345)

    def test_batched_nano_closes_original_events_instead_of_counting_jobs(self):
        record = cohort()
        record['verified_nano'][0]['events'] = 2
        record['selected_events'].append(dict(event_id=[1, 1, 3], native_weight=3., sampling_probability=.25))
        self.assertEqual(len(streaming.validate_cohort(record)), 3)
        record['verified_nano'][0]['events'] = 1
        with self.assertRaisesRegex(ValueError, 'selected event cohort'):
            streaming.validate_cohort(record)

    def test_missing_slow_job_cannot_become_a_normalized_cohort(self):
        record = cohort()
        record["verified_nano"].pop()
        with self.assertRaisesRegex(ValueError, "predetermined"):
            streaming.validate_cohort(record)

    def test_duplicate_original_and_invalid_probability_fail(self):
        for edit in (lambda rows: rows.append(copy.deepcopy(rows[0])),
                     lambda rows: rows[0].update(sampling_probability=0),
                     lambda rows: rows[0].update(native_weight=float("nan"))):
            record = cohort()
            edit(record["selected_events"])
            with self.assertRaises(ValueError):
                streaming.validate_cohort(record)

    def test_second_filter_efficiency_and_selected_sum_denominator_rejected(self):
        for edit in (dict(no_second_filter_efficiency=False), dict(full_source_trials=0),
                     dict(histogram_weight="native_weight * filter_efficiency")):
            record = cohort()
            record["normalization"].update(edit)
            with self.assertRaisesRegex(ValueError, "exposure authority"):
                streaming.validate_cohort(record)

    def test_snapshot_pools_distinct_source_trials_within_each_stratum(self):
        records = [receipt(0), receipt(1), receipt(2, "qcd_20toinf")]
        groups = streaming.snapshot_authorities(records, [row["cohort_id"] for row in records])
        self.assertEqual(sum(row["normalization"]["full_source_trials"]
                             for row in groups["qcd_0to1"]), 24690)
        self.assertEqual(len(groups["qcd_20toinf"]), 1)

    def test_duplicate_full_gen_source_cannot_double_the_exposure(self):
        records = [receipt(0), receipt(1)]
        for field in ("source_id", "gen_sha256"):
            bad = copy.deepcopy(records)
            bad[1]["normalization"][field] = bad[0]["normalization"][field]
            with self.assertRaisesRegex(ValueError, "repeats"):
                streaming.snapshot_authorities(bad, ["cohort0", "cohort1"])

    def test_fastest_finished_files_do_not_replace_predetermined_snapshot(self):
        with self.assertRaisesRegex(ValueError, "predetermined"):
            streaming.snapshot_authorities([receipt(0)], ["cohort0", "cohort1"])

    def test_cross_section_and_stratum_mismatch_fail(self):
        records = [receipt(0), receipt(1)]
        records[1]["normalization"]["sigma_nd_pb"] *= 2
        with self.assertRaisesRegex(ValueError, "cross section"):
            streaming.snapshot_authorities(records, ["cohort0", "cohort1"])
        records = [receipt(0)]
        records[0]["normalization"]["stratum"] = "qcd_20toinf"
        with self.assertRaisesRegex(ValueError, "strata"):
            streaming.snapshot_authorities(records, ["cohort0"])

    def cleanup_record(self):
        base = "root://eoshome-j.cern.ch//eos/user/j/jniedzie/shift_cmssw/qcd_pThat0to1GeV/campaign/"
        return dict(cohort_id="one", nano_inputs=[dict(path=base + "nanoAOD/nano_job1.root", sha256="a" * 64, events=1)],
                    merged_nano=dict(path=base + "nanoAOD_merged/one.root", sha256="b" * 64),
                    histogram=dict(path=base + "histograms/one.root"),
                    raw_histogram=dict(path=base + "histograms/one_raw.root"),
                    merged_nano_integrity=dict(schema="shift-nano-all-branch-equivalence-v1",
                        complete=True, edm_metadata_baskets_preserved=True,
                        edm_metadata_values_preserved=True, input_sha256=["a" * 64],
                        merged_sha256="b" * 64, events=1, event_branches=1,
                        all_branch_inventory_sha256="c" * 64,
                        trees={name: dict(entries=1, branches=1) for name in
                               ("Events", "Runs", "LuminosityBlocks", "MetaData", "ParameterSets")}))

    def test_cleanup_requires_complete_branch_run_lumi_and_edm_proof_before_any_io(self):
        for missing in ("merged_nano_integrity", "Runs", "LuminosityBlocks", "ParameterSets"):
            record = self.cleanup_record()
            if missing == "merged_nano_integrity":
                del record[missing]
            else:
                del record["merged_nano_integrity"]["trees"][missing]
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary, \
                    patch.dict(sys.modules, teaHelpers=types.SimpleNamespace(eos_xrootd_url=lambda p: None)), \
                    patch.object(streaming, "stage") as stage, \
                    patch.object(streaming.subprocess, "run") as command:
                result = streaming.retire_consumed_nano(record, Path(temporary) / "cleanup.json")
                self.assertFalse(result["complete"])
                stage.assert_not_called()
                command.assert_not_called()

    def test_cleanup_only_removes_exact_consumed_nano_after_explicit_absence(self):
        record = self.cleanup_record()
        native = []

        def stage(source, destination, checksum):
            destination.write_bytes(b"payload")

        def command(argv, **options):
            native.append(argv)
            return subprocess.CompletedProcess(argv, 54 if "stat" in argv else 0,
                                               stdout="", stderr="[3011] not found" if "stat" in argv else "")

        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(sys.modules, teaHelpers=types.SimpleNamespace(eos_xrootd_url=lambda p: None)), \
                patch.object(streaming, "stage", stage), patch.object(streaming.subprocess, "run", command):
            path = Path(temporary) / "cleanup.json"
            result = streaming.retire_consumed_nano(record, path)
            self.assertTrue(result["complete"])
            self.assertEqual(result["removed_bytes"], 7)
            self.assertEqual(len(result["deleted"]), 1)
            self.assertEqual(json.loads(path.read_text()), result)
        self.assertEqual([argv[2] for argv in native], ["rm", "stat"])
        self.assertTrue(all("/nanoAOD/nano_job1.root" in argv[-1] for argv in native))

    def test_cleanup_refuses_gen_alias_duplicate_and_unknown_post_delete_state(self):
        for change in ("gen", "duplicate", "alias", "unknown"):
            record = self.cleanup_record()
            if change == "gen":
                record["nano_inputs"][0]["path"] = record["nano_inputs"][0]["path"].replace("/nanoAOD/", "/GEN/")
            elif change == "duplicate":
                record["nano_inputs"] *= 2
            elif change == "alias":
                record["merged_nano"]["path"] = record["nano_inputs"][0]["path"]
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary, \
                    patch.dict(sys.modules, teaHelpers=types.SimpleNamespace(eos_xrootd_url=lambda p: None)), \
                    patch.object(streaming, "stage", side_effect=lambda p, d, h: d.write_bytes(b"payload")), \
                    patch.object(streaming.subprocess, "run", return_value=subprocess.CompletedProcess([], 8, "", "network unavailable")) as command:
                result = streaming.retire_consumed_nano(record, Path(temporary) / "cleanup.json")
                self.assertFalse(result["complete"])
                self.assertIn("error", result)
                if change != "unknown":
                    command.assert_not_called()

    def test_publication_checks_reserved_receipt_space_before_upload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.root"
            path.write_bytes(b"1234")
            with self.assertRaisesRegex(ValueError, "reservation"):
                streaming.publication_budget([path], [], 65536)
            result = streaming.publication_budget([path], [dict(bytes=10)], 100000)
            self.assertEqual(result["planned_bytes"], 65550)

    def test_batched_publication_requires_an_explicit_larger_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "batched.root"
            with path.open("wb") as stream:
                stream.truncate(33 * 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "reservation"):
                streaming.publication_budget([path], [])
            result = streaming.publication_budget([path], [], 128 * 1024 * 1024)
            self.assertEqual(result["maximum_bytes"], 128 * 1024 * 1024)
            result = streaming.publication_budget([path], [], 512 * 1024 * 1024)
            self.assertEqual(result["maximum_bytes"], 512 * 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "reservation"):
                streaming.publication_budget([path], [], 513 * 1024 * 1024)

    def test_snapshot_publishes_durable_plot_inputs_and_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            hist = base / "input.root"
            hist.write_bytes(b"histograms")
            summary = {name: dict(sumw=0., sumw2=0.) for name in streaming.REQUIRED_HISTOGRAMS}
            record = receipt()
            for kind in ("histogram", "raw_histogram"):
                record[kind] = dict(path=str(hist), sha256=streaming.sha256(hist), histograms=summary)
            source = base / "cohort.json"
            source.write_text(json.dumps(record))
            inventory = base / "inventory.json"
            inventory.write_text(json.dumps(dict(schema="shift-streaming-histogram-snapshot-inputs-v1",
                snapshot_id="one", expected_cohort_ids=["cohort0"],
                cohort_receipts=[dict(path=str(source), sha256=streaming.sha256(source))])))
            reference = base / "references.json"
            reference.write_text(json.dumps(dict(schema="shift-sm-reference-histograms-v1",
                samples=[dict(stratum="dy_20to-1", path="/eos/reference.root")])) )
            args = types.SimpleNamespace(inventory=inventory, output_directory=base / "output",
                merge_script=base / "merge.py", plotter=None, plot_output=None,
                publish_directory="/eos/snapshot", reference_inputs=reference)
            uploaded = {}

            def upload(path, destination):
                uploaded[path.name] = path.read_bytes()
                return destination + "/" + path.name

            with patch.object(streaming, "stage", side_effect=lambda p, d, h: d.write_bytes(Path(p).read_bytes())), \
                    patch.object(streaming, "merge", side_effect=lambda i, d, m, s: d.write_bytes(b"merged")), \
                    patch.object(streaming, "histogram_summary", return_value=summary), \
                    patch.object(streaming, "embed_metadata"), patch.object(streaming, "publish", upload):
                streaming.snapshot(args)
            complete = json.loads(uploaded["snapshot_complete.json"])
            inputs = json.loads(uploaded["plot_inputs.json"])
            self.assertEqual(complete["plot_inputs"]["path"], "/eos/snapshot/plot_inputs.json")
            self.assertEqual(inputs["samples"][0]["path"], "/eos/snapshot/qcd_0to1_histogram.root")
            self.assertEqual(inputs["reference_samples"][0]["stratum"], "dy_20to-1")

    def test_targeted_diagnostic_cannot_enter_a_normalized_snapshot(self):
        diagnostic = dict(schema="shift-streaming-histogram-diagnostic-v1", complete=True,
                          diagnostic_only=True, normalization_ready=False)
        with self.assertRaisesRegex(ValueError, "complete histogram cohorts"):
            streaming.snapshot_authorities([diagnostic], ["hand_picked_event"])

    def test_diagnostic_requires_explicitly_disabled_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = Path(temporary) / "input.json"
            descriptor.write_text(json.dumps(dict(schema="shift-streaming-nano-diagnostic-v1",
                                                  diagnostic_only=True, normalization_ready=True)))
            args = types.SimpleNamespace(descriptor=descriptor)
            with self.assertRaisesRegex(ValueError, "unnormalized"):
                streaming.diagnostic(args)

    def test_diagnostic_rejects_reco_receipt_for_another_gen_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            receipt = base / "reco.json"
            receipt.write_text(json.dumps(dict(complete=True, source_descriptor_sha256="unrelated")))
            descriptor = base / "input.json"
            descriptor.write_text(json.dumps(dict(schema="shift-streaming-nano-diagnostic-v1",
                diagnostic_only=True, normalization_ready=False, nano_inputs=[dict(
                    stratum="qcd_20toinf", reco_receipt=str(receipt), source_descriptor_sha256="wanted",
                    gen_sha256="a" * 64, event_ids=[[1, 1, 1]], native_weights=[1.])])) )
            args = types.SimpleNamespace(descriptor=descriptor, plotter=None, output_directory=base / "out")
            def stage(source, destination, checksum):
                destination.write_bytes(Path(source).read_bytes())
            with patch.object(streaming, "stage", stage), patch.object(streaming, "root_file") as root:
                with self.assertRaisesRegex(ValueError, "pinned original"):
                    streaming.diagnostic(args)
                root.assert_not_called()


if __name__ == "__main__":
    unittest.main()
