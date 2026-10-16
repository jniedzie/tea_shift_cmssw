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
    def test_valid_cohort_retains_double_weight_and_full_trial_exposure(self):
        record = cohort()
        self.assertEqual(len(streaming.validate_cohort(record)), 2)
        self.assertEqual(record["normalization"]["full_source_trials"], 12345)

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
        return dict(cohort_id="one", nano_inputs=[dict(path=base + "nanoAOD/nano_job1.root", sha256="a" * 64)],
                    merged_nano=dict(path=base + "nanoAOD_merged/one.root"),
                    histogram=dict(path=base + "histograms/one.root"),
                    raw_histogram=dict(path=base + "histograms/one_raw.root"))

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
