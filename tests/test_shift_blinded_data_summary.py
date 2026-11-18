# Code style: AI-code. AI-generated adversarial blinding regression tests.
"""Synthetic records only; no real collision data are opened by these tests."""

from array import array
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils/shift_blinded_data_summary.py"
spec = importlib.util.spec_from_file_location("shift_blinded_data_summary", MODULE_PATH)
blind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blind)


def policy():
    return dict(schema="shift-data-blinding-policy-v1", version="synthetic-v1", blinded=True,
                unit="candidate", mass_branch="ShiftDimuonVertex_mass", mass_min_gev=2.9,
                mass_max_gev=3.3, histogram_bins=4, endpoints="inclusive")


class BlindedSummaryTest(unittest.TestCase):
    def test_every_candidate_is_gated_and_only_allowlisted_aggregates_escape(self):
        summary = blind.BlindedSummary(policy(), "mc-as-data")
        summary.add_event((1, 2, 3), 4, [3.05, 666.123456, 3.25, 0.456789])
        summary.add_event((1, 2, 4), 2, [777.234567, 888.345678])
        result = summary.result([])
        self.assertEqual(set(result), blind.SUMMARY_KEYS)
        self.assertEqual(result["unique_processed_events"], 2)
        self.assertEqual(result["events_with_allowed_pair"], 1)
        self.assertEqual(result["allowed_pairs"], 2)
        self.assertEqual(result["mass_bin_counts"], [0, 1, 0, 1])
        encoded = json.dumps(result)
        for canary in ("666.123456", "777.234567", "888.345678", "0.456789"):
            self.assertNotIn(canary, encoded)
        for name in ("outside_pairs", "total_pairs", "muonIdx1", "bdtScore", "refittedMass"):
            self.assertNotIn(name, encoded)

    def test_inclusive_boundaries_and_adjacent_values(self):
        summary = blind.BlindedSummary(policy(), "mc-as-data")
        values = [math.nextafter(2.9, -math.inf), 2.9, math.nextafter(2.9, math.inf),
                  math.nextafter(3.3, -math.inf), 3.3, math.nextafter(3.3, math.inf)]
        summary.add_event((1, 1, 1), len(values), values)
        self.assertEqual(summary.result([])["mass_bin_counts"], [2, 0, 0, 2])

    def test_duplicate_identity_keeps_exact_first_owner(self):
        summary = blind.BlindedSummary(policy(), "mc-as-data")
        summary.add_event((1, 2, 2 ** 64 - 1), 1, [3.05])
        summary.add_event((1, 2, 2 ** 64 - 1), 2, [3.15, 3.25])
        summary.add_event((2, 2, 2 ** 64 - 1), 0, [])
        result = summary.result([])
        self.assertEqual(result["unique_processed_events"], 2)
        self.assertEqual(result["allowed_pairs"], 1)
        self.assertEqual(result["duplicate_events_skipped"], 1)

    def test_identity_digest_is_not_mass_conditioned_or_order_dependent(self):
        first, second = (blind.BlindedSummary(policy(), "mc-as-data") for _ in range(2))
        first.add_event((1, 1, 2), 1, [3.1])
        first.add_event((1, 1, 1), 0, [])
        second.add_event((1, 1, 1), 1, [90.])
        second.add_event((1, 1, 2), 1, [91.])
        self.assertEqual(first.result([])["processed_identity_sha256"],
                         second.result([])["processed_identity_sha256"])

    def test_malformed_identities_and_counts_fail_without_candidate_values(self):
        for identity in ((-1, 1, 1), (2 ** 32, 1, 1), (1, 1, 2 ** 64),
                         (1, 1, 1.0), (True, 1, 1), (1, 1)):
            with self.subTest(identity=identity), self.assertRaises(blind.BlindingError):
                blind.BlindedSummary(policy(), "mc-as-data").add_event(identity, 0, [])
        for count, masses in ((-1, []), (True, [3.1]), (1, []), (0, [3.1]),
                              (1, [math.nan]), (1, [math.inf]), (1, [-123456.789])):
            with self.subTest(count=count, masses=masses), self.assertRaises(blind.BlindingError) as failure:
                blind.BlindedSummary(policy(), "mc-as-data").add_event((1, 1, 1), count, masses)
            self.assertNotIn("123456", str(failure.exception))

    def test_missing_or_unsupported_policy_fails_closed(self):
        for key in policy():
            broken = policy()
            del broken[key]
            with self.subTest(missing=key), self.assertRaises(blind.BlindingError):
                blind.BlindedSummary(broken, "mc-as-data")
        for key, value in (("blinded", False), ("unit", "event"), ("version", ""),
                           ("mass_branch", "ShiftDimuonVertex_refittedMass"),
                           ("mass_min_gev", math.nan), ("mass_max_gev", 2.8),
                           ("histogram_bins", True), ("endpoints", "exclusive")):
            broken = policy()
            broken[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(blind.BlindingError):
                blind.BlindedSummary(broken, "mc-as-data")
        with self.assertRaises(blind.BlindingError):
            blind.BlindedSummary(dict(policy(), unblind=True), "mc-as-data")

    def test_collision_data_require_exact_policy_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence.json"
            authorization = dict(schema="shift-data-commissioning-authorization-v1",
                                 purpose="jpsi-control-commissioning", mass_branch=blind.MASS_BRANCH,
                                 approved_mass_min_gev=2.9, approved_mass_max_gev=3.3, endpoints="inclusive")
            evidence.write_text(json.dumps(authorization))
            reference = dict(path=str(evidence), sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
            qualified = dict(schema="shift-data-blinding-qualification-v2",
                             purpose="jpsi-control-commissioning", approved_mass_min_gev=2.9,
                             approved_mass_max_gev=3.3, authorization=reference, mc_response_evidence=reference,
                             policy_sha256=blind.digest_json(policy()), mc_as_data_passed=True,
                             independent_review_passed=True, executable_sha256=blind.executable_sha256(),
                             input_provenance_sha256="a" * 64)
            blind.BlindedSummary(policy(), "collision-data", qualified, "a" * 64)
            for receipt in (None, {}, dict(qualified, independent_review_passed=False),
                            dict(qualified, policy_sha256="0" * 64),
                            dict(qualified, executable_sha256="0" * 64),
                            dict(qualified, approved_mass_max_gev=4.),
                            dict(qualified, purpose="final-jpsi-yield"),
                            dict(qualified, authorization=dict(reference, sha256="0" * 64))):
                with self.subTest(receipt=receipt), self.assertRaises(blind.BlindingError):
                    blind.BlindedSummary(policy(), "collision-data", receipt, "a" * 64)
            with self.assertRaises(blind.BlindingError):
                blind.BlindedSummary(policy(), "collision-data", qualified, "b" * 64)
            widened = dict(policy(), mass_max_gev=4.)
            widened_qualification = dict(qualified, policy_sha256=blind.digest_json(widened),
                                         approved_mass_max_gev=4.)
            with self.assertRaises(blind.BlindingError):
                blind.BlindedSummary(widened, "collision-data", widened_qualification, "a" * 64)
            evidence.write_text('{"synthetic_evidence": false}')
            with self.assertRaises(blind.BlindingError):
                blind.BlindedSummary(policy(), "collision-data", qualified, "a" * 64)
        with self.assertRaises(blind.BlindingError):
            blind.BlindedSummary(policy(), "unknown")

    def test_atomic_output_does_not_replace_evidence_or_leave_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            result = blind.BlindedSummary(policy(), "mc-as-data").result([])
            blind.write_summary(output, result)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                blind.write_summary(output, result)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [output])
            bad = copy.deepcopy(result)
            bad["candidate_rows"] = [[999.12345]]
            with self.assertRaises(blind.BlindingError):
                blind.write_summary(Path(directory) / "bad.json", bad)
            self.assertFalse((Path(directory) / "bad.json").exists())
            for key, value in (("policy", dict(policy(), candidate_rows=[999.12345])),
                               ("mass_bin_edges_gev", [999.12345]),
                               ("input_provenance", [dict(candidate_rows=[999.12345])])):
                bad = dict(result)
                bad[key] = value
                with self.subTest(key=key), self.assertRaises(blind.BlindingError):
                    blind.write_summary(Path(directory) / "bad.json", bad)
                self.assertFalse((Path(directory) / "bad.json").exists())

    def test_input_provenance_cannot_carry_candidate_payload(self):
        summary = blind.BlindedSummary(policy(), "mc-as-data")
        with self.assertRaises(blind.BlindingError):
            summary.result([dict(path="input.root", root_uuid="synthetic", entries=1,
                                 candidate_rows=[999.123456])])

    def test_validation_state_is_copied_and_internal_mutation_fails_closed(self):
        original = policy()
        summary = blind.BlindedSummary(original, "mc-as-data")
        original["mass_max_gev"] = 1000.
        summary.add_event((1, 1, 1), 1, [999.123456])
        self.assertEqual(summary.result([])["allowed_pairs"], 0)
        summary.policy["mass_max_gev"] = 1000.
        with self.assertRaises(blind.BlindingError):
            summary.result([])

    def test_manifest_origin_paths_and_schema_fail_closed(self):
        record = dict(path="a.root", root_uuid="synthetic-uuid", origin="collision-data", sha256="a" * 64, bytes=1)
        good = dict(schema=blind.MANIFEST_SCHEMA, inputs=[record])
        self.assertEqual(blind.validate_manifest(good, ["a.root"], "collision-data"),
                         {"a.root": record})
        for manifest, paths, kind in ((None, ["a.root"], "mc-as-data"),
                                      (good, ["a.root"], "mc-as-data"),
                                      (good, ["b.root"], "collision-data"),
                                      (dict(good, candidate_rows=[999.12345]), ["a.root"], "collision-data")):
            with self.subTest(kind=kind, paths=paths), self.assertRaises(blind.BlindingError):
                blind.validate_manifest(manifest, paths, kind)


try:
    import ROOT
except ImportError:
    ROOT = None


@unittest.skipIf(ROOT is None, "ROOT runtime is unavailable")
class BlindedRootReaderTest(unittest.TestCase):
    def run_cli(self, directory, paths, manifest):
        policy_path, manifest_path = directory / "policy.json", directory / "manifest.json"
        policy_path.write_text(json.dumps(policy()))
        manifest_path.write_text(json.dumps(manifest))
        return subprocess.run([
            sys.executable, str(MODULE_PATH), "--input", *map(str, paths),
            "--input-kind", "mc-as-data", "--policy", str(policy_path),
            "--provenance-manifest", str(manifest_path), "--output", str(directory / "summary.json"),
        ], capture_output=True, text=True, timeout=60)

    def manifest(self, paths):
        records = []
        for path in paths:
            source = ROOT.TFile.Open(str(path))
            records.append(dict(path=str(path), root_uuid=source.GetUUID().AsString(), origin="mc-as-data",
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size))
            source.Close()
        return dict(schema=blind.MANIFEST_SCHEMA, inputs=records)

    def fixture(self, path, rows, missing_mass=False, wrong_count=False):
        source = ROOT.TFile(str(path), "RECREATE")
        tree = ROOT.TTree("Events", "Synthetic blinded fixture")
        run, lumi, event, count = array("I", [1]), array("I", [1]), array("Q", [0]), array("I", [0])
        masses = array("d", [0.] * 10)
        alternate = array("d", [999.123456] * 10)
        tree.Branch("run", run, "run/i")
        tree.Branch("luminosityBlock", lumi, "luminosityBlock/i")
        tree.Branch("event", event, "event/l")
        tree.Branch("nShiftDimuonVertex", count, "nShiftDimuonVertex/i")
        if not missing_mass:
            tree.Branch("ShiftDimuonVertex_mass", masses,
                        "ShiftDimuonVertex_mass[10]/D" if wrong_count else
                        "ShiftDimuonVertex_mass[nShiftDimuonVertex]/D")
        tree.Branch("ShiftDimuonVertex_refittedMass", alternate,
                    "ShiftDimuonVertex_refittedMass[nShiftDimuonVertex]/D")
        for identity, values in rows:
            event[0], count[0] = identity, len(values)
            for index, mass in enumerate(values):
                masses[index] = mass
            tree.Fill()
        tree.Write()
        source.Close()

    def test_real_root_reader_gates_mixed_events_and_deduplicates_sorted_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.root", Path(directory) / "b.root"
            self.fixture(first, [(2 ** 64 - 1, [3.05, 666.123456]), (2, [])])
            self.fixture(second, [(2 ** 64 - 1, [3.25]), (3, [777.234567])])
            manifest = self.manifest([second, first])
            result = blind.root_summary([second, first], policy(), "mc-as-data", provenance_manifest=manifest)
            self.assertEqual(result["unique_processed_events"], 3)
            self.assertEqual(result["allowed_pairs"], 1)
            self.assertEqual(result["mass_bin_counts"], [0, 1, 0, 0])
            self.assertEqual(result["duplicate_events_skipped"], 1)
            for canary in ("666.123456", "777.234567", "999.123456"):
                self.assertNotIn(canary, json.dumps(result))

    def test_missing_schema_wrong_count_and_nonfinite_mass_produce_no_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            for variant in ("missing", "wrong_count", "nan"):
                path = Path(directory) / (variant + ".root")
                self.fixture(path, [(1, [math.nan] if variant == "nan" else [3.1])],
                             missing_mass=variant == "missing", wrong_count=variant == "wrong_count")
                with self.subTest(variant=variant), self.assertRaises(blind.BlindingError):
                    blind.root_summary([path], policy(), "mc-as-data", provenance_manifest=self.manifest([path]))

    def test_stale_uuid_fails_before_candidate_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.root"
            self.fixture(path, [(1, [3.1])])
            manifest = self.manifest([path])
            manifest["inputs"][0]["root_uuid"] = "stale-uuid"
            with self.assertRaises(blind.BlindingError):
                blind.root_summary([path], policy(), "mc-as-data", provenance_manifest=manifest)

    def test_same_uuid_root_update_is_rejected_by_actual_byte_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.root"
            self.fixture(path, [(1, [3.1])])
            manifest = self.manifest([path])
            source = ROOT.TFile.Open(str(path), "UPDATE")
            before_uuid = source.GetUUID().AsString()
            ROOT.TObjString("synthetic changed bytes").Write("ChangedPayload")
            source.Close()
            source = ROOT.TFile.Open(str(path))
            self.assertEqual(source.GetUUID().AsString(), before_uuid)
            source.Close()
            self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["inputs"][0]["sha256"])
            with self.assertRaises(blind.BlindingError):
                blind.root_summary([path], policy(), "mc-as-data", provenance_manifest=manifest)

    def test_cli_publishes_only_allowed_aggregates_without_console_or_temporary_canaries(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "input.root"
            self.fixture(path, [(1, [3.05, 666.123456]), (2, [777.234567])])
            result = self.run_cli(directory, [path], self.manifest([path]))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for target in directory.iterdir():
                if target.suffix != ".root":
                    for canary in ("666.123456", "777.234567", "999.123456", "refittedMass"):
                        self.assertNotIn(canary, target.read_text() + result.stdout + result.stderr)
            self.assertEqual({item.name for item in directory.iterdir()},
                             {"input.root", "policy.json", "manifest.json", "summary.json"})

    def test_cli_bad_mass_and_mislabeled_origin_fail_without_output_or_value_logs(self):
        for malformed in ("nan", "mislabeled"):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                path = directory / "input.root"
                self.fixture(path, [(1, [math.nan] if malformed == "nan" else [666.123456])])
                manifest = self.manifest([path])
                if malformed == "mislabeled":
                    manifest["inputs"][0]["origin"] = "collision-data"
                result = self.run_cli(directory, [path], manifest)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((directory / "summary.json").exists())
                for canary in ("666.123456", "999.123456", "refittedMass"):
                    self.assertNotIn(canary, result.stdout + result.stderr)
                self.assertEqual({item.name for item in directory.iterdir()},
                                 {"input.root", "policy.json", "manifest.json"})


if __name__ == "__main__":
    unittest.main()
