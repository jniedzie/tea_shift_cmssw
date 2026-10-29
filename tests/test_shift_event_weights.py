# Code style: AI-code. AI-generated checks for independent decay and detector weights.
from array import array
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "event_weights", Path(__file__).resolve().parents[1] / "utils" / "shift_event_weights.py")
weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(weights)
try:
    import ROOT
except ImportError:
    ROOT = None


def row(event=1, native=2., probability=.25, correction=.125):
    return dict(event_id=[1, 1, event], source_id="original-source",
        original_source_sha256="a" * 64, proposal_draw_id=0,
        native_weight=native, sampling_probability=probability,
        sampling_gen_weight=native / probability,
        proposal_density_ratio=1. / correction, proposal_correction=correction,
        proposal_gen_weight=(native / probability) * correction)


EXPOSURE = dict(source_id="original-source", original_source_sha256="a" * 64,
                full_source_trials=1000, full_source_sumw=1000.)


class ProposalWeightTests(unittest.TestCase):
    def test_native_LO_exposure_is_not_mislabeled_as_fixed_source_trials(self):
        exposure = dict(source_id='original-source', original_source_sha256='a'*64,
                        denominator_kind='accepted_gen_events', full_source_events=1000)
        result = weights.validate_proposal_rows([row(correction=1.)], [exposure])[0]
        self.assertEqual(result['shiftProposalGenWeight'], 8.)
        for bad in (dict(exposure, full_source_trials=1000),
                    dict(exposure, full_source_events=0),
                    dict(exposure, denominator_kind='unknown')):
            with self.assertRaises(ValueError): weights.validate_proposal_rows([row()], [bad])
        trial = dict(EXPOSURE, full_source_events=1000)
        with self.assertRaises(ValueError): weights.validate_proposal_rows([row()], [trial])

    def test_old_detector_correction_and_new_likelihood_enter_once(self):
        result = weights.validate_proposal_rows([row()])[0]
        self.assertEqual(result["shiftOriginalGenWeight"], 2.)
        self.assertEqual(result["shiftSamplingGenWeight"], 8.)
        self.assertEqual(result["shiftDecayProposalWeight"], .125)
        self.assertEqual(result["shiftProposalGenWeight"], 1.)

    def test_decay_weight_above_one_is_not_an_inclusion_probability(self):
        result = weights.validate_proposal_rows([row(native=-2., probability=.5, correction=8.)])[0]
        self.assertEqual(result["shiftSamplingProbability"], .5)
        self.assertEqual(result["shiftDecayProposalWeight"], 8.)
        self.assertEqual(result["shiftProposalGenWeight"], -32.)

    def test_zero_signed_and_tiny_native_weights_are_retained(self):
        rows = [row(event=1, native=0.), row(event=2, native=-1.), row(event=3, native=1.e-100)]
        result = weights.validate_proposal_rows(rows)
        self.assertEqual([r["shiftProposalGenWeight"] for r in result[:2]], [0., -.5])
        self.assertTrue(math.isclose(result[2]["shiftProposalGenWeight"], 5.e-101, rel_tol=1.e-14))

    def test_declared_density_and_final_weight_must_match_exact_contract(self):
        for key, value in (("proposal_density_ratio", 4.), ("proposal_gen_weight", 8.),
                           ("sampling_gen_weight", 32.)):
            candidate = row()
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                weights.validate_proposal_rows([candidate])

    def test_nonfinite_zero_support_and_probability_above_one_are_rejected(self):
        for key, values in (("sampling_probability", [0., -.1, 1.1, math.nan]),
                            ("proposal_density_ratio", [0., -1., math.inf]),
                            ("proposal_correction", [0., -1., math.nan]),
                            ("native_weight", [math.inf, True])):
            for value in values:
                candidate = row()
                candidate[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    weights.validate_proposal_rows([candidate])

    def test_final_double_underflow_and_overflow_are_rejected(self):
        for candidate in (row(native=1.e-200, correction=1.e-200),
                          row(native=1.e200, probability=1., correction=1.e200)):
            with self.assertRaisesRegex(ValueError, "Double support"):
                weights.validate_proposal_rows([candidate])

    def test_repeated_originals_are_rejected_even_with_different_draws(self):
        first, second = row(), row()
        second["proposal_draw_id"] = 1
        with self.assertRaisesRegex(ValueError, "Repeated original"):
            weights.validate_proposal_rows([first, second])

    def test_original_source_draw_binding_is_required(self):
        for key, value in (("source_id", ""), ("original_source_sha256", "bad"),
                           ("proposal_draw_id", None), ("proposal_draw_id", True)):
            candidate = row()
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                weights.validate_proposal_rows([candidate])

    def test_full_source_exposure_is_separate_and_cannot_be_counted_twice(self):
        result = weights.validate_proposal_rows([row()], [EXPOSURE])
        self.assertEqual(result[0]["shiftProposalGenWeight"], 1.)
        self.assertEqual(EXPOSURE["full_source_trials"], 1000)
        with self.assertRaisesRegex(ValueError, "Repeated original source exposure"):
            weights.validate_proposal_rows([row()], [EXPOSURE, EXPOSURE])
        with self.assertRaisesRegex(ValueError, "exact"):
            weights.validate_proposal_rows([row()], [])

    def test_zero_admitted_sources_remain_in_the_full_exposure(self):
        empty_source = dict(source_id="zero-admitted-source", original_source_sha256="b" * 64,
                            full_source_trials=2000)
        exposure = [EXPOSURE, empty_source]
        result = weights.validate_proposal_rows([row()], exposure)
        self.assertEqual(len(result), 1)
        self.assertEqual(sum(source["full_source_trials"] for source in exposure), 3000)
        self.assertEqual(weights.validate_proposal_rows([], exposure), [])
        with self.assertRaises(ValueError):
            weights.validate_proposal_rows([])

    def test_explicitly_rejected_original_cannot_be_histogrammed(self):
        candidate = dict(row(), admitted=False)
        with self.assertRaisesRegex(ValueError, "Only admitted"):
            weights.validate_proposal_rows([candidate])

    def test_ordinary_full_event_proposal_has_unit_likelihood(self):
        candidate = dict(row(correction=1.), proposal_kind="ordinary")
        self.assertEqual(weights.validate_proposal_rows([candidate])[0]["shiftDecayProposalWeight"], 1.)
        with self.assertRaisesRegex(ValueError, "ordinary full-event"):
            weights.validate_proposal_rows([dict(row(), proposal_kind="ordinary")])


@unittest.skipUnless(ROOT is not None, "Run inside the native ROOT runtime")
class ProposalNanoTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.path = Path(self.scratch.name) / "nano.root"

    def fixture(self, rows, detector_type=None):
        output = ROOT.TFile(str(self.path), "RECREATE")
        tree = ROOT.TTree("Events", "Frozen native event payload")
        integers = {key: array("Q", [0]) for key in ("run", "luminosityBlock", "event")}
        for name, buffer in integers.items():
            tree.Branch(name, buffer, name + "/l")
        native = array("f", [0.])
        tree.Branch("genWeight", native, "genWeight/F")
        detector = {}
        if detector_type:
            for name in weights.DETECTOR_BRANCHES:
                detector[name] = array("d" if detector_type == "D" else "f", [0.])
                tree.Branch(name, detector[name], name + "/" + detector_type)
        for original in rows:
            for name, value in zip(integers, original["event_id"]):
                integers[name][0] = value
            native[0] = original["native_weight"]
            expected = weights.validate_proposal_rows([original])[0]
            for name, buffer in detector.items():
                buffer[0] = expected[name]
            tree.Fill()
        tree.Write()
        runs = ROOT.TTree("Runs", "Full original GEN denominator")
        count, sumw = array("q", [1000]), array("d", [1000.])
        runs.Branch("genEventCount", count, "genEventCount/L")
        runs.Branch("genEventSumw", sumw, "genEventSumw/D")
        runs.Fill()
        runs.Write()
        ROOT.TObjString("original metadata retained byte for byte").Write("OriginalMetadata")
        output.Close()

    def annotate(self, rows, expected=None):
        expected = expected or hashlib.sha256(self.path.read_bytes()).hexdigest()
        return weights.annotate_proposal_weights(self.path, rows, [EXPOSURE], expected_nano_sha256=expected)

    def test_double_fields_preserve_native_payload_and_full_denominator(self):
        rows = [row(event=1, native=1.e-100), row(event=2, native=-2., correction=8.), row(event=3, native=0.)]
        self.fixture(rows)
        result = self.annotate(rows)
        self.assertFalse(result["normalization_ready"])
        self.assertTrue(result["one_draw_per_original"])
        source = ROOT.TFile.Open(str(self.path))
        tree = source.Get("Events")
        for name in weights.DETECTOR_BRANCHES + weights.PROPOSAL_BRANCHES:
            self.assertEqual(tree.GetLeaf(name).GetTypeName(), "Double_t")
        for event, original in zip(tree, rows):
            expected_float = struct.unpack("f", struct.pack("f", original["native_weight"]))[0]
            self.assertEqual(float(event.genWeight), expected_float)
            self.assertEqual(float(event.shiftProposalGenWeight), original["proposal_gen_weight"])
        runs = source.Get("Runs")
        runs.GetEntry(0)
        self.assertEqual(int(runs.genEventCount), 1000)
        self.assertEqual(float(runs.genEventSumw), 1000.)
        self.assertEqual(str(source.Get("OriginalMetadata").GetString()), "original metadata retained byte for byte")
        source.Close()

    def test_native_LO_annotation_preserves_full_run_payload_without_invented_trials(self):
        rows = [row(native=1., correction=1.)]
        self.fixture(rows)
        exposure = dict(source_id='original-source', original_source_sha256='a'*64,
                        denominator_kind='accepted_gen_events', full_source_events=1000)
        result = weights.annotate_proposal_weights(self.path, rows, [exposure],
            expected_nano_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(result['source_exposures'], [exposure])
        self.assertFalse(result['normalization_ready'])
        source = ROOT.TFile.Open(str(self.path))
        event = source.Get('Events'); event.GetEntry(0)
        self.assertEqual(float(event.shiftProposalGenWeight), 4.)
        self.assertEqual(float(event.genWeight), 1.)
        run = source.Get('Runs'); run.GetEntry(0)
        self.assertEqual(int(run.genEventCount), 1000)
        source.Close()

    def test_existing_detector_weights_stay_unchanged_and_cannot_be_applied_twice(self):
        rows = [row()]
        self.fixture(rows, "D")
        self.annotate(rows)
        with self.assertRaisesRegex(ValueError, "second correction"):
            self.annotate(rows)

    def test_float_detector_fields_are_not_silently_promoted(self):
        rows = [row()]
        self.fixture(rows, "F")
        with self.assertRaisesRegex(ValueError, "must be Double"):
            self.annotate(rows)

    def test_equal_row_count_does_not_allow_a_different_original_identity(self):
        self.fixture([row()])
        with self.assertRaisesRegex(ValueError, "identities differ"):
            self.annotate([row(event=2)])

    def test_wrong_Nano_sibling_with_same_event_and_native_weight_is_rejected(self):
        rows = [row()]
        self.fixture(rows)
        expected = hashlib.sha256(self.path.read_bytes()).hexdigest()
        source = ROOT.TFile.Open(str(self.path), "UPDATE")
        ROOT.TObjString("different ordinary/biased GEN draw").Write("DrawAuthority")
        source.Close()
        with self.assertRaisesRegex(ValueError, "graph/draw receipt"):
            self.annotate(rows, expected)

    def test_zero_admitted_Nano_keeps_full_original_exposure(self):
        self.fixture([])
        result = self.annotate([])
        self.assertEqual(result["events"], 0)
        self.assertEqual(result["source_exposures"], [EXPOSURE])
        self.assertFalse(result["normalization_ready"])


def response_row(event=1, native=2., probability=.25, q=.1, population=10, selected=1):
    return dict(row(event,native,probability,correction=1.), response_draw_id='d'*64,
        response_sampling_probability=q,response_population_events=population,
        response_selected_events=selected)


class ResponseWeightTests(unittest.TestCase):
    def test_p_and_q_enter_once_and_native_detector_weights_stay_separate(self):
        result = weights.validate_response_sampling_rows([response_row()],[EXPOSURE])[0]
        self.assertEqual(result['shiftOriginalGenWeight'],2.)
        self.assertEqual(result['shiftSamplingGenWeight'],8.)
        self.assertEqual(result['shiftResponseSamplingWeight'],10.)
        self.assertEqual(result['shiftResponseGenWeight'],80.)
        self.assertNotIn('shiftDecayProposalWeight',result)

    def test_response_cell_probability_must_match_complete_population(self):
        for bad in [dict(response_row(),response_sampling_probability=.2),
                    dict(response_row(),response_population_events=0),
                    dict(response_row(),response_selected_events=11),
                    dict(response_row(),response_draw_id='unfrozen')]:
            with self.assertRaises(ValueError):
                weights.validate_response_sampling_rows([bad],[EXPOSURE])

    def test_missing_or_repeated_full_exposure_and_originals_are_rejected(self):
        for exposures in (None,[],[EXPOSURE,EXPOSURE]):
            with self.assertRaises(ValueError):
                weights.validate_response_sampling_rows([response_row()],exposures)
        with self.assertRaisesRegex(ValueError,'Repeated original'):
            weights.validate_response_sampling_rows([response_row(),response_row()],[EXPOSURE])

    def test_signed_tiny_zero_and_double_overflow(self):
        rows = [response_row(1,0.),response_row(2,-2.),response_row(3,1.e-100)]
        actual = weights.validate_response_sampling_rows(rows,[EXPOSURE])
        self.assertEqual(actual[0]['shiftResponseGenWeight'],0.)
        self.assertEqual(actual[1]['shiftResponseGenWeight'],-80.)
        self.assertTrue(math.isclose(actual[2]['shiftResponseGenWeight'],4.e-99,rel_tol=1.e-14))
        with self.assertRaisesRegex(ValueError,'Double support'):
            weights.validate_response_sampling_rows([response_row(native=1.e307)],[EXPOSURE])


@unittest.skipUnless(ROOT is not None, 'Run inside the native ROOT runtime')
class ResponseNanoTests(unittest.TestCase):
    setUp = ProposalNanoTests.setUp
    fixture = ProposalNanoTests.fixture

    def annotate(self, rows, expected=None):
        return weights.annotate_response_sampling_weights(self.path,rows,[EXPOSURE],
            expected_nano_sha256=expected or hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_appended_double_response_weights_preserve_detector_and_native_run_data(self):
        rows = [response_row(1),response_row(2,native=1.e-100,q=1.,population=1)]
        self.fixture(rows,'D')
        original = ROOT.TFile.Open(str(self.path))
        untouched = weights._key_payloads(self.path,original)
        original.Close()
        annotation = self.annotate(rows)
        self.assertEqual(annotation['response_correction_application_count'],1)
        self.assertEqual(annotation['source_exposures'],[EXPOSURE])
        output = ROOT.TFile.Open(str(self.path))
        self.assertEqual({k:v for k,v in weights._key_payloads(self.path,output).items()
            if k[0]!=weights.RESPONSE_METADATA_KEY},untouched)
        tree = output.Get('Events')
        for name in weights.RESPONSE_BRANCHES:
            self.assertEqual(tree.GetLeaf(name).GetTypeName(),'Double_t')
        for event,expected in zip(tree,weights.validate_response_sampling_rows(rows,[EXPOSURE])):
            for name in weights.DETECTOR_BRANCHES+weights.RESPONSE_BRANCHES:
                self.assertEqual(float(getattr(event,name)),expected[name])
        output.Close()

    def test_second_response_correction_and_unbound_input_are_rejected(self):
        rows = [response_row()]
        self.fixture(rows,'D')
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            self.annotate(rows,expected='0'*64)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),before)
        self.annotate(rows)
        with self.assertRaisesRegex(ValueError,'second correction'):
            self.annotate(rows)

    def test_wrong_old_p_or_missing_double_contract_cannot_be_hidden(self):
        rows = [response_row()]
        self.fixture(rows,'D')
        with self.assertRaisesRegex(ValueError,'unchanged shiftSampling'):
            self.annotate([dict(response_row(),sampling_probability=.5)])
        self.path.unlink()
        self.fixture(rows)
        with self.assertRaisesRegex(ValueError,'existing detector Double'):
            self.annotate(rows)


if __name__ == "__main__":
    unittest.main()
