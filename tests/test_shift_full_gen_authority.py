# Code style: AI-code. AI-generated surviving full-GEN authority regression tests.
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'configs'))
from shift_sample_paths import full_gen_normalization_authority, full_gen_weight_sums


class PreservedFullGenAuthorityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.campaign = 'shift_detector_representative_20261007_v10'
        self.historical = dict(parent_manifest_sha256='a' * 64, ordinary_manifest_sha256='b' * 64,
                               weighted_manifest_sha256='c' * 64, weighted_accounting_sha256='d' * 64)
        self.plan = {
            'schema': 'shift-detector-pps-sample-v1', 'parent_manifest_sha256': self.historical['parent_manifest_sha256'],
            'normalization_authorities': {
                'ordinary': {'manifest_sha256': self.historical['ordinary_manifest_sha256']},
                'weighted': {'manifest_sha256': self.historical['weighted_manifest_sha256'],
                             'accounting_sha256': self.historical['weighted_accounting_sha256'],
                             'strata': {'qcd_20toinf': {'accepted': 80, 'sumw': .5, 'trials': 100, 'sigma_mb': 2e-6}}},
            },
            'strata': {
                'qcd_0to1': {'original_events': 100, 'original_sumw': 100, 'realized_events': 9,
                    'corrected_sumw': 90, 'original_normalization_records': [
                        {'stratum': 'qcd_0to1', 'events': 100, 'normalization_record': None}]},
                'qcd_20toinf': {'original_events': 80, 'original_sumw': .5, 'realized_events': 10,
                    'corrected_sumw': .4, 'original_normalization_records': [
                        {'stratum': 'qcd_20toinf', 'events': 80, 'normalization_record': 'native-ledger.json'}]},
            },
        }
        self.ordinary = {'complete': True, 'mode': 'complete', 'manifest_sha256': self.historical['ordinary_manifest_sha256'],
                         'totals': {'qcd_0to1': 100},
                         'normalization_inputs': {'qcd_0to1': {'events': 100, 'cross_section_pb': 9.5}}}
        self.authority = {'schema': 'shift-preserved-full-GEN-normalization-v1', 'complete': True,
            'campaign': self.campaign, 'full_original_events': 180, 'recorded_original_hashes': self.historical,
            'strata': {'qcd_0to1': {'events': 100, 'sumw': 100, 'cross_section_pb': 9.5, 'trials': None},
                       'qcd_20toinf': {'events': 80, 'sumw': .5, 'cross_section_pb': 2000., 'trials': 100}}}
        self.refresh()

    def write(self, name, record):
        path = self.root / name
        path.write_text(json.dumps(record))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def refresh(self):
        self.authority['records'] = {
            'sampling_plan': {'file': 'sampling_plan.json', 'sha256': self.write('sampling_plan.json', self.plan)},
            'ordinary_completion': {'file': 'ordinary_complete.json', 'sha256': self.write('ordinary_complete.json', self.ordinary)},
        }
        self.digest = self.write('authority.json', self.authority)

    def read(self):
        return full_gen_normalization_authority(self.campaign, self.root / 'authority.json', self.digest)

    def test_original_exposure_cross_sections_and_trials_survive_missing_historical_paths(self):
        self.assertEqual(self.read(), {
            'qcd_0to1': {'events': 100, 'sumw': 100., 'cross_section_pb': 9.5, 'trials': None},
            'qcd_20toinf': {'events': 80, 'sumw': .5, 'cross_section_pb': 2000., 'trials': 100}})
        self.assertEqual(full_gen_weight_sums(self.campaign, normalization_authority=self.root / 'authority.json',
            normalization_authority_sha256=self.digest), {'qcd_0to1': 100., 'qcd_20toinf': .5})

    def test_detector_counts_and_corrected_cutflow_cannot_change_full_exposure(self):
        self.plan['strata']['qcd_0to1'].update(realized_events=1, corrected_sumw=10)
        self.plan['strata']['qcd_20toinf'].update(realized_events=1, corrected_sumw=.001)
        self.refresh()
        self.assertEqual(self.read()['qcd_20toinf']['sumw'], .5)
        self.assertEqual(self.read()['qcd_0to1']['events'], 100)

    def test_selected_denominator_is_rejected_even_with_updated_authority_hash(self):
        for name, value in (('qcd_0to1', 9), ('qcd_20toinf', .4)):
            with self.subTest(stratum=name):
                original = self.authority['strata'][name]['sumw']
                self.authority['strata'][name]['sumw'] = value
                self.refresh()
                with self.assertRaisesRegex(RuntimeError, 'denominator or cross section differs'):
                    self.read()
                self.authority['strata'][name]['sumw'] = original

    def test_changed_cross_section_is_rejected(self):
        self.authority['strata']['qcd_20toinf']['cross_section_pb'] *= 1.1
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'cross section differs'):
            self.read()

    def test_changed_record_bytes_are_rejected(self):
        (self.root / 'sampling_plan.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'hash differs'):
            self.read()

    def test_changed_authority_bytes_are_rejected(self):
        (self.root / 'authority.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'hash differs'):
            self.read()

    def test_missing_pin_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'requires its frozen SHA-256'):
            full_gen_normalization_authority(self.campaign, self.root / 'authority.json', None)

    def test_wrong_campaign_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'different or incomplete campaign'):
            full_gen_normalization_authority('shift_detector_representative_20261014_v1',
                                             self.root / 'authority.json', self.digest)
        with self.assertRaisesRegex(RuntimeError, 'different or incomplete campaign'):
            full_gen_weight_sums('legacy_campaign', normalization_authority=self.root / 'authority.json',
                                 normalization_authority_sha256=self.digest)

    def test_incomplete_ordinary_originals_are_rejected(self):
        self.ordinary['totals']['qcd_0to1'] = 90
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'Ordinary full GEN exposure differs'):
            self.read()

    def test_missing_stratum_cannot_fall_back_to_selected_cutflow(self):
        del self.authority['strata']['qcd_0to1']
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'inventories differ'):
            self.read()

    def test_weighted_trial_count_is_preserved(self):
        self.authority['strata']['qcd_20toinf']['trials'] = 80
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'denominator or cross section differs'):
            self.read()

    def test_source_accounting_not_selected_counts_defines_exposure(self):
        self.plan['strata']['qcd_20toinf'].update(original_events=10, original_sumw=.4)
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, 'original full GEN exposure'):
            self.read()


if __name__ == '__main__':
    unittest.main()
