# Code style: AI-code. AI-generated new-campaign histogram normalization tests.
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'configs'))
sys.path.insert(0, str(REPOSITORY / 'bin'))


class CampaignNormalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import ROOT
        except ImportError:
            raise unittest.SkipTest('ROOT is required')
        cls.ROOT = ROOT

    def test_DY_open_edge_alias_keeps_native_cross_section_and_exposure(self):
        from shift_sample_paths import histogram_gen_normalization
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'histograms.root'
            root = self.ROOT.TFile(str(path), 'RECREATE')
            metadata = dict(schema='shift-preserved-full-GEN-histogram-v1', stratum='dy_20toinf',
                full_source_sumw=1952, cross_section_pb=1.27,
                histogram_weight='native_gen_weight/reco_inclusion_probability',
                no_second_filter_efficiency=True)
            self.ROOT.TObjString(json.dumps(metadata)).Write('ShiftFullGenNormalization')
            root.Close()
            result = histogram_gen_normalization(str(path), 'dy_20to-1')
            self.assertEqual(result, dict(cross_section_pb=1.27, initial_weight_sum=1952))
            self.assertEqual(histogram_gen_normalization(str(path), 'dy_20toinf'), result)
            for wrong in ['qcd_20toinf', 'dy_10to20']:
                with self.assertRaises(ValueError):
                    histogram_gen_normalization(str(path), wrong)


if __name__ == '__main__':
    unittest.main()
