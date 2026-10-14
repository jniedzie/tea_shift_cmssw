# Code style: AI-code. AI-generated feature identity and association regression tests.
import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('rich',Path(__file__).parents[1]/'utils/shift_vertex_reco_information.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RichFeatureTests(unittest.TestCase):
    def fixture(self):
        a={k:[[1.,3.]] for k in ('ShiftMuon_'+s for s in m.QUALITY+m.DETAIL)}
        a.update({'run':[2],'luminosityBlock':[3],'event':[2**60+17],'nShiftMuon':[2],
          'ShiftMuon_pt':[[10.,20.]],'ShiftDimuonVertex_muonIdx1':[[0]],'ShiftDimuonVertex_muonIdx2':[[1]]})
        for k in m.EVENT[1:]: a[k]=[7]
        e={'ids':np.array([[2,3,2**60+17]],dtype=np.uint64),'event_index':[0],'pair_index':[0],'muon_indices':[[0,1]]}
        return a,e
    def test_leg_association_and_relative_error(self):
        a,e=self.fixture();x=m.extract_extension(a,e)[0]
        self.assertEqual(x[m.FEATURE_NAMES.index('mu_highPt_ptErr')],3.)
        self.assertEqual(x[m.FEATURE_NAMES.index('mu_highPt_relativePtErr')],.15)
        self.assertEqual(x[m.FEATURE_NAMES.index('mu_lowPt_ptErr')],1.)
    def test_association_adds_information_lost_by_minmax(self):
        a,e=self.fixture();x=m.extract_extension(a,e)
        a['ShiftMuon_ptErr']=[[3.,1.]];other=m.extract_extension(a,e)
        self.assertEqual(sorted([1.,3.]),sorted(a['ShiftMuon_ptErr'][0]))
        self.assertNotEqual(x[0,m.FEATURE_NAMES.index('mu_highPt_ptErr')],
                            other[0,m.FEATURE_NAMES.index('mu_highPt_ptErr')])
    def test_invalid_self_pair_and_collection_length_rejected(self):
        a,e=self.fixture();a['ShiftMuon_nValidHits']=[[1.]]
        with self.assertRaisesRegex(ValueError,'length'):m.extract_extension(a,e)
        a,e=self.fixture();a['ShiftDimuonVertex_muonIdx2']=[[0]];e['muon_indices']=[[0,0]]
        with self.assertRaisesRegex(ValueError,'Invalid'):m.extract_extension(a,e)
    def test_uint64_identity_mismatch_rejected(self):
        a,e=self.fixture();a['event'][0]+=1
        with self.assertRaisesRegex(ValueError,'identity'):m.extract_extension(a,e)
    def test_swapped_pair_rejected(self):
        a,e=self.fixture();e['muon_indices']=[[1,0]]
        with self.assertRaisesRegex(ValueError,'references'):m.extract_extension(a,e)
    def test_tie_order_and_unknown_error(self):
        a,e=self.fixture();a['ShiftMuon_pt']=[[10.,10.]];a['ShiftMuon_ptErr']=[[-1.,3.]]
        x=m.extract_extension(a,e)[0]
        self.assertTrue(np.isnan(x[m.FEATURE_NAMES.index('mu_highPt_relativePtErr')]))
    def test_truth_and_metadata_not_predictors(self):
        self.assertFalse(any(s in k.lower() for k in m.FEATURE_NAMES for s in ('truth','gen','weight','source','event_index','process')))
        a,e=self.fixture();base=m.extract_extension(a,e);a['GenPart_pdgId']=[[999]]
        np.testing.assert_array_equal(base,m.extract_extension(a,e))
if __name__=='__main__':unittest.main()
