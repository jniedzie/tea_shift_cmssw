# Code style: AI-code. AI-generated historical physical-group regression tests.
import importlib.util
from pathlib import Path
import sys
import unittest
import types
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
import shift_vertex_legacy_cache as m


class HistoricalIdentityTests(unittest.TestCase):
    def test_large_uint64_events_distinct(self):
        ids=np.array([[1,1,2**60+1],[1,1,2**60+2]],dtype=np.uint64)
        keys=m.parent_keys(ids,'original-source')
        self.assertNotEqual(keys[0],keys[1])
        np.testing.assert_array_equal(keys,m.parent_keys(ids,'original-source'))

    def test_same_raw_ids_different_domains_do_not_collide(self):
        ids=np.array([[1,1,1]],dtype=np.uint64)
        self.assertNotEqual(m.parent_keys(ids,'qcd-bin1')[0],m.parent_keys(ids,'jpsi-bin1')[0])

    def test_duplicates_and_float_ids_rejected(self):
        for ids in [np.array([[1,1,1],[1,1,1]]),np.array([[1.,1.,1.]])]:
            with self.assertRaises(ValueError):m.parent_keys(ids,'context')

    def test_context_is_relocation_invariant_and_recipe_bound(self):
        source=dict(process='qcd',stratum='qcd_1to2',original_campaign='/original/campaign',
                    checkpoint={'sha256':'frozen'},generator_process='QCD',nano={'path':'old'})
        checkpoint={'contract':{'SHIFT_LSS_FIELD_MODE':'ir1_atlas_proxy','PROCESS':'QCD'}}
        first=m.source_context(source,checkpoint)
        source['nano']['path']='relocated';self.assertEqual(first,m.source_context(source,checkpoint))
        source['checkpoint']['sha256']='different';self.assertNotEqual(first,m.source_context(source,checkpoint))
        checkpoint['contract']['SHIFT_LSS_FIELD_MODE']='cms_ir5'
        with self.assertRaisesRegex(ValueError,'geometry'):m.source_context(source,checkpoint)

    def chunk_fixture(self):
        import awkward as ak
        a=dict(run=np.array([7,7],dtype=np.uint32),luminosityBlock=np.array([1,1],dtype=np.uint32),
               event=np.array([2**60+1,2**60+2],dtype=np.uint64),
               genWeight=np.ones(2),nShiftDimuonVertex=np.array([0,1]))
        for key in ['topologyMin','topologyMax','isOS','dcaValid','vx','vy','vz']:
            a['ShiftDimuonVertex_'+key]=ak.Array([[],[2 if key.startswith('topology') else 1]])
        def raw(arrays):
            return dict(X=np.ones((1,50)),ids=np.array([[7,1,2**60+2]],dtype=np.uint64),
                        event_index=np.array([1]),pair_index=np.array([0]),muon_indices=np.array([[0,1]]),
                        nuisance=np.array([[1.,2.,3.]]))
        features=types.SimpleNamespace(RECO_BRANCHES=tuple(a),FEATURE_NAMES=tuple('x'+str(i) for i in range(50)),extract_reco=raw)
        rich=types.SimpleNamespace(BRANCHES=tuple(a),QUALITY=(),DETAIL=(),
                                  FEATURE_NAMES=tuple('extra'+str(i) for i in range(69)),
                                  extract_extension=lambda arrays,reco:np.ones((1,69)))
        labels=types.SimpleNamespace(extract_labels=lambda arrays,reco:(np.array([1]),np.array(['common_vertex'])),
                                    label_diagnostics=lambda arrays,reco:{})
        modules=dict(features=features,labels=labels,shift_vertex_reco_information=rich)
        source=dict(process='qcd',stratum='qcd_1to2',identity_context='legacy_qcd_1to2')
        return a,modules,source

    def test_chunk_offsets_and_zero_pair_parent_preserved(self):
        a,modules,source=self.chunk_fixture()
        reco,labels,events=m.extract_chunk(a,{},modules,source,3,5000,'original')
        self.assertEqual(reco['event_index'].tolist(),[5001])
        self.assertEqual(events['event_index'].tolist(),[5000,5001])
        self.assertEqual(events['pair_count'].tolist(),[0,1])
        self.assertEqual(reco['physical_parent_key'][0],events['physical_parent_key'][1])
        self.assertEqual(reco['source_index'].tolist(),[3])
        self.assertEqual(reco['X'].shape,(1,119))

    def test_changed_native_weight_rejected(self):
        a,modules,source=self.chunk_fixture();a['genWeight'][1]=2.
        with self.assertRaisesRegex(ValueError,'weight'):
            m.extract_chunk(a,{},modules,source,3,5000,'original')


if __name__=='__main__':unittest.main()
