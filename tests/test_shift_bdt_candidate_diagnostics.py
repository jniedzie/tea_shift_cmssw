import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
from shift_bdt_candidate_diagnostics import relative_features,fit_population,threshold_counts


class RelativeFeatureTest(unittest.TestCase):
    def test_expanded_test_excludes_new_training_despite_old_heldout(self):
        e=dict(process=np.array(['dy','dy','jpsi','signal']),ids=np.array([[1,1,i] for i in range(4)]),heldout=np.ones(4,dtype=bool))
        s=dict(heldout=e['heldout'],expanded_split=np.array([0,1,2,2]),expanded_heldout=np.array([False,False,True,False]))
        mask,_=fit_population(s,e,'expanded-test');np.testing.assert_array_equal(mask,[False,False,True,False])
        del s['expanded_split']
        with self.assertRaises(ValueError):fit_population(s,e,'expanded-test')

    def test_expanded_event_cannot_cross_partitions_and_mask_cannot_lie(self):
        e=dict(process=np.array(['dy','dy']),ids=np.array([[1,1,8],[1,1,8]]),heldout=np.ones(2,dtype=bool))
        s=dict(expanded_split=np.array([0,2]),expanded_heldout=np.array([False,True]))
        with self.assertRaises(ValueError):fit_population(s,e,'expanded-test')
        s['expanded_split']=np.array([2,2])
        with self.assertRaises(ValueError):fit_population(s,e,'expanded-test')

    def test_variable_validation_count_retains_inclusive_ties(self):
        result=threshold_counts(np.arange(31.),np.ones(31),np.ones(31),3.)
        self.assertEqual(result['genuine_pairs'],31);self.assertEqual(result['passed_pairs'],28)
        with self.assertRaises(ValueError):threshold_counts(np.arange(31.),np.ones(31),np.ones(31),20.)

    def test_leg_exchange_and_spatial_translation_leave_relative_features_unchanged(self):
        fields=['mu_'+side+'_'+k for side in ('highPt','lowPt') for k in ('px','py','pz','vx','vy','vz')]
        fields+=['mu_'+side+'_'+k for side in ('min','max') for k in ('ptErr','nValidMuonHits','normalizedChi2','etaErr','phiErr')]
        fields+=['pair_'+k for k in ('vxErr','vyErr','vzErr','dca','vx','vy','vz')]+['opening_angle']
        X=np.arange(1.,len(fields)+1)[None,:];first,names=relative_features(X,fields)
        other=X.copy()
        for k in ('px','py','pz'):
            a,b=fields.index('mu_highPt_'+k),fields.index('mu_lowPt_'+k);other[:,[a,b]]=other[:,[b,a]]
        for i,name in enumerate(fields):
            if name.endswith(('_vx','_vy','_vz')):other[:,i]+=1e6
        second,_=relative_features(other,fields);np.testing.assert_allclose(first,second)
        for k in ('vxErr','vyErr','vzErr'):other[:,fields.index('pair_'+k)]=0
        result,_=relative_features(other,fields)
        self.assertTrue(np.isnan(result[0,names.index('DCA / total vertex-error scale')]))
        self.assertTrue(np.isnan(result[0,names.index('DCA / transverse vertex-error scale')]))


if __name__=='__main__':unittest.main()
