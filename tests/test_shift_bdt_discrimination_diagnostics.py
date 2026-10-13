"""Guard truth-class diagnostics against selection leakage and score cosmetics."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC=importlib.util.spec_from_file_location('diagnostics',Path(__file__).resolve().parents[1]/'utils/shift_bdt_discrimination_diagnostics.py')
d=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(d)


class DiagnosticsTest(unittest.TestCase):
    def test_test_partition_cannot_change_feature_direction_or_rank(self):
        X=np.array([[0.,4.],[1.,3.],[2.,2.],[3.,1.],[0.,3.],[3.,1.],[5.,5.],[6.,6.]])
        y=np.array([0,0,1,1,0,1,0,1]);split=np.array([0,0,0,0,1,1,2,2]);process=np.full(8,'dy')
        first=d.single_feature_audit(X,y,process,split,['a','b'],['a'])
        X[split==2]=np.nan;y[split==2]=1
        self.assertEqual(first,d.single_feature_audit(X,y,process,split,['a','b'],['a']))

    def test_signed_coordinates_are_not_missing_sentinels(self):
        result=d.input_audit(np.array([[-1.,np.nan,0.],[-5.,-1.,0.]]),['pair_vz','mu_min_ptErr','mu_min_nLostHits'],[1.,2.,0.])
        self.assertIsNone(result['per_feature'][0]['negative_in_nonnegative_field'])
        self.assertEqual(result['per_feature'][1]['negative_in_nonnegative_field'],1)
        self.assertEqual(result['per_feature'][1]['imputed_rows'],1)
        self.assertEqual(result['per_feature'][2]['imputed_rows'],0)

    def test_tree_votes_keep_original_branching_and_training_imputation(self):
        model=dict(columns=[0],imputer_statistics=[2.],trees=[dict(children_left=[1,-1,-1],children_right=[2,-1,-1],
            feature=[0,-2,-2],threshold=[1.,-2.,-2.],leaf_class=[None,0,1])])
        np.testing.assert_array_equal(d.tree_votes(model,np.array([[0.],[1.],[2.],[np.nan]])),[[-1],[-1],[1],[1]])

    def test_score_range_is_uncalibrated_monotonic_vote_mapping(self):
        from sklearn.metrics import roc_auc_score
        votes=np.array([-.8,-.1,.3,.9]);score=1/(1+np.exp(-2*votes));y=[0,1,0,1]
        self.assertAlmostEqual(roc_auc_score(y,votes),roc_auc_score(y,score))
        self.assertTrue(np.all((score>=1/(1+np.exp(2)))&(score<=1/(1+np.exp(-2)))))


if __name__=='__main__':unittest.main()
