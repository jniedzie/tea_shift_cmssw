# Code style: AI-code. AI-generated tests for frozen-prefix diagnostic guards.
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
from shift_bdt_overtraining_diagnostics import prefix_scores,validation_cut,choose_prefix


class PrefixDiagnosticsTest(unittest.TestCase):
    def test_prefix_uses_only_its_own_original_coefficients(self):
        left=dict(children_left=[1,-1,-1],children_right=[2,-1,-1],feature=[0,-2,-2],threshold=[1.,-2.,-2.],leaf_class=[None,0,1],coefficient=2.)
        right=dict(left,leaf_class=[None,1,0],coefficient=1.)
        model=dict(columns=[0],imputer_statistics=[0.],trees=[left,right]);result=prefix_scores(model,np.array([[0.],[2.]]),(1,2))
        np.testing.assert_allclose(result[1],1/(1+np.exp(-2*np.array([-1.,1.]))))
        np.testing.assert_allclose(result[2],1/(1+np.exp(-2*np.array([-1.,1.])/3)))

    def test_cut_does_not_use_training_or_test_scores(self):
        y=np.ones(12);split=np.array([0]+[1]*10+[2]);score=np.arange(12.)
        before=validation_cut(score,y,split);score[split!=1]=1e9
        self.assertEqual(before,validation_cut(score,y,split))

    def test_prefix_choice_ignores_test_performance(self):
        rows={str(n):dict(metrics={part:dict(sm=dict(accidental_rejection=value,auc=value)) for part,value in [('validation',validation),('test',test)]}) for n,validation,test in [(50,.7,.1),(400,.6,.99)]}
        self.assertEqual(choose_prefix(rows),'50')
        rows['400']['metrics']['test']['sm']['accidental_rejection']=1.
        self.assertEqual(choose_prefix(rows),'50')


if __name__=='__main__':unittest.main()
