# Code style: AI-code. AI-generated identity and signed-weight regression tests.
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from shift_bdt_category_diagnostics import fractions, roc, validate_identity


class CategoryDiagnosticsTest(unittest.TestCase):
    def fixture(self):
        reco = {'ids':np.array([[1,1,10],[1,1,11],[1,1,12]],dtype=np.uint64),
                'process':np.array(['dy']*3), 'pair_index':np.zeros(3,dtype=int),
                'source_index':np.arange(3), 'topology':np.array(['Both-Both']*3)}
        scores = {key:value.copy() for key,value in reco.items() if key != 'topology'}
        scores.update(expanded_split=np.array([0,2,2]), expanded_heldout=np.array([False,True,True]))
        return reco, {'y':np.array([1,1,0])}, scores

    def test_identity_accepts_exact_population(self):
        self.assertEqual(validate_identity(*self.fixture()).tolist(),[False,True,True])

    def test_swapped_source_rejected(self):
        reco, labels, scores = self.fixture()
        scores['source_index'][1:]=scores['source_index'][1:][::-1]
        with self.assertRaisesRegex(ValueError,'identity mismatch'): validate_identity(reco,labels,scores)

    def test_duplicate_candidate_rejected(self):
        reco, labels, scores = self.fixture()
        reco['ids'][2]=reco['ids'][1]; scores['ids']=reco['ids'].copy()
        with self.assertRaisesRegex(ValueError,'Duplicate'): validate_identity(reco,labels,scores)

    def test_parent_split_leak_rejected(self):
        reco, labels, scores = self.fixture()
        reco['ids'][1]=reco['ids'][0]; scores['ids']=reco['ids'].copy()
        reco['pair_index'][1]=1; scores['pair_index']=reco['pair_index'].copy()
        with self.assertRaisesRegex(ValueError,'crosses splits'): validate_identity(reco,labels,scores)

    def test_wrong_test_mask_rejected(self):
        reco, labels, scores = self.fixture(); scores['expanded_heldout'][0]=True
        with self.assertRaisesRegex(ValueError,'test mask'): validate_identity(reco,labels,scores)

    def test_signed_weights_preserved_without_clipping(self):
        row=fractions(np.array([True,True]),np.array([True,False]),{'sampling':np.array([2.,-1.])})
        self.assertEqual(row['sampling']['fraction'],2.)
        self.assertEqual(row['sampling']['negative_parents'],1)
        self.assertEqual(row['raw'],.5)

    def test_roc_ties_not_order_dependent(self):
        for labels in [np.array([1,0]),np.array([0,1])]:
            efficiency,rejection,auc=roc(np.array([.5,.5]),labels)
            self.assertEqual(auc,.5)
            self.assertEqual(len(efficiency),2)


if __name__ == '__main__':
    unittest.main()
