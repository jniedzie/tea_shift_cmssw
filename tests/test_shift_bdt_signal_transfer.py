# Code style: AI-code. AI-generated fresh-signal parent/denominator integrity tests.
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'utils'))
from shift_bdt_signal_transfer import validate_signal_parent_alignment


class FreshSignalIdentityTest(unittest.TestCase):
    def fixture(self):
        events={'ids':np.array([[1,1,1],[1,1,2],[2,1,1]],dtype=np.uint64),
                'source_index':np.array([0,0,1]),'point':np.array(['a','a','b']),
                'native_weight':np.array([1.,1.,1.])}
        data={'ids':events['ids'][[1,2]].copy(),'source_index':np.array([0,1]),
              'event_index':np.array([1,0]),'native_weight':np.array([1.,1.]),'point':np.array(['a','b'])}
        labels={'y':np.array([1,-1]),'signal_associated':np.array([True,False])}
        return data,labels,events

    def test_zero_pair_parent_retained(self):
        data,labels,events=self.fixture();validate_signal_parent_alignment(data,labels,events)
        self.assertEqual(len(events['ids']),3);self.assertEqual(len(data['ids']),2)

    def test_changed_parent_index_rejected(self):
        data,labels,events=self.fixture();data['event_index'][0]=0
        with self.assertRaisesRegex(ValueError,'parent identity'):validate_signal_parent_alignment(data,labels,events)

    def test_cross_point_swap_rejected(self):
        data,labels,events=self.fixture();data['point'][0]='b'
        with self.assertRaisesRegex(ValueError,'model point'):validate_signal_parent_alignment(data,labels,events)

    def test_weight_change_rejected(self):
        data,labels,events=self.fixture();data['native_weight'][0]=2
        with self.assertRaisesRegex(ValueError,'parent weight'):validate_signal_parent_alignment(data,labels,events)

    def test_duplicated_parent_rejected(self):
        data,labels,events=self.fixture();events['ids'][2]=events['ids'][0]
        with self.assertRaisesRegex(ValueError,'Repeated parent'):validate_signal_parent_alignment(data,labels,events)

    def test_unknown_not_signal_genuine(self):
        data,labels,events=self.fixture();labels['signal_associated'][1]=True
        with self.assertRaisesRegex(ValueError,'qualified genuine'):validate_signal_parent_alignment(data,labels,events)


if __name__=='__main__':unittest.main()
