# Code style: AI-code. AI-generated checks for leakage and sparse-statistics reporting.
import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'utils'))
from shift_classifier_crossfit import digest, fold_masks, ownership_partitions
from shift_classifier_crossfit import task_training_data
from shift_classifier_crossfit_report import compare, empirical_roc, paired_metric, verified_predictions


def dataset():
    n = 200
    return dict(X=np.zeros((n, 121)), feature_names=np.array(['feature'+str(i) for i in range(121)]),
                y=np.tile([0, 1, -1, 1], 50), process=np.full(n, 'qcd'), point=np.full(n, 'qcd-low'),
                category=np.full(n, 'Both-Both'), ids=np.column_stack((np.ones(n, dtype=int), np.ones(n, dtype=int), np.arange(n))),
                pair_index=np.zeros(n, dtype=int), weight=np.ones(n), mass=np.ones(n),
                group_id=np.repeat(['source'+str(i) for i in range(50)], 4),
                event_owner_id=np.array(['original-event'+str(i) for i in range(n)]))


def result(directory, data):
    directory.mkdir(); models = []; cut = np.full(len(data['y']), .7); fold, _ = ownership_partitions(data['group_id'])
    for f in (0, 1):
        train, validation, scored = fold_masks(data, f, 'sm-only'); p=directory/('fold'+str(f));p.mkdir()
        (p/'model.pkl').write_bytes(b'guard-test-model')
        np.savez_compressed(p/'partitions.npz', train=np.flatnonzero(train), validation=np.flatnonzero(validation),
                            scored=np.flatnonzero(scored), validation_scores=np.full(validation.sum(), .7))
        models.append(dict(scored_fold=f, threshold=.7, model_sha256=digest(p/'model.pkl'),
                           groups={name:sorted(set(data['group_id'][mask])) for name,mask in
                                   (('train',train),('validation',validation),('scored',scored))}))
    pred={k:v for k,v in data.items() if k not in ('X','feature_names')};score=np.where(data['y']==1,.8,.2)
    pred.update(row_index=np.arange(len(cut)),crossfit_score=score,crossfit_cut=cut,crossfit_pass=score>=cut,scored_fold=fold)
    np.savez_compressed(directory/'predictions.npz',**pred)
    receipt=dict(completed=True,input_sha256='fixture-hash',fit_mode='sm-only',models=models,
                 predictions_sha256=digest(directory/'predictions.npz'),unscored_rows=0)
    (directory/'completion.json').write_text(json.dumps(receipt))
    return receipt


class CrossFitReportTest(unittest.TestCase):
    def test_zero_survivors_and_identical_models_keep_finite_sample_uncertainty(self):
        data=dataset();mask=data['y']==0;outcomes=[np.zeros(len(mask),dtype=bool)]*2
        row=paired_metric(data,mask,outcomes,40)
        self.assertGreater(row['baseline']['ci95'][1],0)
        lo,hi=row['paired_difference']['ci95'];self.assertLess(lo,0);self.assertGreater(hi,0)

    def test_reweighted_or_multiple_pairs_do_not_create_independent_event_owners(self):
        data=dataset();data['event_owner_id'][1]=data['event_owner_id'][0]
        mask=np.arange(200)<2;row=paired_metric(data,mask,[mask,mask],40)
        self.assertEqual(row['raw_pairs'],2);self.assertEqual(row['independent_event_owners'],1)
        self.assertEqual(row['native_pair_sumw2'],2);self.assertEqual(row['native_event_owner_sumw2'],4)
        self.assertEqual(row['native_effective_event_owners'],1)

    def test_empty_cells_are_unknown_and_tied_roc_scores_have_auc_half(self):
        data=dataset();mask=np.zeros(200,dtype=bool)
        self.assertIsNone(paired_metric(data,mask,[mask,mask],40)['candidate'])
        self.assertEqual(empirical_roc(np.array([0,1]),np.array([.5,.5]))['auc'],.5)
        self.assertIsNone(empirical_roc(np.array([1,1]),np.array([.2,.8]))['auc'])

    def test_changed_source_membership_is_rejected_even_with_valid_prediction_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            data=dataset();p=Path(temporary)/'model';receipt=result(p,data)
            verified_predictions(data,'fixture-hash',p)
            receipt['models'][0]['groups']['train'].append(receipt['models'][0]['groups']['scored'][0])
            (p/'completion.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'source ownership'):
                verified_predictions(data,'fixture-hash',p)

    def test_cut_must_reproduce_own_inner_validation_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            data=dataset();p=Path(temporary)/'model';receipt=result(p,data)
            prediction=dict(np.load(p/'predictions.npz',allow_pickle=False));prediction['crossfit_cut'][:]=.5
            prediction['crossfit_pass']=prediction['crossfit_score']>=.5
            np.savez_compressed(p/'predictions.npz',**prediction)
            for model in receipt['models']:model['threshold']=.5
            receipt['predictions_sha256']=digest(p/'predictions.npz');(p/'completion.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'inner-validation'):
                verified_predictions(data,'fixture-hash',p)

    def test_current_bsm_excludes_historical_controls_and_missing_grid_points_remain_unknown(self):
        data=dataset();data['process']=data['process'].astype('U16');data['process'][:3]='signal'
        data['domain']=np.full(200,'historical',dtype='U20');data['domain'][:3]=['current_BSM','signal_control','current_BSM']
        data['point']=data['point'].astype('U20');data['point'][:3]=['shared-point','shared-point','unknown-only']
        data['y'][:3]=[1,1,-1];data['feature_names']=data['feature_names'].astype('U30')
        data['feature_names'][:3]=['pair_vx','pair_vy','pair_vz']
        passed=np.zeros(200,dtype=bool);passed[0]=True
        prediction=dict(crossfit_pass=passed,crossfit_score=passed.astype(float))
        report=compare(data,[prediction,prediction],20,['shared-point','unknown-only','absent-point'])
        current=report['signal_points']['shared-point']['genuine_retention']
        self.assertEqual(current['raw_pairs'],1);self.assertEqual(current['baseline']['raw_pair_fraction'],1)
        pooled=report['pooled_signal_points_diagnostic']['shared-point']['genuine_retention']
        self.assertEqual(pooled['raw_pairs'],2);self.assertEqual(pooled['baseline']['raw_pair_fraction'],.5)
        self.assertIsNone(report['signal_points']['absent-point']['genuine_retention']['baseline'])
        self.assertTrue(report['signal_points']['absent-point']['coverage']['absent_from_candidate_rows'])
        self.assertIsNone(report['signal_points']['unknown-only']['genuine_retention']['baseline'])
        self.assertEqual(report['signal_points']['unknown-only']['unknown_retention']['raw_pairs'],1)
        self.assertEqual(sum(c['metrics']['genuine_retention']['raw_pairs'] for c in report['signal_mass_flight']['cells']),1)
        self.assertEqual(sum(c['metrics']['genuine_retention']['raw_pairs'] for c in report['pooled_signal_mass_flight_diagnostic']['cells']),2)

    def test_single_task_report_uses_task_truth_and_separates_excluded_duplicates(self):
        data=dataset();data['y']=np.tile([1,0,0,-1],50)
        data['label_reason']=np.tile(['common_vertex','different_vertices','same_muon_twice','unknown_match'],50)
        prediction=dict(crossfit_pass=np.ones(200,dtype=bool),crossfit_score=np.full(200,.8))
        distinct=compare(task_training_data(data,'distinct-muons'),[prediction],20,task='distinct-muons')
        self.assertEqual(distinct['views']['pooled_SM']['distinct_muon_retention']['raw_pairs'],100)
        self.assertEqual(distinct['views']['pooled_SM']['duplicate_muon_rejection']['raw_pairs'],50)
        self.assertIsNone(distinct['views']['pooled_SM']['duplicate_muon_rejection']['paired_difference'])
        origin=compare(task_training_data(data,'vertex-origin'),[prediction],20,task='vertex-origin')
        self.assertEqual(origin['views']['pooled_SM']['common_origin_retention']['raw_pairs'],50)
        self.assertEqual(origin['views']['pooled_SM']['different_origin_rejection']['raw_pairs'],50)
        self.assertEqual(origin['views']['pooled_SM']['unknown_origin_retention']['raw_pairs'],50)
        self.assertEqual(origin['views']['pooled_SM']['excluded_duplicate_pairs'],50)


if __name__=='__main__':unittest.main()
