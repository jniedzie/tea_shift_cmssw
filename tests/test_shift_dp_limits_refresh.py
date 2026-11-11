# Code style: AI-code. AI-generated guards for qualified partial-input refreshes.
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ANALYSIS=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ANALYSIS/'utils'),str(ANALYSIS/'bin')]
from shift_dp_limits_refresh import (mass_fingerprint,qualify_rate,qualify_row,select_signals,
                                    same_scan_support,passing_event_statistics)
from shift_limits_current_plots import comparison_figure_directory


class RefreshInputsTest(unittest.TestCase):
    def fixture(self):
        previous=dict(name='m0p7_prompt',mass_gev=.7,sha256='a'*64,rate_authority_sha256='b'*64)
        row=dict(name='m0p7_prompt',mass_gev=.7,epsilon=.001,proper_ctau_mm=.01,
                 histogram_sha256='a'*64,rate_sha256='b'*64,histogram_bytes=1024,
                 qualification=dict(sha256='c'*64),full_benchmark_components=True,
                 cumulative_compatible_inputs_included=True,retained_reference=True)
        catalog=dict(schema='shift-dp-qualified-signal-catalog-v1',complete=True,
                     partial_production=True,ready_signals=[row])
        return dict(signals=[previous]),catalog

    def test_partial_production_keeps_qualified_ready_inputs(self):
        baseline,catalog=self.fixture()
        self.assertEqual(select_signals(catalog,baseline),catalog['ready_signals'])
        for change in ['duplicate','missing','small_increment','unsupported_epsilon']:
            bad=copy.deepcopy(catalog)
            if change=='duplicate':bad['ready_signals']*=2
            if change=='missing':bad['ready_signals']=[]
            if change=='small_increment':bad['ready_signals'][0]['cumulative_compatible_inputs_included']=False
            if change=='unsupported_epsilon':bad['ready_signals'][0]['epsilon']=.002
            with self.subTest(change=change),self.assertRaises(ValueError):select_signals(bad,baseline)

    def test_reference_pin_and_cumulative_qualification_are_required(self):
        baseline,catalog=self.fixture();row=catalog['ready_signals'][0]
        proof=dict(schema='shift-dp-retained-SensCalc-qualification-v1',complete=True,
                   reference_snapshot=dict(sha256='d'*64),
                   points={row['name']:dict(histogram_sha256='a'*64,rate_sha256='b'*64)})
        qualify_row(row,proof,'d'*64)
        with self.assertRaisesRegex(ValueError,'another reference'):qualify_row(row,proof,'e'*64)
        new=dict(row,retained_reference=False,input_signature='f'*64)
        newproof=dict(schema='shift-dp-cumulative-publication-qualification-v1',complete=True,
                      point=new['name'],input_signature='f'*64,
                      all_ready_compatible_inputs_included=True,mechanism_denominators_retained=True)
        qualify_row(new,newproof,'d'*64)
        with self.assertRaisesRegex(ValueError,'complete cumulative'):
            qualify_row(new,dict(newproof,mechanism_denominators_retained=False),'d'*64)

    def test_old_DY_cannot_enter_a_new_model_catalog(self):
        row=dict(name='m2_prompt',mass_gev=2.,epsilon=.001,proper_ctau_mm=.01)
        rate=dict(row,mechanism='pure-on-shell-drell-yan')
        with self.assertRaisesRegex(ValueError,'Old NNPDF'):qualify_rate(row,rate)
        rate['dy_prescription']=dict(name='msht20-nnlo',authority=dict(
            sha256='18fac0c030160c00d8fd50a3ce145a5e435546a5b3426d6caa4942af1c2384e5'))
        qualify_rate(row,rate)
        with self.assertRaisesRegex(ValueError,'physical point'):
            qualify_rate(dict(row,epsilon=.0005),rate)

    def test_overlap_requires_real_component_sum(self):
        row=dict(name='m1p5_prompt',mass_gev=1.5,epsilon=.001,proper_ctau_mm=.01)
        with self.assertRaisesRegex(ValueError,'actual SensCalc plus DY'):
            qualify_rate(row,dict(row,mechanism='pure-on-shell-drell-yan'))

    def test_changed_weights_normalization_or_lifetime_invalidates_fit_reuse(self):
        row=dict(epsilon=.001,proper_ctau_mm=.01,cross_section_pb=2.,full_gen_sumw=100.,
                 prompt=True,mechanism='senscalc',array_key='point')
        model=SimpleNamespace(signals={.7:[row]},arrays={'point':np.array([[1.,2.]]),
                                                       'point_sumw2':np.array([[1.,4.]])})
        original=mass_fingerprint(model,.7)
        for field,value in [('full_gen_sumw',101.),('proper_ctau_mm',.02),('cross_section_pb',3.)]:
            changed=copy.deepcopy(model);changed.signals[.7][0][field]=value
            self.assertNotEqual(original,mass_fingerprint(changed,.7))
        changed=copy.deepcopy(model);changed.arrays['point_sumw2'][0,1]=5.
        self.assertNotEqual(original,mass_fingerprint(changed,.7))

    def test_frozen_figure_location_avoids_live_configuration(self):
        expected=Path('/tmp/frozen-figure').resolve()
        self.assertEqual(comparison_figure_directory(SimpleNamespace(
            comparison_figure_directory=str(expected))),expected)

    def test_legacy_high_epsilon_roots_cannot_be_reused_under_the_new_cap(self):
        anchors=[dict(epsilon=1.e-6,prompt=False),dict(epsilon=.001,prompt=True)]
        self.assertTrue(same_scan_support(dict(epsilon_support=[1.e-6,.001]),anchors,.001))
        self.assertFalse(same_scan_support(dict(epsilon_support=[1.e-6,1.]),anchors,.001))
        self.assertFalse(same_scan_support(dict(epsilon_support=[2.e-6,.001]),anchors,.001))

    def test_event_statistics_exclude_zero_and_count_multiplicity_once_with_overflow(self):
        # Positive bins hold weighted event entries, not multiplicity times weight.
        histogram=SimpleNamespace(GetNbinsX=lambda:3,
            GetXaxis=lambda:SimpleNamespace(FindBin=lambda value:1),
            GetBinContent=lambda index:[100.,1000.,4.,2.,1.][index],
            GetBinError=lambda index:[10.,100.,4.,2.,1.][index])
        source=SimpleNamespace(IsZombie=lambda:False,Get=lambda name:histogram,Close=lambda:None)
        with patch.dict(sys.modules,ROOT=SimpleNamespace(TFile=SimpleNamespace(Open=lambda path:source))):
            result=passing_event_statistics('/tmp/test.root')
        self.assertEqual(result['event_passing_sumw'],7.)
        self.assertEqual(result['event_passing_sumw2'],21.)
        self.assertAlmostEqual(result['event_neff'],49./21.)


if __name__=='__main__':unittest.main()
