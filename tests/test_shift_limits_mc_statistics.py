# Code style: AI-code. AI-generated finite-MC likelihood regression checks.
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
from shift_limits import digest
from shift_limits_current import Model,native,scan
from shift_limits_profile import LOG_KAPPA,profile_q,profile_q_mc


class FiniteMCTest(unittest.TestCase):
    def test_known_templates_recover_existing_likelihood_exactly(self):
        signal=np.array([2.,7.]);background=np.array([15.5,400.])
        self.assertEqual(profile_q_mc(signal,signal*0,[background],[background*0]),profile_q(1.,signal,background))

    def test_high_mc_statistics_approach_known_template_result(self):
        signal=np.array([2.,7.]);background=np.array([15.5,400.])
        result=profile_q_mc(signal,signal**2/1.e9,[background],[background**2/1.e9])
        self.assertAlmostEqual(result,profile_q(1.,signal,background),places=6)

    def test_one_effective_signal_event_agrees_with_direct_all_nuisance_fit(self):
        signal=8.;background=15.5;ns=1.;nb=10.
        def explicit(x):
            ss=signal*math.exp(LOG_KAPPA*x[0]+x[2])
            bb=background*math.exp(LOG_KAPPA*x[1]+x[3])
            prediction=ss+bb
            data=prediction-background-background*math.log(prediction/background)
            auxiliary=ns*(math.expm1(x[2])-x[2])+nb*(math.expm1(x[3])-x[3])
            return 2*(data+auxiliary)+x[0]**2+x[1]**2
        fit=minimize(explicit,np.zeros(4),method='BFGS',options={'gtol':1.e-8})
        actual=profile_q_mc([signal],[signal**2/ns],[[background]],[[background**2/nb]])
        self.assertAlmostEqual(actual,fit.fun,places=7)
        self.assertLess(actual,profile_q(1.,[signal],[background]))

    def test_sparse_signal_profile_also_handles_a_known_background(self):
        signal=8.;background=15.5
        def explicit(x):
            prediction=signal*math.exp(LOG_KAPPA*x[0]+x[2])+background*math.exp(LOG_KAPPA*x[1])
            return (2*(prediction-background-background*math.log(prediction/background)
                       +math.expm1(x[2])-x[2])+x[0]**2+x[1]**2)
        fit=minimize(explicit,np.zeros(3),method='BFGS',options={'gtol':1.e-8})
        actual=profile_q_mc([signal],[signal**2],[[background]],[[0.]])
        self.assertAlmostEqual(actual,fit.fun,places=7)

    def test_large_control_yields_do_not_break_sparse_mc_profile(self):
        # A rounded positive-weight fixture spanning very different template
        # normalizations. Unit theta steps used to hit an unresolved gamma pole
        # during the global fit, despite the optimum being well behaved.
        signal=np.array([19000.,4600.,2000.])
        variance=np.array([2.25e8,5.42e6,2.18e6])
        background=np.array([[9.11e6,0.,27.],[.18,1.3e7,0.],[6.19e9,0.,0.]])
        background_variance=np.array([[8.29e9,0.,.0714],[1.76e-5,2.38e9,0.],[1.01e14,0.,0.]])
        actual=profile_q_mc(signal,variance,background,background_variance)
        self.assertAlmostEqual(actual,18.365069159,places=6)
        self.assertLess(actual,profile_q(1.,signal,background.sum(axis=0)))

    def test_more_independent_sm_mc_cannot_weaken_the_profile(self):
        signal=np.array([8.,1.5]);variance=signal**2
        background=np.array([[15.5,5.]]);background_variance=background**2/10.
        values=[profile_q_mc(signal,variance,background,background_variance/factor) for factor in (1.,10.,100.)]
        self.assertTrue(all(b>=a-1.e-10 for a,b in zip(values,values[1:])))
        self.assertLessEqual(values[-1],profile_q_mc(signal,variance,background,background_variance*0)+1.e-10)

    def test_literally_known_zero_background_keeps_signal_mc_uncertainty(self):
        with self.assertRaisesRegex(ValueError,'unknown'):
            profile_q_mc([3.],[9.],[[0.]],[[0.]])
        actual=profile_q_mc([3.],[9.],[[0.]],[[0.]],allow_zero_background=True)
        def explicit(theta):
            value=3*math.exp(LOG_KAPPA*theta)
            return 2*math.log1p(value)+theta**2
        from scipy.optimize import minimize_scalar
        expected=minimize_scalar(explicit,bounds=(-20,0),method='bounded').fun
        self.assertAlmostEqual(actual,expected,places=8)
        self.assertLess(actual,profile_q(1.,[3.],[0.]))

    def test_invalid_positive_weight_moments_are_rejected(self):
        for variance in [[-1.],[float('nan')]]:
            with self.assertRaises(ValueError):profile_q_mc([1.],variance,[[2.]],[[1.]])
        with self.assertRaises(ValueError):profile_q_mc([0.],[1.],[[2.]],[[1.]])

    def make_model(self):
        model=Model.__new__(Model)
        model.config=SimpleNamespace(luminosity=1.,categories=('Both-Both',))
        model.mc_statistics=True
        model.arrays={'a':np.array([[2.,4.]]),'a_sumw2':np.array([[2.,8.]]),
                      'b':np.array([[8.,2.]]),'b_sumw2':np.array([[16.,4.]])}
        model.signals={1.:[dict(array_key='a',epsilon=.001,cross_section_pb=1.,full_gen_sumw=1.,prompt=False),
                          dict(array_key='b',epsilon=.01,cross_section_pb=100.,full_gen_sumw=1.,prompt=True)]}
        model.background_components=np.array([[[3.,0.]],[[0.,0.]]])
        model.background_variances=np.array([[[9.,0.]],[[0.,0.]]])
        return model

    def test_interpolated_variance_uses_squared_coefficients(self):
        model=self.make_model();epsilon=math.sqrt(.001*.01)
        value,variance=model.signal_moments(1.,epsilon)
        np.testing.assert_allclose(value,[[50.,30.]])
        np.testing.assert_allclose(variance,[[450.,300.]])
        # Physical epsilon scaling changes yields and errors together, leaving
        # the endpoint effective MC count unchanged.
        a,av=model.signal_moments(1.,.01);b,bv=model.signal_moments(1.,.02)
        np.testing.assert_allclose(a*a/av,b*b/bv)

    def test_empty_unrelated_process_does_not_remove_positive_total_background(self):
        model=self.make_model()
        signal,variance,backgrounds,bvars,report=model.mc_cells(1.,.001,'Both-Both','background')
        np.testing.assert_allclose(signal,[2.])
        np.testing.assert_allclose(backgrounds,[[3.],[0.]])
        self.assertEqual(report['omitted_total_zero_background_mc_bins'],1)
        self.assertEqual(report['omitted_signal_yield'],4.)
        self.assertEqual(report['total_signal_yield'],6.)

    def test_legacy_native_cards_cannot_silently_drop_mc_constraints(self):
        with self.assertRaisesRegex(ValueError,'legacy native cards omit finite-MC'):
            native(SimpleNamespace(include_mc_stat_uncertainties=True))

    def test_checkpoint_cannot_mix_nominal_and_finite_mc_records(self):
        for saved_mode,current_mode in [(None,True),(True,False),(False,True)]:
            with self.subTest(saved=saved_mode,current=current_mode),tempfile.TemporaryDirectory() as directory:
                output=Path(directory);(output/'snapshot.json').write_text('{}')
                saved={'snapshot_sha256':digest(output/'snapshot.json'),'records':[]}
                if saved_mode is not None:saved['include_mc_stat_uncertainties']=saved_mode
                (output/'scan.checkpoint.json').write_text(json.dumps(saved))
                model=SimpleNamespace(output=output,mc_statistics=current_mode,signals={})
                with patch('shift_limits_current.Model',return_value=model):
                    with self.assertRaisesRegex(ValueError,'checkpoint uses a different finite-MC mode'):scan(None)
                self.assertFalse((output/'scan.json').exists())

    def test_legacy_nominal_checkpoint_resumes_with_explicit_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);(output/'snapshot.json').write_text('{}')
            (output/'scan.checkpoint.json').write_text(json.dumps({'snapshot_sha256':digest(output/'snapshot.json'),'records':[]}))
            with patch('shift_limits_current.Model',return_value=SimpleNamespace(output=output,mc_statistics=False,signals={})):
                scan(None)
            self.assertIs(json.loads((output/'scan.json').read_text())['include_mc_stat_uncertainties'],False)

    def test_nominal_plotter_rejects_mc_scan_before_native_override(self):
        from shift_limits_current_plots import boundaries
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'scan.json').write_text(json.dumps({'include_mc_stat_uncertainties':True,'records':[]}))
            (output/'native_results.json').write_text('Must not read the legacy native overrides')
            with self.assertRaisesRegex(ValueError,'dedicated mc_statistics comparison'):
                boundaries(SimpleNamespace(output=output,mc_statistics=False),.95,'Both-Both')

    def test_legacy_native_cards_reject_mc_scan_even_with_nominal_config(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'scan.json').write_text(json.dumps({'include_mc_stat_uncertainties':True,'records':[]}))
            with self.assertRaisesRegex(ValueError,'cannot consume a finite-MC scan'):
                native(SimpleNamespace(output_path=directory,include_mc_stat_uncertainties=False))


if __name__=='__main__':unittest.main()
