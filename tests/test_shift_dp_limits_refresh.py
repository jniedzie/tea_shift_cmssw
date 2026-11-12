# Code style: AI-code. AI-generated guards for qualified partial-input refreshes.
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ANALYSIS=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ANALYSIS/'utils'),str(ANALYSIS/'bin')]
from shift_dp_limits_refresh import (mass_fingerprint,qualify_rate,qualify_row,select_signals,
                                    same_scan_support,passing_event_statistics,settings,check_pdfs)
from shift_limits_current_plots import comparison_figure_directory
from shift_dp_progress_plots import with_retained_statistics


class RefreshInputsTest(unittest.TestCase):
    def test_retained_statistics_preserve_old_totals_without_pooling_models(self):
        old=dict(name='m15_prompt',event_neff=1596.,event_histogram_readback='passed')
        retained=dict(rows=[old,dict(old,name='m5_prompt',event_neff=500.)])
        updated=dict(rows=[dict(old,name='m5_prompt',event_neff=25.)])
        total=with_retained_statistics(updated,retained)
        by_name={r['name']:r for r in total['rows']}
        self.assertEqual(by_name['m15_prompt']['event_neff'],1596.)
        self.assertTrue(by_name['m15_prompt']['retained_reference'])
        self.assertEqual(by_name['m5_prompt']['event_neff'],25.)
        self.assertFalse(by_name['m5_prompt']['retained_reference'])
        self.assertNotIn('retained_reference',old)

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
            if change=='unsupported_epsilon':bad['ready_signals'][0]['epsilon']=.2
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

    def dy_rate(self, mass=40., epsilon=.03, version='msht20-nnlo-v2'):
        hashes={'msht20-nnlo':'18fac0c030160c00d8fd50a3ce145a5e435546a5b3426d6caa4942af1c2384e5',
                'msht20-nnlo-v2':'b27dd724696f8f4d72b04d94fe9c87caf61620a646ade8c7ef27ff122036f691'}
        row=dict(name='physical_anchor',mass_gev=mass,epsilon=epsilon,proper_ctau_mm=.01)
        rate=dict(row,mechanism='pure-on-shell-drell-yan',dy_prescription=dict(
            name=version,epsilon_upper_bound=.1 if version.endswith('v2') else .001,
            authority=dict(sha256=hashes[version])))
        return row,rate

    def photon_metadata(self):
        return dict(experimental_photon_benchmark=True,photon_benchmark_approximation=dict(
            schema='shift-experimental-photon-benchmark-epsilon-v1',maximum_epsilon=.03,
            source_scaling='epsilon squared',
            physical_width='Unchanged pinned DarkCast photon-like spectral widths',
            neutral_current_width_included=False))

    def test_v2_native_domain_does_not_widen_v1_or_unqualified_tables(self):
        row,rate=self.dy_rate(epsilon=.1)
        self.assertEqual(qualify_rate(row,rate)['maximum_epsilon'],.1)
        oldrow,old=self.dy_rate(epsilon=.002,version='msht20-nnlo')
        with self.assertRaisesRegex(ValueError,'epsilon domain'):qualify_rate(oldrow,old)
        for change in ['hash','mass','epsilon']:
            bad=copy.deepcopy(rate);point=copy.deepcopy(row)
            if change=='hash':bad['dy_prescription']['authority']['sha256']='f'*64
            if change=='mass':bad['mass_gev']=point['mass_gev']=20.
            if change=='epsilon':bad['epsilon']=point['epsilon']=.10001
            with self.subTest(change=change),self.assertRaises(ValueError):qualify_rate(point,bad)

    def test_large_photon_epsilon_requires_the_exact_experimental_approximation(self):
        row,rate=self.dy_rate(mass=5.,epsilon=.03)
        with self.assertRaisesRegex(ValueError,'epsilon domain'):qualify_rate(row,rate)
        rate.update(self.photon_metadata())
        domain=qualify_rate(row,rate)
        self.assertEqual(domain['maximum_epsilon'],.03)
        self.assertTrue(domain['experimental_photon_benchmark'])
        for key,value in [('maximum_epsilon',.1),('source_scaling','none'),
                          ('neutral_current_width_included',True)]:
            bad=copy.deepcopy(rate);bad['photon_benchmark_approximation'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'approximation'):qualify_rate(row,bad)
        with self.assertRaisesRegex(ValueError,'explicit experimental flag'):
            qualify_rate(row,dict(rate,experimental_photon_benchmark=False))
        low=dict(row,mass_gev=.7)
        senscalc=dict(low,mechanism='senscalc-meson-fragmentation-bremsstrahlung',**self.photon_metadata())
        self.assertEqual(qualify_rate(low,senscalc)['maximum_epsilon'],.03)

    def test_overlap_domain_is_the_intersection_of_actual_component_authorities(self):
        row,dy=self.dy_rate(mass=1.5,epsilon=.03)
        dy.update(self.photon_metadata())
        meson=dict(row,mechanism='senscalc-meson-fragmentation-bremsstrahlung',**self.photon_metadata())
        rate=dict(schema='shift-dark-photon-senscalc-dy-component-rates-v1',physical_point=row,
            components=[dict(mechanism=meson['mechanism'],rate=meson),dict(mechanism=dy['mechanism'],rate=dy)])
        self.assertEqual(qualify_rate(row,rate)['maximum_epsilon'],.03)
        oldrow,old=self.dy_rate(mass=1.5,epsilon=.03,version='msht20-nnlo')
        rate['components'][1]['rate']=old
        with self.assertRaisesRegex(ValueError,'epsilon domain'):qualify_rate(row,rate)

    def test_refresh_discovers_exact_anchor_names_and_domains_from_inventory(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'inventory.json'
            rows=[dict(name='m40_epsilon_0p03',epsilon_domain=dict(maximum_epsilon=.1)),
                  dict(name='m0p212_lab_1000m',epsilon_domain=dict(maximum_epsilon=.01))]
            path.write_text(json.dumps(dict(signals=rows)))
            config=settings(Path(temporary),Path('/tmp/background.json'),path,Path('/tmp/figures'))
            self.assertIsNone(config.response_anchors)
            self.assertEqual(config.response_anchor_names,tuple(r['name'] for r in rows))
            self.assertEqual(config.epsilon_max,.1)

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
        anchors[-1]['epsilon_domain']=dict(maximum_epsilon=.001)
        self.assertTrue(same_scan_support(dict(epsilon_support=[1.e-6,.001]),anchors,.1))
        self.assertFalse(same_scan_support(dict(epsilon_support=[1.e-6,.1]),anchors,.1))

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

    def test_final_figures_are_checked_by_name_including_zoom_and_strength_map(self):
        names=['limits_current_cl90_comparison.pdf','limits_current_cl90_background_comparison.pdf',
               'limits_current_cl90_zero_background_comparison.pdf','background_mass_templates.pdf',
               'signal_coverage.pdf','limits_before_after.pdf','limits_before_after_zoom.pdf',
               'signal_strength_map.pdf','signal_statistics_before_after.pdf']
        class Document:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __len__(self):return 1
            def __getitem__(self,index):
                return SimpleNamespace(rect=SimpleNamespace(width=600),get_drawings=lambda:[1])
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            for name in names:(output/name).write_bytes(b'figure bytes')
            with patch.dict(sys.modules,fitz=SimpleNamespace(open=lambda path:Document())):
                self.assertEqual(len(check_pdfs(output)),9)
                (output/'extra_diagnostic.pdf').write_bytes(b'extra figure')
                self.assertEqual(len(check_pdfs(output)),10)
                (output/'signal_strength_map.pdf').unlink()
                with self.assertRaisesRegex(ValueError,'signal_strength_map.pdf'):check_pdfs(output)


if __name__=='__main__':unittest.main()
