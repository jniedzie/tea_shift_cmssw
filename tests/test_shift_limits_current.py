# Code style: AI-code. AI-generated regression checks for published signal limits.
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
from shift_limits import digest
from shift_limits_current import Model,signal_normalization


class PublishedLimitsTest(unittest.TestCase):
    def test_full_gen_normalization_does_not_use_sampled_cutflow_or_repeat_branching(self):
        # A sampled cutflow may be 3225.4 while the original exposure is 3196.
        rate={'physical_selected_cross_sections_pb':[{'value_pb':2.},{'value_pb':1.}],
              'full_gen':{'sum_weights':3196.},'physical_br_mumu':.42}
        sigma,exposure=signal_normalization(rate,{'cross_section_pb':3.,'initial_weight_sum':3196.})
        self.assertEqual((sigma,exposure),(3.,3196.))
        self.assertAlmostEqual(300*sigma*10/exposure,9000/3196)
        with self.assertRaises(ValueError):
            signal_normalization(rate,{'cross_section_pb':3.,'initial_weight_sum':3225.4})

    def make_model(self,directory,mechanisms,zero_first=False):
        output=Path(directory);arrays={};rows=[]
        for i,mechanism in enumerate(mechanisms):
            key='signals_'+str(i);arrays[key]=np.array([[0. if zero_first and i==0 else 1.,0.]])
            rows.append(dict(name='m0p25_lab_'+('0p01m' if i==0 else '5m'),array_key=key,
                mass_gev=.25,epsilon=1.e-4/(i+1),proper_ctau_mm=.01*(i+1)**2,
                cross_section_pb=2.,full_gen_sumw=100.,prompt=False,mechanism=mechanism))
        np.savez_compressed(output/'histograms.npz',**arrays)
        (output/'snapshot.json').write_text(json.dumps(dict(histogram_cache=str(output/'histograms.npz'),
            histogram_cache_sha256=digest(output/'histograms.npz'),signals=rows,backgrounds=[])))
        return Model(SimpleNamespace(output_path=str(output),luminosity=300.,
            response_anchors=('prompt','lab_0p01m','lab_5m'),categories=('Both-Both',)))

    def test_centimetre_anchor_is_used_without_inventing_lifetime_support(self):
        with tempfile.TemporaryDirectory() as directory:
            model=self.make_model(directory,['senscalc'])
            self.assertEqual(model.unused_signals,[])
            self.assertAlmostEqual(model.signal(.25,1.e-4).sum(),6.)
            self.assertIsNone(model.signal(.25,5.e-5))
            self.assertIsNone(model.signal(.25,2.e-4))

    def test_old_eta_and_new_senscalc_lifetimes_cannot_be_interpolated_together(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'Different signal production prescriptions'):
                self.make_model(directory,['senscalc','eta-etaprime'])

    def test_zero_observed_mc_is_not_a_zero_acceptance_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            model=self.make_model(directory,['senscalc','senscalc'],zero_first=True)
            self.assertEqual(len(model.signals[.25]),1)
            self.assertIn('response remains unknown',model.unused_signals[0]['unused_reason'])
            self.assertIsNone(model.signal(.25,1.e-4))

    def test_plot_does_not_fill_a_nonexcluded_gap_between_islands(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from shift_limits_current_plots import draw_regions
        fig,axis=plt.subplots()
        row=dict(mass_gev=.7,mode='background',excluded_intervals=[[.0003,.0008],[.0017,1.]],
                 roots=[.0003,.0008,.0017],epsilon_support=[2.e-7,1.],upper_prompt_continuation=True)
        draw_regions(axis,[row],'background','red','test')
        for line in axis.lines:
            values=line.get_ydata()
            self.assertFalse(min(values)<.001<max(values))
        plt.close(fig)

    def test_radiative_meson_endpoint_closes_without_an_above_threshold_tail(self):
        from shift_limits_pheno_comparison import radiative_phase_space
        parent=.547862
        values=radiative_phase_space([0.,parent/2,parent,2*parent],parent)
        np.testing.assert_allclose(values,[1.,(.75)**3,0.,0.],rtol=1.e-14)
        with self.assertRaises(ValueError):radiative_phase_space([-.1],parent)


if __name__=='__main__':unittest.main()
