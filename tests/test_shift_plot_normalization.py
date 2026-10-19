# Code style: AI-code. AI-generated regression test.
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "configs"))
from shift_sample_paths import full_gen_weight_sums


class FullGenPlotNormalizationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.campaign = "shift_detector_representative_20261007_v10"
        ordinary = self.write("ordinary.json", {"schema": "frozen-ordinary-source"})
        weighted = self.write("weighted.json", {"schema": "frozen-weighted-source"})
        self.accounting = {
            "schema": "shift-weighted-high-bin-final-accounting-v1", "complete": True,
            "manifest_sha256": self.digest(weighted),
            "strata": {"qcd_20toinf": {"accepted": 1000, "sumw": 0.5}},
        }
        accounting = self.write("accounting.json", self.accounting)
        parent = self.write("parent.json", {
            "schema": "shift-gen-to-nano-plan-v1",
            "strata": {"qcd_0to1": 100, "qcd_20toinf": 1000},
            "ordinary_manifest": str(ordinary), "ordinary_manifest_sha256": self.digest(ordinary),
            "weighted_manifest": str(weighted), "weighted_manifest_sha256": self.digest(weighted),
        })
        self.plan = {
            "schema": "shift-detector-pps-sample-v1", "parent_manifest": str(parent),
            "parent_manifest_sha256": self.digest(parent),
            "normalization_authorities": {
                "ordinary": {"manifest": str(ordinary), "manifest_sha256": self.digest(ordinary)},
                "weighted": {"manifest": str(weighted), "manifest_sha256": self.digest(weighted),
                             "accounting": str(accounting), "accounting_sha256": self.digest(accounting),
                             "strata": self.accounting["strata"]},
            },
            "strata": {
                "qcd_0to1": {"original_events": 100, "original_sumw": 100,
                             "realized_events": 9, "corrected_sumw": 90,
                             "original_normalization_records": [
                                 {"stratum": "qcd_0to1", "events": 100, "normalization_record": None}]},
                "qcd_20toinf": {"original_events": 1000, "original_sumw": 0.5,
                                "realized_events": 10, "corrected_sumw": 0.4,
                                "original_normalization_records": [
                                    {"stratum": "qcd_20toinf", "events": 1000,
                                     "normalization_record": "weighted-ledger.json"}]},
            },
        }
        self.manifest = {
            "schema": "shift-gen-to-nano-plan-v1", "eos_output": "/eos/" + self.campaign,
            "strata": {"qcd_0to1": 9, "qcd_20toinf": 10},
            "detector_sampling": {"plan": "sampling_plan.json",
                                  "parent_manifest_sha256": self.digest(parent)},
        }
        self.refresh()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def refresh(self):
        path = self.write("sampling_plan.json", self.plan)
        self.manifest["detector_sampling"]["plan_sha256"] = self.digest(path)
        self.write("manifest.json", self.manifest)

    def sums(self):
        return full_gen_weight_sums(self.campaign, self.root)

    def test_full_parent_exposure_ignores_selected_counts_and_sums(self):
        self.assertEqual(self.sums(), {"qcd_0to1": 100., "qcd_20toinf": 0.5})

    def test_legacy_campaign_needs_no_representative_provenance(self):
        self.assertIsNone(full_gen_weight_sums("legacy_10k", self.root / "absent"))

    def test_missing_representative_authority_fails_closed(self):
        (self.root / "accounting.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
            self.sums()

    def test_changed_sampling_payload_fails_closed(self):
        (self.root / "sampling_plan.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "hash differs"):
            self.sums()

    def test_wrong_campaign_fails_closed(self):
        self.manifest["eos_output"] = "/eos/another_campaign"
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "requested campaign"):
            self.sums()

    def test_weighted_selected_sum_cannot_replace_parent_sum(self):
        self.plan["strata"]["qcd_20toinf"]["original_sumw"] = 0.4
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "Weighted full GEN sum differs"):
            self.sums()

    def test_unit_weight_selected_sum_cannot_replace_parent_count(self):
        self.plan["strata"]["qcd_0to1"]["original_sumw"] = 90
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "Unit-weight full GEN exposure differs"):
            self.sums()

    def test_missing_stratum_cannot_fall_back_to_selected_cutflow(self):
        del self.plan["strata"]["qcd_20toinf"]
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "inventories differ"):
            self.sums()


class AvailableHistogramPlotConfigTest(unittest.TestCase):
    def setUp(self):
        import ROOT
        self.ROOT = ROOT
        self.repository = Path(__file__).resolve().parents[1]
        for name in ('tea/pylibs/plotting', 'tea/pylibs/logger'):
            directory = str(self.repository / name)
            sys.path.insert(0, directory)
            self.addCleanup(sys.path.remove, directory)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.campaign = 'shift_detector_representative_20261007_v10'
        self.bsm_campaign = '20261013-unfinished-production'
        self.sm = self.write_histograms(
            self.base / 'qcd_pThat0to1GeV' / self.campaign / 'histograms_merged', 20.)
        self.signal = self.write_histograms(
            self.base / 'darkphoton_m15_lab_100m' / self.bsm_campaign / 'histograms_merged', 5.)

    def write_histograms(self, directory, initial_sum):
        directory.mkdir(parents=True)
        path = directory / 'partial.root'
        output = self.ROOT.TFile(str(path), 'RECREATE')
        cutflow = self.ROOT.TH1D('cutFlow', '', 2, 0, 2)
        cutflow.SetBinContent(1, initial_sum)
        cutflow.SetBinContent(2, 4.)
        cutflow.Write()
        histogram = self.ROOT.TH1D('available_observable', '', 2, 0, 2)
        histogram.Fill(.5, 3.)
        histogram.Fill(1.5, 1.)
        histogram.Write()
        output.Close()
        return path

    def config(self, *, bsm_campaign=None, masses=(15,), flights=(100,), epsilons=None, histogram_rebins=None,
               streaming_inventory=None, campaign=None, cross_sections=None):
        from Histogram import Histogram

        def histogram(*args, **kwargs):
            definition = Histogram(*args, **kwargs)
            if histogram_rebins and definition.name in histogram_rebins:
                definition.rebin = histogram_rebins[definition.name]
            return definition

        with patch.multiple('shift_paths', base_path=str(self.base), campaign=(streaming_inventory or campaign or self.campaign),
                            cross_sections=(cross_sections if cross_sections is not None else
                                            {'qcd': {'0to1': 2., '1to2': 3.}}),
                            bsm_campaign=bsm_campaign, bsm_masses_to_plot=list(masses),
                            bsm_mean_lab_flights_to_plot=list(flights), bsm_epsilons_to_plot=epsilons,
                            histogram_cache_directory=None), \
                patch('Histogram.Histogram', side_effect=histogram), \
                patch('shift_sample_paths.full_gen_weight_sums', side_effect=AssertionError('Full GEN validation')), \
                patch('shift_sample_paths._read_normalization_record', side_effect=AssertionError('Production records')), \
                patch('shift_sample_paths._bsm_campaign_records', side_effect=AssertionError('BSM production validation')):
            return SimpleNamespace(**runpy.run_path(str(self.repository / 'configs/shift_plotter_config.py')))

    def streaming_inventory(self):
        rows = {}
        for kind, policy in (('path', 'native_gen_weight/reco_inclusion_probability'),
                             ('raw_path', 'one_per_reconstructed_entry')):
            path = self.write_histograms(self.base / kind, 4.)
            source = self.ROOT.TFile(str(path), 'UPDATE')
            self.ROOT.TObjString(json.dumps(dict(schema='shift-streaming-normalization-v1',
                stratum='qcd_0to1', closed_cohorts=True, full_source_trials=12345,
                sigma_nd_pb=7.e10, histogram_weight=policy,
                no_second_filter_efficiency=True))).Write('ShiftStreamingNormalization')
            source.Close()
            rows[kind] = str(path)
        inventory = self.base / 'streaming.json'
        inventory.write_text(json.dumps(dict(schema='shift-streaming-plot-inputs-v1',
            samples=[dict(stratum='qcd_0to1', **rows)])))
        return str(inventory)

    def test_streaming_rate_replaces_qcd_and_preserves_other_sm_and_signals(self):
        from HistogramNormalizer import HistogramNormalizer
        from Sample import SampleType

        dy = self.write_histograms(self.base / 'dy_mHat0p211317to0p5GeV' /
                                   self.campaign / 'histograms_merged', 30.)
        jpsi = self.write_histograms(self.base / 'jpsi_pThat0to1GeV' /
                                     self.campaign / 'histograms_merged', 40.)
        sections = {'dy': {'0.211317to0.5': 5.}, 'jpsi': {'0to1': 6.}, 'qcd': {'0to1': 999.}}
        inventory = self.streaming_inventory()
        record = json.loads(Path(inventory).read_text())
        record['reference_samples'] = []
        for stratum, path, exposure, cross_section in (
                ('dy_0.211317to0.5', dy, 30., 5.), ('jpsi_0to1', jpsi, 40., 6.)):
            source = self.ROOT.TFile(str(path), 'UPDATE')
            self.ROOT.TObjString(json.dumps(dict(schema='shift-preserved-full-GEN-histogram-v1',
                stratum=stratum, full_source_sumw=exposure, cross_section_pb=cross_section,
                no_second_filter_efficiency=True,
                histogram_weight='native_gen_weight/reco_inclusion_probability'))).Write('ShiftFullGenNormalization')
            source.Close()
            record['reference_samples'].append(dict(stratum=stratum, path=str(path)))
        Path(inventory).write_text(json.dumps(record))
        with patch('Logger.warn'):
            config = self.config(streaming_inventory=inventory,
                                 bsm_campaign=self.bsm_campaign, cross_sections=sections)
        self.assertEqual([sample.name for sample in config.samples],
                         ['dy_0.211317to0.5', 'jpsi_0to1', 'qcd_0to1', 'm15_lab_100m'])
        self.assertEqual([sample.file_path for sample in config.samples[:2]], [str(dy), str(jpsi)])
        qcd = next(sample for sample in config.samples if sample.name == 'qcd_0to1')
        self.assertNotEqual(qcd.file_path, str(self.sm))
        self.assertEqual(qcd.cross_section, 7.e10)
        self.assertEqual(qcd.initial_weight_sum, 12345)
        self.assertEqual(config.custom_stacks_order, [sample.name for sample in config.samples])
        self.assertGreater(len(config.histograms), 6)
        self.assertTrue(any(sample.type == SampleType.signal for sample in config.samples))
        self.assertEqual(next(h for h in config.histograms
                              if h.name == 'dimuon/ShiftDimuonVertexBoth-Both_mass').legend_label,
                         'Both-Both')
        normalizer = HistogramNormalizer(config)
        self.addCleanup(lambda: [source.Close() for source in normalizer.input_files.values()])
        self.assertEqual(normalizer.background_initial_sum_weights[('qcd_0to1', -1)], 12345)

    def test_named_campaign_selects_published_qcd_and_full_exposure(self):
        inventory = self.streaming_inventory()
        campaign = '20261016_mu_enriched_weekend_v1'
        with patch('shift_sample_paths.campaign_histogram_inventory', return_value=inventory) as resolver:
            config = self.config(campaign=campaign)
        resolver.assert_called_once_with(campaign)
        self.assertEqual([sample.name for sample in config.samples], ['qcd_0to1'])
        self.assertEqual(config.samples[0].initial_weight_sum, 12345)
        self.assertEqual(config.samples[0].cross_section, 7.e10)

    def test_production_raw_preview_uses_no_luminosity_or_rate_weights(self):
        from HistogramNormalizer import NormalizationType
        utils = str(self.repository / 'utils')
        sys.path.insert(0, utils)
        self.addCleanup(sys.path.remove, utils)
        import shift_streaming_histograms as production
        inventory = self.streaming_inventory()
        configs = []

        def run_plotter(command, **options):
            with patch.dict(sys.modules):
                sys.modules.pop('shift_plotter_config', None)
                configs.append(SimpleNamespace(**runpy.run_path(str(Path(options['cwd'])/command[-1]))))

        with patch.multiple('shift_paths', base_path=str(self.base), campaign=self.campaign,
                            cross_sections={}, bsm_campaign=None), \
                patch.object(production.subprocess, 'run', side_effect=run_plotter):
            production.render_plots(Path(inventory), Path('plotter.py'), self.base/'plots',
                                    self.base, ['raw'], caption='test counts')
        config = configs[0]
        self.assertEqual([sample.name for sample in config.samples], ['qcd_0to1'])
        self.assertEqual(len(config.histograms), 6)
        self.assertTrue(all(h.norm_type == NormalizationType.none for h in config.histograms))
        self.assertTrue(all(h.legend_label == 'test counts' for h in config.histograms))
        self.assertFalse(hasattr(config, 'lumi_label_value'))
        self.assertEqual(config.output_path, str(self.base/'plots/raw'))

    def test_streaming_reference_preserves_original_weighted_gen_denominator(self):
        from HistogramNormalizer import HistogramNormalizer

        inventory = self.streaming_inventory()
        reference = self.write_histograms(self.base / 'reference', 17.)
        source = self.ROOT.TFile(str(reference), 'UPDATE')
        self.ROOT.TObjString(json.dumps(dict(schema='shift-preserved-full-GEN-histogram-v1',
            stratum='jpsi_10to20', full_source_sumw=.0009177813540578304,
            cross_section_pb=23.906403461436227, no_second_filter_efficiency=True,
            histogram_weight='native_gen_weight/reco_inclusion_probability'))).Write('ShiftFullGenNormalization')
        source.Close()
        record = json.loads(Path(inventory).read_text())
        record['reference_samples'] = [dict(stratum='jpsi_10to20', path=str(reference))]
        Path(inventory).write_text(json.dumps(record))
        config = self.config(streaming_inventory=inventory,
                             cross_sections={'jpsi': {'10to20': 999.}, 'qcd': {'0to1': 999.}})
        jpsi = next(sample for sample in config.samples if sample.name == 'jpsi_10to20')
        self.assertEqual(jpsi.file_path, str(reference))
        self.assertEqual(jpsi.initial_weight_sum, .0009177813540578304)
        self.assertEqual(jpsi.cross_section, 23.906403461436227)
        normalizer = HistogramNormalizer(config)
        self.addCleanup(lambda: [source.Close() for source in normalizer.input_files.values()])
        self.assertEqual(normalizer.background_initial_sum_weights[('jpsi_10to20', -1)], .0009177813540578304)

    def test_partial_sm_plots_without_any_production_validation(self):
        with patch('Logger.warn') as warn:
            config = self.config()
        self.assertEqual([sample.name for sample in config.samples], ['qcd_0to1'])
        self.assertEqual(config.samples[0].file_path, str(self.sm))
        self.assertEqual(config.samples[0].initial_weight_sum, -1)
        self.assertTrue(warn.called)  # The missing bin does not block the ready one.

    def test_partial_bsm_needs_no_plan_marker_or_complete_campaign(self):
        with patch('Logger.warn'):
            config = self.config(bsm_campaign=self.bsm_campaign, flights=(None, 100))
        self.assertEqual([sample.name for sample in config.samples], ['qcd_0to1', 'm15_lab_100m'])
        self.assertEqual(config.samples[1].file_path, str(self.signal))
        self.assertTrue(all(sample.initial_weight_sum == -1 for sample in config.samples))

    def test_short_category_is_visible_once_in_the_highest_legend(self):
        from HistogramPlotter import HistogramPlotter

        labels = {
            'muon/ShiftMuon_pt_variable': 'All categories',
            'muon/ShiftMuonBothEndcaps_pt_variable': 'Both Endcaps',
            'dimuon/ShiftDimuonVertex_mass': 'All categories',
            'dimuon/ShiftDimuonVertexBoth-Both_mass': 'Both-Both',
            'event/Event_nShiftMuonBothEndcaps': 'Both Endcaps',
            'event/Event_nShiftDimuonVertexBoth-Both': 'Both-Both',
        }
        for campaign in (None, self.bsm_campaign):
            with self.subTest(campaign=campaign), patch('Logger.warn'):
                config = self.config(bsm_campaign=campaign)
                config.histograms = [histogram for histogram in config.histograms
                                     if histogram.name in labels]
                config.output_path = str(self.base / 'plots')
                plotter = HistogramPlotter(config)
                for file in plotter.normalizer.input_files.values():
                    self.addCleanup(file.Close)
                plotter.histosamples = [(histogram, sample)
                                       for histogram in config.histograms
                                       for sample in config.samples]
                plotter.setupLegends()
                self.assertEqual(len(plotter.legends), len(labels))
                for histogram in config.histograms:
                    category = labels[histogram.name]
                    legends = plotter.legends[histogram.name]
                    highest = max(legends.values(), key=lambda legend: legend.GetY2NDC())
                    for sample_type, legend in legends.items():
                        title = config.legends[sample_type].title
                        expected = title
                        if legend is highest:
                            expected = f'{title} ({category})' if title else category
                        self.assertEqual(legend.GetHeader(), expected)

    def test_legacy_epsilon_sample_without_flight_is_not_mislabeled_prompt(self):
        signal = self.write_histograms(
            self.base / 'darkphoton_m15_epsilon_1e-06' / self.bsm_campaign / 'histograms_merged', 5.)
        with patch('Logger.warn'):
            config = self.config(bsm_campaign=self.bsm_campaign, flights=(), epsilons=[1.e-6])
        self.assertEqual([sample.name for sample in config.samples], ['qcd_0to1', 'm15_epsilon_1e-06'])
        self.assertEqual(config.samples[1].file_path, str(signal))
        self.assertIn('mean lab flight unavailable', config.samples[1].legend_description)
        self.assertEqual(config.samples[1].initial_weight_sum, -1)

    def test_lab_legends_do_not_show_translated_ctau_and_prompt_stays_prompt(self):
        prompt = self.write_histograms(
            self.base / 'darkphoton_m15_prompt' / self.bsm_campaign / 'histograms_merged', 5.)
        for path, flight in ((prompt, None), (self.signal, 100.)):
            point = path.parent.parent.parent.name.removeprefix('darkphoton_')
            (path.parent.parent.parent / 'sample.json').write_text(json.dumps(dict(
                schema='shift-dark-photon-sample-v1', point=point, mass_gev=15.,
                proper_ctau_mm=.001 if flight is None else 1243.56890159,
                epsilon=.001 if flight is None else 2.449e-8,
                target_mean_lab_flight_m=flight, selector_kind='mean_lab_flight')))
        with patch('Logger.warn'):
            config = self.config(bsm_campaign=self.bsm_campaign, flights=(None, 100), epsilons=[])
        signals = config.samples[1:]
        self.assertEqual(signals[0].legend_description, "A' 15 GeV, prompt")
        self.assertEqual(signals[1].legend_description, "A' 15 GeV, #LT L_{lab} #GT = 100 m")
        self.assertEqual([sample.line_style for sample in signals], [1, 1])
        self.assertNotEqual(signals[0].line_color, signals[1].line_color)

    def test_multiple_masses_keep_mass_colour_and_lab_flight_line_style(self):
        for point in ('m15_lab_5m', 'm40_lab_5m', 'm40_lab_100m'):
            self.write_histograms(self.base / ('darkphoton_' + point) / self.bsm_campaign / 'histograms_merged', 5.)
        with patch('Logger.warn'):
            config = self.config(bsm_campaign=self.bsm_campaign, masses=(15, 40), flights=(5, 100), epsilons=[])
        signals = {sample.name: sample for sample in config.samples[1:]}
        for mass in (15, 40):
            self.assertEqual(signals[f'm{mass}_lab_5m'].line_color, signals[f'm{mass}_lab_100m'].line_color)
            self.assertEqual(signals[f'm{mass}_lab_5m'].line_style, 2)
            self.assertEqual(signals[f'm{mass}_lab_100m'].line_style, 4)
        self.assertNotEqual(signals['m15_lab_5m'].line_color, signals['m40_lab_5m'].line_color)

    def test_epsilon_origin_lab_legends_use_reference_flights_with_separate_colours(self):
        for point, flight, epsilon in (('m15_lab_0p06m', .060014743, 1.e-6),
                                      ('m15_lab_6em06m', .0000060014743, 1.e-4)):
            path = self.write_histograms(
                self.base / ('darkphoton_' + point) / self.bsm_campaign / 'histograms_merged', 5.)
            (path.parent.parent.parent / 'sample.json').write_text(json.dumps(dict(
                schema='shift-dark-photon-sample-v1', point=point, mass_gev=15.,
                proper_ctau_mm=.7463256382295375, epsilon=epsilon,
                target_mean_lab_flight_m=None, mean_lab_flight_m=flight,
                selector_kind='epsilon')))
        with patch('Logger.warn'):
            config = self.config(bsm_campaign=self.bsm_campaign, flights=(), epsilons=[1.e-6, 1.e-4])
        signals = config.samples[1:]
        self.assertEqual({sample.legend_description for sample in signals}, {
            "A' 15 GeV, #LT L_{lab} #GT #approx 0.06 m",
            "A' 15 GeV, #LT L_{lab} #GT #approx 6e-06 m"})
        self.assertEqual([sample.line_style for sample in signals], [1, 1])
        self.assertNotEqual(signals[0].line_color, signals[1].line_color)

    def test_native_normalization_uses_available_cutflow_and_matches_signal_area(self):
        from HistogramNormalizer import HistogramNormalizer, NormalizationType
        with patch('Logger.warn'):
            config = self.config(bsm_campaign=self.bsm_campaign)
        config.luminosity = 3.
        normalizer = HistogramNormalizer(config)
        self.addCleanup(lambda: [file.Close() for file in normalizer.input_files.values()])
        background, signal = config.samples
        self.assertEqual(normalizer.background_initial_sum_weights[(background.name, -1)], 20.)
        self.assertEqual(normalizer.signal_initial_sum_weights[signal.name], 5.)
        def histogram(sample):
            native = normalizer.input_files[sample.name].Get('available_observable').Clone(sample.name)
            native.SetDirectory(0)
            return SimpleNamespace(name='available_observable', hist=native,
                                   norm_type=NormalizationType.to_background, scale_by_bin_width=False)
        background_hist, signal_hist = histogram(background), histogram(signal)
        normalizer.normalize(background_hist, background)
        self.assertAlmostEqual(background_hist.hist.Integral(), 1.2)
        normalizer.normalize(signal_hist, signal, total_backgrounds_integral=background_hist.hist.Integral())
        self.assertAlmostEqual(signal_hist.hist.Integral(), 1.2)

    def test_rebin_one_preserves_recorded_variable_bins_and_density_area(self):
        from array import array
        from Histogram import Histogram
        from Sample import Sample, SampleType

        with patch('Logger.warn'):
            config = self.config(histogram_rebins={'dimuon/ShiftDimuonVertex_mass': 1})
        self.assertEqual(config.variable_histogram_rebin, 1)
        definition = next(histogram for histogram in config.histograms
                          if histogram.name == 'dimuon/ShiftDimuonVertex_mass')
        edges = (0., 10., 10.1, 10.2, 30., 31., 32., 50., 51., 52., 100.)
        source = self.ROOT.TH1D('native_mass_bins', '', len(edges) - 1, array('d', edges))
        source.SetDirectory(0)
        source.Sumw2()
        for index in range(1, source.GetNbinsX() + 1):
            source.SetBinContent(index, index)
            source.SetBinError(index, index ** .5)
        original_area = source.Integral()
        original_edges = tuple(source.GetBinLowEdge(index)
                               for index in range(1, source.GetNbinsX() + 2))
        histogram = Histogram(name=definition.name, rebin=definition.rebin,
                              bin_edges=definition.bin_edges,
                              scale_by_bin_width=definition.scale_by_bin_width)
        histogram.set_hist(source)
        histogram.setup(Sample(name='signal', type=SampleType.signal))
        self.assertEqual(tuple(histogram.hist.GetBinLowEdge(index)
                               for index in range(1, histogram.hist.GetNbinsX() + 2)),
                         original_edges)
        self.assertAlmostEqual(histogram.hist.Integral('width'), original_area)
        for index in range(1, histogram.hist.GetNbinsX() + 1):
            width = histogram.hist.GetBinWidth(index)
            self.assertAlmostEqual(histogram.hist.GetBinContent(index) * width, index)
            self.assertAlmostEqual(histogram.hist.GetBinError(index) * width, index ** .5)

    def test_rebin_one_keeps_all_irregular_momentum_bins_in_every_category(self):
        from array import array
        from copy import deepcopy
        from Sample import Sample, SampleType

        with patch('Logger.warn'):
            config = self.config()
        definitions = [histogram for histogram in config.histograms
                       if histogram.name.endswith('_variable')]
        self.assertEqual(len(definitions), 12)
        edges = (0., .1, .2, .3, .4, .5, 1., 2., 4., 8., 16., 32.)
        for definition in definitions:
            with self.subTest(histogram=definition.name):
                source = self.ROOT.TH1D('native_momentum_bins', '', len(edges) - 1,
                                        array('d', edges))
                source.SetDirectory(0)
                source.Sumw2()
                for index in range(1, source.GetNbinsX() + 1):
                    source.SetBinContent(index, index)
                    source.SetBinError(index, index ** .5)
                original_edges = tuple(source.GetBinLowEdge(index)
                                       for index in range(1, source.GetNbinsX() + 2))
                original_area = source.Integral()
                histogram = deepcopy(definition)
                histogram.set_hist(source)
                histogram.setup(Sample(name='signal', type=SampleType.signal))
                self.assertEqual(histogram.hist.GetNbinsX(), len(edges) - 1)
                self.assertEqual(tuple(histogram.hist.GetBinLowEdge(index)
                                       for index in range(1, histogram.hist.GetNbinsX() + 2)),
                                 original_edges)
                self.assertAlmostEqual(histogram.hist.Integral('width'), original_area)
                for index in range(1, histogram.hist.GetNbinsX() + 1):
                    width = histogram.hist.GetBinWidth(index)
                    self.assertAlmostEqual(histogram.hist.GetBinContent(index) * width, index)
                    self.assertAlmostEqual(histogram.hist.GetBinError(index) * width, index ** .5)

    def test_regular_rebin_five_survives_config_and_stack_preparation(self):
        from HistogramPlotter import HistogramPlotter

        name = 'dimuon/ShiftDimuonVertex_mass'
        with self.ROOT.TFile.Open(str(self.sm), 'UPDATE') as output:
            directory = output.mkdir('dimuon')
            directory.cd()
            source = self.ROOT.TH1D('ShiftDimuonVertex_mass', '', 100, 0., 100.)
            source.Sumw2()
            for index in range(1, 101):
                source.Fill(index - .5, 1.)
            source.Write()
        with patch('Logger.warn'):
            config = self.config(histogram_rebins={name: 5})
        definition = next(histogram for histogram in config.histograms if histogram.name == name)
        self.assertEqual(definition.rebin, 5)
        self.assertIsNone(definition.bin_edges)
        config.histograms = [definition]
        config.histograms2D = []
        config.output_path = str(self.base / 'plots')
        config.luminosity = 10.
        plotter = HistogramPlotter(config)
        for file in plotter.normalizer.input_files.values():
            self.addCleanup(file.Close)
        sample = config.samples[0]
        plotter.addHistosample(definition, sample, plotter.normalizer.input_files[sample.name])
        plotter.setupLegends()
        plotter.buildStacks()
        histogram = plotter.histosamples[0][0].hist
        self.assertEqual(histogram.GetNbinsX(), 20)
        for index in range(1, 21):
            self.assertEqual(histogram.GetBinWidth(index), 5.)
        # 100 entries * 2 pb * 10 pb^-1 / 20 original weight sum.
        self.assertAlmostEqual(histogram.Integral('width'), 100.)


if __name__ == "__main__":
    unittest.main()
