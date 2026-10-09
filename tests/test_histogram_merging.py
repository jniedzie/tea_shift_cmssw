"""Merge the preceding step's outputs with exact inventories and worker checks."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "bin"))
spec = importlib.util.spec_from_file_location("tea_merge", PROJECT / "bin/merge.py")
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


class HistogramMergeTest(unittest.TestCase):
    def test_groups_exact_inputs_by_process_bin(self):
        jobs = merge.collect_explicit_jobs([
            ("/nano/a.root", "", "/qcd/a.root"),
            ("/nano/b.root", "", "/jpsi/a.root"),
            ("/nano/a.root", "", "/qcd/b.root"),
        ])
        self.assertEqual({job[5]: job[-1] for job in jobs}, {
            "/qcd_merged/ntuple_0.root": ["/qcd/a.root", "/qcd/b.root"],
            "/jpsi_merged/ntuple_0.root": ["/jpsi/a.root"],
        })

    def test_duplicate_input_and_output_alias_rejected(self):
        for entries in ([('nano/a', '', '/hist/a.root'), ('nano/b', '', '/hist/a.root')],
                        [('nano/a', '', '/hist/a.root'), ('nano/b', '', '/hist_merged/ntuple_0.root')]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                merge.collect_explicit_jobs(entries)

    def test_tree_and_histogram_outputs_are_separate(self):
        jobs = merge.collect_explicit_jobs([
            ('/nano/a.root', '/sample/trees/a.root', '/sample/histograms/a.root'),
            ('/nano/b.root', '/sample/trees/b.root', '/sample/histograms/b.root'),
            ('/nano/c.root', '', '/sample/histograms/c.root'),
        ])
        self.assertEqual({job[0]: (job[5], job[-1]) for job in jobs}, {
            'trees': ('/sample/trees_merged/ntuple_0.root', ['/sample/trees/a.root', '/sample/trees/b.root']),
            'histograms': ('/sample/histograms_merged/ntuple_0.root',
                           ['/sample/histograms/a.root', '/sample/histograms/b.root', '/sample/histograms/c.root']),
        })

    def test_exact_outputs_support_batching_and_provenance(self):
        jobs = merge.collect_explicit_jobs([
            ('nano', '', f'/sample/histograms/{name}.root') for name in ('c', 'a', 'b')
        ], chunk_size=2, provenance_tag='abcdef0')
        self.assertEqual([job[5] for job in jobs], [
            '/sample/histograms_merged/ntuple_0_abcdef0.root',
            '/sample/histograms_merged/ntuple_1_abcdef0.root',
        ])
        self.assertEqual([job[-1] for job in jobs], [
            ['/sample/histograms/a.root', '/sample/histograms/b.root'], ['/sample/histograms/c.root'],
        ])

    def test_key_filter_inspects_produced_outputs(self):
        with patch.object(merge, 'skip_files_without_keys', return_value=['/hist/a.root']) as filter_keys:
            jobs = merge.collect_explicit_jobs([
                ('/nano/a.root', '', '/hist/a.root'), ('/nano/b.root', '', '/hist/empty.root'),
            ], skip_no_keys=True)
        filter_keys.assert_called_once_with(['/hist/a.root', '/hist/empty.root'])
        self.assertEqual(jobs[0][-1], ['/hist/a.root'])

    def test_exact_inventory_takes_precedence_over_directory_globs(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'files.py'
            config.write_text(
                "input_output_file_list = [('nano.root', '', '/hist/a.root')]\n"
                "output_hists_dir = '/stale/histograms'\n"
            )
            with patch.object(sys, 'argv', ['merge.py', '--files_config', str(config), '--dry']), \
                 patch.object(merge, 'collect_jobs') as discover, patch.object(merge, 'print_job_summary') as summary:
                merge.main()
            discover.assert_not_called()
            self.assertEqual(summary.call_args.args[0][0][-1], ['/hist/a.root'])

    def test_output_list_provider_uses_same_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'files.py'
            config.write_text(
                "def get_input_output_file_lists():\n"
                "  return [[('nano/a.root', '', '/hist/a.root')], [('nano/b.root', '', '/hist/b.root')]]\n"
            )
            with patch.object(sys, 'argv', ['merge.py', '--files_config', str(config), '--dry']), \
                 patch.object(merge, 'print_job_summary') as summary:
                merge.main()
            self.assertEqual(summary.call_args.args[0][0][-1], ['/hist/a.root', '/hist/b.root'])

    def test_missing_input_stops_before_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "files.py"
            config.write_text(f"input_output_file_list = [('nano.root', '', '{temporary}/missing.root')]\n")
            with patch.object(sys, "argv", ['merge.py', '--files_config', str(config), '--condor']), \
                 patch.object(merge, "submit_condor_jobs") as submit:
                with self.assertRaisesRegex(RuntimeError, 'incomplete.*No merge jobs submitted'):
                    merge.main()
                submit.assert_not_called()

    def test_incomplete_snapshot_keeps_bins_separate_and_skips_empty_bins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / 'good.root'
            good.write_bytes(b'present')
            empty = root / 'empty.root'
            empty.touch()
            jobs = merge.collect_explicit_jobs([
                ('nano/a.root', '', str(good)),
                ('nano/b.root', '', str(empty)),
                ('nano/c.root', '', str(root / 'jpsi/missing.root')),
            ])
            with patch.object(merge, 'warn') as warn:
                selected = merge.select_available_jobs(jobs, True)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0][5], str(root) + '_merged/ntuple_0.root')
            self.assertEqual(selected[0][-1], [str(good)])
            self.assertIn('jpsi_merged', warn.call_args.args[0])
            self.assertIn(root.name + '_merged', warn.call_args.args[0])

    def test_no_available_inputs_produces_no_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'files.py'
            config.write_text(f"input_output_file_list = [('nano.root', '', '{temporary}/missing.root')]\n")
            with patch.object(sys, 'argv', ['merge.py', '--files_config', str(config), '--condor', '--allow-incomplete']), \
                 patch.object(merge, 'submit_condor_jobs') as submit, patch.object(merge, 'warn'):
                self.assertEqual(merge.cli(), 0)
                submit.assert_not_called()

    def test_allow_incomplete_is_available_to_files_config_only_while_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'files.py'
            config.write_text(
                'import os\n'
                'if os.environ.get("SHIFT_ALLOW_INCOMPLETE_NANO") != "1":\n'
                '  raise RuntimeError("partial inventory flag missing")\n'
                'input_output_file_list = []\n'
            )
            with patch.object(sys, 'argv', ['merge.py', '--files_config', str(config), '--allow-incomplete']):
                self.assertEqual(merge.cli(), 0)
            self.assertNotIn('SHIFT_ALLOW_INCOMPLETE_NANO', os.environ)

    def test_allow_incomplete_dry_run_freezes_only_available_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / 'good.root'
            good.write_bytes(b'present')
            config = root / 'files.py'
            config.write_text(f"input_output_file_list = [('nano/a.root', '', '{good}'), "
                              f"('nano/b.root', '', '{root}/missing.root')]\n")
            with patch.object(sys, 'argv', ['merge.py', '--files_config', str(config), '--condor', '--dry', '--allow-incomplete']), \
                 patch.object(merge, 'submit_condor_jobs') as submit, \
                 patch.object(merge, 'print_job_summary') as summary, \
                 patch.object(merge, 'get_facility', return_value='default'), \
                 patch.object(merge.tempfile, 'mkdtemp', return_value=temporary), \
                 patch.object(merge, 'warn'):
                merge.main()
                self.assertEqual(submit.call_args.args[1][0][-1], [str(good)])
                self.assertTrue(submit.call_args.args[-1])
                summary.assert_not_called()

    def test_cli_reports_incomplete_run_through_logger_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'files.py'
            config.write_text(f"input_output_file_list = [('nano.root', '', '{temporary}/missing.root')]\n")
            for flags in ([], ['--condor']):
                with self.subTest(flags=flags):
                    result = subprocess.run([sys.executable, str(PROJECT / 'bin/merge.py'),
                        '--files_config', str(config), *flags], capture_output=True, text=True,
                        env=dict(os.environ, PYTHONPATH=str(PROJECT / 'bin')), timeout=30)
                    output = result.stdout + result.stderr
                    self.assertEqual(result.returncode, 1, output)
                    self.assertIn('FATAL: Merge failed: Histogram production is incomplete', output)
                    self.assertIn('--allow-incomplete', output)
                    self.assertNotIn('Traceback', output)

    def test_bad_root_input_stops_before_hadd_or_publication(self):
        with patch.object(merge, "validate_root_file", side_effect=RuntimeError('bad ROOT')), \
             patch.object(merge, "run_command") as run:
            with self.assertRaisesRegex(RuntimeError, 'bad ROOT'):
                merge.run_condor_merge(dict(output_file='/tmp/out.root', input_files=['/tmp/bad.root']))
            run.assert_not_called()

    def test_hadd_failure_does_not_publish(self):
        with patch.object(merge, "validate_root_file"), \
             patch.object(merge, "run_command", side_effect=RuntimeError('hadd failed')) as run:
            with self.assertRaisesRegex(RuntimeError, 'hadd failed'):
                merge.run_condor_merge(dict(output_file='/eos/home-j/jniedzie/out.root',
                    input_files=['/tmp/input.root'], preserve_input_compression=False,
                    hadd_files_per_pass=100, hadd_workers=1))
            self.assertEqual(run.call_count, 1)
            self.assertNotIn('-k', run.call_args.args[0])

    def test_eos_uses_remote_inputs_and_checked_publication(self):
        with patch.object(merge, "validate_root_file") as validate, \
             patch.object(merge, "run_command") as run:
            merge.run_condor_merge(dict(output_file='/eos/home-j/jniedzie/bin_merged/out.root',
                input_files=['/eos/home-j/jniedzie/bin/input.root'], preserve_input_compression=False,
                hadd_files_per_pass=100, hadd_workers=1))
            self.assertEqual(validate.call_args_list[0].args[0],
                             'root://eoshome-j.cern.ch//eos/user/j/jniedzie/bin/input.root')
            self.assertEqual(run.call_args_list[1].args[0],
                ['xrdfs', 'root://eoshome-j.cern.ch', 'mkdir', '-p', '/eos/user/j/jniedzie/bin_merged'])
            publication = run.call_args_list[2].args[0]
            self.assertIn('--posc', publication)
            self.assertIn('--cksum', publication)

    def test_generated_worker_merges_nested_weighted_histograms(self):
        import ROOT
        with tempfile.TemporaryDirectory(dir=PROJECT) as temporary:
            root = Path(temporary)
            (root / 'histograms').mkdir()
            inputs = []
            for index, weight in enumerate((2.0, 3.0)):
                path = root / 'histograms' / f'input{index}.root'
                with ROOT.TFile(str(path), 'RECREATE') as output:
                    output.mkdir('event').cd()
                    hist = ROOT.TH1D('weighted', '', 2, 0, 2)
                    hist.Sumw2()
                    hist.Fill(0.5, weight)
                    hist.Write()
                inputs.append(str(path))
            # Exercise local CLI filtering with two present files and one missing.
            local_destination = root / 'histograms_merged/ntuple_0.root'
            files_config = root / 'files.py'
            files_config.write_text('input_output_file_list = ' + repr([
                (str(root / f'nano{index}.root'), '', path)
                for index, path in enumerate([*inputs, str(root / 'histograms/missing.root')])
            ]))
            result = subprocess.run([sys.executable, str(PROJECT / 'bin/merge.py'),
                '--files_config', str(files_config), '--allow-incomplete', '--hadd-workers', '1'],
                capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(PROJECT / 'bin')), timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('Incomplete histogram productions:', result.stdout)
            self.assertNotIn('RuntimeWarning', result.stdout + result.stderr)
            with ROOT.TFile.Open(str(local_destination)) as output:
                hist = output.Get('event/weighted')
                self.assertEqual(hist.GetEntries(), 2)
                self.assertEqual(hist.GetBinContent(1), 5)
                self.assertAlmostEqual(hist.GetBinError(1), 13 ** 0.5)
            destination = root / 'histograms_merged/ntuple_0.root'
            script, config = merge.create_condor_job(str(root), 'histograms', 'qcd/bin', 0,
                str(destination), inputs, False, 100, 1)
            result = subprocess.run(['bash', script], cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with ROOT.TFile.Open(str(destination)) as output:
                hist = output.Get('event/weighted')
                self.assertEqual(hist.GetEntries(), 2)
                self.assertEqual(hist.GetBinContent(1), 5)
                self.assertAlmostEqual(hist.GetBinError(1), 13 ** 0.5)


if __name__ == '__main__':
    unittest.main()
