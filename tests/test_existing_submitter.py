"""Check the existing submitter/runner without executing or submitting jobs."""

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'bin'))
runner_spec = importlib.util.spec_from_file_location('condor_runner', PROJECT / 'tea/apps/examples/condor_runner.py')
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


class ExistingRunnerTest(unittest.TestCase):
    def test_eos_home_alias(self):
        self.assertEqual(runner.eos_path('/eos/home-j/jniedzie/a/nano.root'), '/eos/user/j/jniedzie/a/nano.root')

    def test_child_failure_is_returned_without_publishing(self):
        with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=17)) as run:
            self.assertEqual(runner.execute(['./app', '--input_path', '/tmp/input.root']), 17)
            self.assertEqual(run.call_count, 1)

    def test_eos_output_is_staged_and_published_with_checksum(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command.copy())
            if command[0] == './app':
                Path(command[command.index('--output_hists_path') + 1]).write_bytes(b'synthetic histogram')
            return SimpleNamespace(returncode=0)
        with patch.object(runner.subprocess, 'run', side_effect=run):
            result = runner.execute(['./app', '--input_path', '/eos/home-j/jniedzie/a/nano.root',
                                     '--output_hists_path', '/eos/home-j/jniedzie/a/histograms_job0.root'])
        self.assertEqual(result, 0)
        self.assertEqual(calls[0][0], 'xrdcp')
        self.assertTrue(calls[0][-2].endswith('/eos/user/j/jniedzie/a/nano.root'))
        self.assertTrue(calls[1][calls[1].index('--input_path') + 1].startswith('/tmp/'))
        self.assertEqual(calls[-1][0], 'xrdcp')
        self.assertIn('--posc', calls[-1])
        self.assertIn('--cksum', calls[-1])
        self.assertTrue(calls[-1][-1].endswith('/eos/user/j/jniedzie/a/histograms_job0.root'))


@unittest.skipUnless(shutil.which('condor_submit'), 'Native Condor parser unavailable')
class ExistingSubmitterCondorTest(unittest.TestCase):
    def test_existing_template_honors_cap_and_shards_real_native_ads(self):
        from SubmissionManager import SubmissionManager, SubmissionSystem
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'tea').symlink_to(PROJECT / 'tea', target_is_directory=True)
            work = root / 'bin'
            work.mkdir()
            files = work / 'files.py'
            files.write_text("input_output_file_list = [('/tmp/input.root', '', '/tmp/output.root')]\n")
            previous = Path.cwd()
            try:
                os.chdir(work)
                for process in (0, 499, 500, 1001):
                    manager = SubmissionManager(SubmissionSystem.condor, 'shift_histogrammer', 'config.py', str(files))
                    manager.job_flavour, manager.memory_request = 'longlunch', 2.0
                    manager.materialize_max, manager.resubmit_job = 1000, process
                    manager.save_logs = True
                    with patch('SubmissionManager.get_facility', return_value='lxplus'):
                        manager._SubmissionManager__setup_temp_file_paths()
                        manager._SubmissionManager__copy_templates()
                        manager._SubmissionManager__set_condor_script_variables(1002)
                    preview = work / f'preview_{process}.ads'
                    environment = {key: value for key, value in os.environ.items()
                                   if key not in ('BASH_ENV', 'ENV', 'CONDOR_CONFIG') and not key.startswith('_CONDOR_')}
                    environment.update(CONDOR_CONFIG='/dev/null', SKIP_LOCAL_CONFIG_FILE='TRUE',
                                       _CONDOR_SKIP_LOCAL_CONFIG_FILE='TRUE', _CONDOR_NETWORK_INTERFACE='127.0.0.1',
                                       _CONDOR_FULL_HOSTNAME='localhost')
                    result = subprocess.run([shutil.which('condor_submit'), '-dry-run', str(preview), manager.condor_config_name],
                                            env=environment, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    ad = preview.read_text()
                    output = re.search(r'^Out\s*=\s*"([^"]+)"', ad, re.M).group(1)
                    error = re.search(r'^Err\s*=\s*"([^"]+)"', ad, re.M).group(1)
                    self.assertIn(f'output/{process // 500:03d}/', output)
                    self.assertIn(f'error/{process // 500:03d}/', error)
                    self.assertIn('JobMaterializeMaxIdle=1000', ad.replace(' ', ''))
                    self.assertIn('Environment=""', ad.replace(' ', ''))
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
