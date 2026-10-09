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

    def test_application_signal_is_reported_with_conventional_exit_code(self):
        with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=-7)), \
             patch.object(runner, 'fatal') as fatal:
            self.assertEqual(runner.execute(['./app']), 135)
            self.assertIn('SIGBUS', fatal.call_args.args[0])

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
        self.assertIn('--force', calls[-1])
        self.assertIn('--cksum', calls[-1])
        self.assertTrue(calls[-1][-1].endswith('/eos/user/j/jniedzie/a/histograms_job0.root'))


class SubmitterFatalErrorTest(unittest.TestCase):
    def test_config_errors_exit_before_starting_jobs_without_traceback(self):
        # Exercise the real CLI/config loader, while making a submission an
        # explicit failure. No ROOT environment or scheduler is required.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'SubmissionManager.py').write_text(
                'from enum import Enum\n'
                'class SubmissionSystem(Enum):\n'
                '  unknown=0; local=1; condor=2; local_parallel=3\n'
                'class SubmissionManager:\n'
                '  def __init__(self, *args):\n'
                '    raise AssertionError("A job was started")\n')
            environment = dict(os.environ, PYTHONPATH=os.pathsep.join([
                str(root), str(PROJECT / 'tea/pylibs/logger')]))
            for contents, reason in (
                (None, 'No such file'),
                ('raise RuntimeError("Nano production is not marked complete; wait for completion")',
                 'Nano production is not marked complete; wait for completion'),
                ('broken = [', 'files.py, line 1'),
            ):
                with self.subTest(reason=reason):
                    config = root / 'files.py'
                    if contents is not None:
                        config.write_text(contents)
                    result = subprocess.run([
                        sys.executable, str(PROJECT / 'tea/apps/examples/submitter.py'),
                        '--app', 'shift_histogrammer', '--config', 'config.py',
                        '--files_config', str(config), '--local'],
                        env=environment, capture_output=True, text=True, timeout=30)
                    output = result.stdout + result.stderr
                    self.assertEqual(result.returncode, 1, output)
                    self.assertIn('FATAL: Submission preparation failed', output)
                    self.assertIn(str(config), output)
                    self.assertIn(reason, output)
                    self.assertNotIn('Traceback', output)
                    self.assertNotIn('A job was started', output)


@unittest.skipUnless(shutil.which('condor_submit') and str(PROJECT).startswith('/afs/'),
                     'Native Condor parser and writable AFS checkout required')
class ExistingSubmitterCondorTest(unittest.TestCase):
    def test_cern_batch_rejects_eos_python_runtime(self):
        from SubmissionManager import SubmissionManager, SubmissionSystem
        manager = object.__new__(SubmissionManager)
        manager.submission_system = SubmissionSystem.condor
        with patch('SubmissionManager.get_facility', return_value='lxplus'), \
             patch.object(sys, 'executable', '/eos/user/j/test/environment/bin/python'):
            with self.assertRaisesRegex(RuntimeError, 'Activate an AFS TEA environment'):
                manager._SubmissionManager__set_python_executable()

    def test_existing_template_honors_cap_and_shards_real_native_ads(self):
        from SubmissionManager import SubmissionManager, SubmissionSystem
        with tempfile.TemporaryDirectory(dir=PROJECT) as temporary:
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
                    with patch('SubmissionManager.get_facility', return_value='lxplus'), \
                         patch.dict(os.environ, {'TEA_CONDOR_DIR': str(root / 'staging'),
                                                   'CONDA_BUILD_SYSROOT': '/test/sysroot with spaces'}):
                        manager._SubmissionManager__setup_temp_file_paths()
                        manager._SubmissionManager__copy_templates()
                        manager._SubmissionManager__set_run_script_variables()
                        self.assertIn("export CONDA_BUILD_SYSROOT='/test/sysroot with spaces'",
                                      Path(manager.condor_run_script_name).read_text())
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
