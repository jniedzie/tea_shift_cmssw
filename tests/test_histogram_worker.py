"""Runtime transfer recovery must never accept a partial or corrupt payload."""
import importlib.util
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('worker', Path(__file__).resolve().parents[1] / 'utils/histogram_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class RuntimeTransferTest(unittest.TestCase):
    def run_download(self, failures, corrupt=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Test the actual shell retry block with deterministic transfer failures.
            script = r'''set -euo pipefail
tea_stage="$1"
tea_url="synthetic"
timeout() {
  n=$(cat "$tea_stage/attempts" 2>/dev/null || echo 0)
  n=$((n+1)); echo "$n" > "$tea_stage/attempts"
  echo partial > "$tea_stage/runtime.tar.gz"
  if [[ "$n" -le FAILURES ]]; then return 124; fi
  echo PAYLOAD > "$tea_stage/runtime.tar.gz"
}
sleep() { :; }
''' .replace('FAILURES', str(failures)).replace('PAYLOAD', 'wrong' if corrupt else 'complete')
            script += worker.DOWNLOAD + '\n'
            digest = hashlib.sha256(b'complete\n').hexdigest()
            script += f'printf "%s  %s\\n" "{digest}" "$tea_stage/runtime.tar.gz" | sha256sum --check --status\n'
            script += 'echo published > "$tea_stage/published"\n'
            result = subprocess.run(['bash', '-c', script, 'test', str(root)], capture_output=True, text=True)
            return result.returncode, int((root / 'attempts').read_text()), (root / 'published').exists()

    def test_transient_timeout_retries_before_publication(self):
        self.assertEqual(self.run_download(2), (0, 3, True))

    def test_repeated_timeout_is_bounded_and_never_publishes(self):
        self.assertEqual(self.run_download(3), (75, 3, False))

    def test_bad_payload_is_fatal_and_never_publishes(self):
        self.assertEqual(self.run_download(0, corrupt=True), (1, 1, False))

    def test_unknown_wrapper_rejected(self):
        with self.assertRaises(ValueError):
            worker.harden_worker('#!/bin/bash\nexit 0\n')

    def test_only_runtime_timeouts_request_scheduler_retry(self):
        for phase, status, expected in [('runtime extraction', 124, 75),
                                        ('runtime relocation', 137, 75),
                                        ('runtime relocation', 1, 1),
                                        ('histogramming and publication', 124, 124),
                                        ('histogramming and publication', 137, 137),
                                        ('histogramming and publication', 75, 1)]:
            with self.subTest(phase=phase, status=status):
                script = f'set -euo pipefail\ntea_phase="{phase}"\n{worker.ERROR_HANDLER}\nbash -c "exit {status}"\n'
                result = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
                self.assertEqual(result.returncode, expected)

    def test_submit_preserves_exact_inputs_and_sets_bounded_recovery(self):
        original = 'executable = old.sh\ntransfer_input_files = payload.tar.gz,exact_inputs.txt\nmax_materialize = 1000\non_exit_hold = ExitCode != 0\nqueue 9911\n'
        prepared = worker.harden_submit(original, Path('/tmp/new_worker.sh'))
        self.assertIn('transfer_input_files = payload.tar.gz,exact_inputs.txt\n', prepared)
        self.assertTrue(prepared.endswith('queue 9911\n'))
        self.assertIn('max_materialize = 100\n', prepared)
        self.assertIn('max_idle = 20\n', prepared)
        self.assertIn('request_memory = 6000\n', prepared)
        self.assertIn('executable = /tmp/new_worker.sh\n', prepared)
        self.assertIn('on_exit_hold = ' + worker.ON_EXIT_HOLD + '\n', prepared)
        self.assertIn('on_exit_remove = ' + worker.ON_EXIT_REMOVE + '\n', prepared)
        self.assertIn('requirements = (' + worker.RETRY_MACHINE_GUARD + ')\n', prepared)

    def test_retry_migration_preserves_original_machine_requirements(self):
        prepared = worker.harden_submit('requirements = TARGET.Arch == "X86_64"\nqueue 1\n', Path('/tmp/worker.sh'))
        self.assertIn('requirements = (TARGET.Arch == "X86_64") && (' + worker.RETRY_MACHINE_GUARD + ')\n', prepared)

    def test_ambiguous_submit_setting_rejected(self):
        with self.assertRaises(ValueError):
            worker.harden_submit('on_exit_hold = False\non_exit_hold = True\nqueue 1\n', Path('/tmp/worker.sh'))


if __name__ == '__main__':
    unittest.main()
