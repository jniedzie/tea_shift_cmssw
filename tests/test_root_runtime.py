import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TEA = Path(__file__).resolve().parents[1] / 'tea'
sys.path[:0] = [str(TEA / 'pylibs/helpers'), str(TEA / 'pylibs/logger')]
from teaHelpers import ensure_root_compiler_environment


class RootRuntimeTest(unittest.TestCase):
    def test_installed_conda_headers_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / 'conda-meta').mkdir()
            include = prefix / 'x86_64-conda-linux-gnu/sysroot/usr/include'
            include.mkdir(parents=True)
            (include / 'assert.h').touch()
            with patch.dict(os.environ, {'ROOTSYS': directory}, clear=True), patch.object(sys, 'platform', 'linux'):
                ensure_root_compiler_environment()
                self.assertEqual(os.environ['CONDA_BUILD_SYSROOT'], str(include.parent.parent))

    def test_explicit_sysroot_preserved(self):
        with patch.dict(os.environ, {'CONDA_BUILD_SYSROOT': '/custom/sysroot'}, clear=True):
            ensure_root_compiler_environment()
            self.assertEqual(os.environ['CONDA_BUILD_SYSROOT'], '/custom/sysroot')

    def test_non_conda_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'ROOTSYS': directory}, clear=True):
                ensure_root_compiler_environment()
                self.assertNotIn('CONDA_BUILD_SYSROOT', os.environ)


if __name__ == '__main__':
    unittest.main()
