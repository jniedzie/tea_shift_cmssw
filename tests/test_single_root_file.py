from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "configs"))
from shift_sample_paths import single_root_file


class SingleRootFileTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_arbitrary_filename_and_unrelated_entries(self):
        expected = self.directory / "unknown name_abc123.root"
        expected.touch()
        (self.directory / "metadata.json").touch()
        (self.directory / "unfinished.root.partial").touch()
        (self.directory / "directory.root").mkdir()
        self.assertEqual(single_root_file(str(self.directory)), str(expected))

    def test_empty_directory_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            single_root_file(self.directory)

    def test_multiple_files_rejected(self):
        for name in ("a.root", "b.root"):
            (self.directory / name).touch()
        with self.assertRaisesRegex(RuntimeError, "found 2: a.root, b.root"):
            single_root_file(self.directory)

    def test_missing_directory_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "does not exist or is not accessible"):
            single_root_file(self.directory / "missing")

    def test_nested_files_not_selected(self):
        nested = self.directory / "nested"
        nested.mkdir()
        (nested / "histograms.root").touch()
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            single_root_file(self.directory)

    def test_optional_missing_and_empty_directories(self):
        self.assertIsNone(single_root_file(self.directory / "missing", allow_missing=True))
        self.assertIsNone(single_root_file(self.directory, allow_missing=True))

    def test_optional_inaccessible_directory(self):
        with patch.object(Path, "iterdir", side_effect=PermissionError("Permission denied")):
            self.assertIsNone(single_root_file(self.directory, allow_missing=True))

    def test_optional_existing_file_selected(self):
        expected = self.directory / "histograms.root"
        expected.touch()
        self.assertEqual(single_root_file(self.directory, allow_missing=True), str(expected))

    def test_optional_multiple_files_still_rejected(self):
        for name in ("a.root", "b.root"):
            (self.directory / name).touch()
        with self.assertRaisesRegex(RuntimeError, "found 2: a.root, b.root"):
            single_root_file(self.directory, allow_missing=True)


if __name__ == "__main__":
    unittest.main()
