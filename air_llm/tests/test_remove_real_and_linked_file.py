"""Tests for deleting original checkpoint files."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


_AIRLLM_DIR = Path(__file__).resolve().parents[1] / "airllm"

if "airllm" not in sys.modules:
    _pkg = types.ModuleType("airllm")
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules["airllm"] = _pkg

from airllm.utils import remove_real_and_linked_file


class TestRemoveRealAndLinkedFile(unittest.TestCase):
    def test_regular_file_is_removed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"weights")

            remove_real_and_linked_file(path)

            self.assertFalse(path.exists())

    def test_relative_path_to_regular_file_is_removed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            absolute_path = Path(tmp) / "weights.bin"
            absolute_path.write_bytes(b"weights")
            relative_path = Path(os.path.relpath(absolute_path))

            remove_real_and_linked_file(relative_path)

            self.assertFalse(absolute_path.exists())

    def test_absolute_string_path_is_removed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"weights")

            remove_real_and_linked_file(str(path))

            self.assertFalse(path.exists())

    def test_symlink_and_target_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.bin"
            link = Path(tmp) / "weights-link.bin"
            target.write_bytes(b"weights")
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symbolic links are not available")

            remove_real_and_linked_file(link)

            self.assertFalse(link.exists())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
