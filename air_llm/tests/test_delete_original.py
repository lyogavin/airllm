"""Tests for remove_real_and_linked_file (issue #297).

Covers:
- Regular file deletion
- Symlink deletion (link removed, target removed)
- Path object input (not just str)
- Missing file does not crash
"""

import importlib.util
import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path

# Import utils directly to avoid pulling in torch via airllm.__init__
_utils_path = os.path.join(os.path.dirname(__file__), "..", "airllm", "utils.py")
_spec = importlib.util.spec_from_file_location("airllm_utils", _utils_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
remove_real_and_linked_file = _mod.remove_real_and_linked_file


class TestRemoveRealAndLinkedFile(unittest.TestCase):
    """Unit tests for remove_real_and_linked_file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        for entry in os.listdir(self.tmpdir):
            path = os.path.join(self.tmpdir, entry)
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
        os.rmdir(self.tmpdir)

    # --- regular files ---------------------------------------------------

    def test_regular_file_str_path(self):
        """A normal file (str input) is deleted exactly once."""
        filepath = os.path.join(self.tmpdir, "model-00001-of-00002.safetensors")
        with open(filepath, "w") as f:
            f.write("fake weights")

        remove_real_and_linked_file(filepath)

        self.assertFalse(os.path.exists(filepath))

    def test_regular_file_path_object(self):
        """A normal file (Path input) is deleted without UnboundLocalError."""
        filepath = Path(self.tmpdir) / "model-00002-of-00002.safetensors"
        filepath.write_text("fake weights")

        remove_real_and_linked_file(filepath)

        self.assertFalse(filepath.exists())

    # --- symlinks --------------------------------------------------------

    @unittest.skipUnless(platform.system() != "Windows", "symlink tests require admin on Windows")
    def test_symlink_removes_link_and_target(self):
        """When to_delete is a symlink, both link and target are removed."""
        target = os.path.join(self.tmpdir, "blobs", "abc123")
        os.makedirs(os.path.dirname(target))
        with open(target, "w") as f:
            f.write("real data")

        link = os.path.join(self.tmpdir, "snapshots", "model.safetensors")
        os.makedirs(os.path.dirname(link))
        os.symlink(target, link)

        remove_real_and_linked_file(link)

        self.assertFalse(os.path.exists(link))
        self.assertFalse(os.path.exists(target))

    @unittest.skipUnless(platform.system() != "Windows", "symlink tests require admin on Windows")
    def test_symlink_with_path_object(self):
        """Symlink removal works with Path input too."""
        target = Path(self.tmpdir) / "blob.bin"
        target.write_bytes(b"\x00" * 64)

        link = Path(self.tmpdir) / "link.bin"
        os.symlink(str(target), str(link))

        remove_real_and_linked_file(link)

        self.assertFalse(link.exists())
        self.assertFalse(target.exists())

    # --- edge cases ------------------------------------------------------

    def test_missing_file_does_not_crash(self):
        """Deleting a non-existent path should not raise."""
        nonexistent = os.path.join(self.tmpdir, "does-not-exist.bin")
        # Should not raise
        remove_real_and_linked_file(nonexistent)

    @unittest.skipUnless(platform.system() != "Windows", "symlink tests require admin on Windows")
    def test_broken_symlink_does_not_crash(self):
        """Deleting a symlink whose target is already gone should not raise."""
        link = os.path.join(self.tmpdir, "broken-link.bin")
        os.symlink("/nonexistent/target", link)

        remove_real_and_linked_file(link)

        self.assertFalse(os.path.exists(link))


if __name__ == "__main__":
    unittest.main()
