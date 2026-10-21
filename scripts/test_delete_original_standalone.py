"""Standalone test for remove_real_and_linked_file (issue #297).

This script tests the function in isolation, avoiding the torch/safetensors
import chain.  It lives in scripts/ rather than tests/ so unittest discovery
does not pick it up (it contains a copied function, not a production import).

Symlink tests are skipped on Windows (requires admin or developer mode).
"""

import os
import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


# ── Copy of the fixed function (from airllm/utils.py) ──────────────────
# NOTE: This is intentionally duplicated for portability.  Keep in sync
# with the real implementation in air_llm/airllm/utils.py.

def remove_real_and_linked_file(to_delete):
    """Remove a file, following symlinks to also remove the target if present."""
    to_delete = os.fspath(to_delete)
    targetpath = None

    if os.path.islink(to_delete):
        targetpath = os.path.realpath(to_delete)

    try:
        os.remove(to_delete)
    except FileNotFoundError:
        return

    if targetpath is not None:
        try:
            os.remove(targetpath)
        except FileNotFoundError:
            pass


# ── Tests ──────────────────────────────────────────────────────────────

class TestRemoveRealAndLinkedFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        for entry in os.listdir(self.tmpdir):
            path = os.path.join(self.tmpdir, entry)
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        os.rmdir(self.tmpdir)

    # --- regular files ---------------------------------------------------

    def test_regular_file_str_path(self):
        """A normal file (str input) is deleted exactly once."""
        filepath = os.path.join(self.tmpdir, "model-00001.safetensors")
        with open(filepath, "w") as f:
            f.write("fake weights")
        remove_real_and_linked_file(filepath)
        self.assertFalse(os.path.exists(filepath))

    def test_regular_file_path_object(self):
        """A normal file (Path input) is deleted without UnboundLocalError."""
        filepath = Path(self.tmpdir) / "model-00002.safetensors"
        filepath.write_text("fake weights")
        remove_real_and_linked_file(filepath)
        self.assertFalse(filepath.exists())

    def test_regular_file_no_double_delete(self):
        """targetpath should be None for a regular file — no double-delete."""
        filepath = os.path.join(self.tmpdir, "shard.bin")
        with open(filepath, "w") as f:
            f.write("data")
        # The old code raised UnboundLocalError here
        remove_real_and_linked_file(filepath)
        self.assertFalse(os.path.exists(filepath))

    # --- edge cases ------------------------------------------------------

    def test_missing_file_does_not_crash(self):
        """Deleting a non-existent path should not raise."""
        nonexistent = os.path.join(self.tmpdir, "does-not-exist.bin")
        remove_real_and_linked_file(nonexistent)  # should not raise

    def test_multiple_files_independently(self):
        """Deleting several files in sequence works."""
        for name in ["a.bin", "b.bin", "c.bin"]:
            with open(os.path.join(self.tmpdir, name), "w") as f:
                f.write(name)
        for name in ["a.bin", "b.bin", "c.bin"]:
            remove_real_and_linked_file(os.path.join(self.tmpdir, name))
        self.assertEqual(os.listdir(self.tmpdir), [])

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

    @unittest.skipUnless(platform.system() != "Windows", "symlink tests require admin on Windows")
    def test_broken_symlink_does_not_crash(self):
        """Deleting a symlink whose target is already gone should not raise."""
        link = os.path.join(self.tmpdir, "broken-link.bin")
        os.symlink("/nonexistent/target", link)
        remove_real_and_linked_file(link)
        self.assertFalse(os.path.exists(link))


if __name__ == "__main__":
    unittest.main()
