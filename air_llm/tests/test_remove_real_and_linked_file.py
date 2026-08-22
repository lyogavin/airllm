import os
import tempfile
import unittest

from airllm.utils import remove_real_and_linked_file


class TestRemoveRealAndLinkedFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_removes_normal_absolute_file_once(self):
        path = os.path.join(self.tmpdir.name, 'shard.bin')
        with open(path, 'w') as f:
            f.write('data')

        remove_real_and_linked_file(path)

        self.assertFalse(os.path.exists(path))

    def test_removes_normal_relative_file_without_crash(self):
        # Relative paths previously crashed: os.path.realpath() always returns an absolute
        # path, so the old string comparison treated any relative path as "linked" and tried
        # to remove the same underlying file twice.
        path = os.path.join(self.tmpdir.name, 'shard_rel.bin')
        with open(path, 'w') as f:
            f.write('data')

        cwd = os.getcwd()
        try:
            os.chdir(self.tmpdir.name)
            remove_real_and_linked_file('shard_rel.bin')
        finally:
            os.chdir(cwd)

        self.assertFalse(os.path.exists(path))

    def test_removes_symlink_and_its_target(self):
        target = os.path.join(self.tmpdir.name, 'blob.bin')
        link = os.path.join(self.tmpdir.name, 'snapshot_link.bin')
        with open(target, 'w') as f:
            f.write('data')
        os.symlink(target, link)

        remove_real_and_linked_file(link)

        self.assertFalse(os.path.exists(link))
        self.assertFalse(os.path.exists(target))


if __name__ == '__main__':
    unittest.main()
