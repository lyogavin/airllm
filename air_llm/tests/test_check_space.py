import os
import tempfile
import unittest
from pathlib import Path

import airllm.utils as utils
from airllm.utils import check_space, NotEnoughSpaceException


class TestCheckSpace(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        # one "checkpoint" file of exactly 1000 bytes
        with open(self.root / 'pytorch_model.bin', 'wb') as f:
            f.write(b'\0' * 1000)
        self._orig_disk_usage = utils.shutil.disk_usage

    def tearDown(self):
        utils.shutil.disk_usage = self._orig_disk_usage
        self.tmpdir.cleanup()

    def _set_free(self, free_bytes):
        utils.shutil.disk_usage = lambda _p: (10 ** 12, 0, free_bytes)

    def test_no_compression_thresholds(self):
        self._set_free(1100)   # > 1000 original
        check_space(self.root)  # should not raise
        self._set_free(900)    # < 1000
        with self.assertRaises(NotEnoughSpaceException):
            check_space(self.root)

    def test_8bit_needs_about_half(self):
        # 8-bit split ~= 500 bytes
        self._set_free(600)
        check_space(self.root, compression='8bit')  # should not raise
        self._set_free(400)
        with self.assertRaises(NotEnoughSpaceException):
            check_space(self.root, compression='8bit')

    def test_4bit_needs_about_a_quarter_not_more_than_original(self):
        # 4-bit split ~= int(1000 * 0.2813) = 281 bytes.
        # 300 bytes of free space is plenty; the old code estimated ~3554 bytes and wrongly raised.
        self._set_free(300)
        check_space(self.root, compression='4bit')  # should NOT raise
        # below the real requirement it should still raise
        self._set_free(200)
        with self.assertRaises(NotEnoughSpaceException):
            check_space(self.root, compression='4bit')

    def test_4bit_needs_less_space_than_8bit(self):
        # For the same model, 4-bit must require less free space than 8-bit. With total=1000,
        # 4-bit needs ~281 and 8-bit needs ~500, so 300 bytes free clears 4-bit but not 8-bit.
        self._set_free(300)
        check_space(self.root, compression='4bit')  # ok
        with self.assertRaises(NotEnoughSpaceException):
            check_space(self.root, compression='8bit')


if __name__ == '__main__':
    unittest.main()
