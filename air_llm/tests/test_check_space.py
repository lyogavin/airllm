import shutil
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from airllm.utils import NotEnoughSpaceException, check_space

_Usage = namedtuple("usage", "total used free")


class TestCheckSpace(unittest.TestCase):
    def _run(self, free, compression):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "model.safetensors").write_bytes(b"\0" * 1_000_000)  # 1 MB "checkpoint"
            with mock.patch.object(shutil, "disk_usage", return_value=_Usage(10**12, 0, free)):
                check_space(Path(d), compression=compression)

    def test_4bit_needs_about_a_quarter_of_the_checkpoint(self):
        self._run(free=400_000, compression="4bit")  # ~0.28 MB of NF4 shards fit
        with self.assertRaises(NotEnoughSpaceException):
            self._run(free=200_000, compression="4bit")

    def test_uncompressed_needs_the_full_checkpoint(self):
        self._run(free=1_100_000, compression=None)
        with self.assertRaises(NotEnoughSpaceException):
            self._run(free=900_000, compression=None)


if __name__ == "__main__":
    unittest.main()
