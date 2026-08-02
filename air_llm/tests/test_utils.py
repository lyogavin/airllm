"""CPU-only, network-free tests for airllm utility functions.

All tests operate on tiny synthetic tensors or temp files — no model download
and no GPU required.
"""

import os
import torch
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# compress_layer_state_dict / uncompress_layer_state_dict (None compression)
# ---------------------------------------------------------------------------

class TestCompressNoneCompression:
    """When compression=None the identity path should be exercised."""

    def test_identity_roundtrip(self):
        from airllm.utils import compress_layer_state_dict, uncompress_layer_state_dict

        sd = {
            "w": torch.randn(4, 4, dtype=torch.float32),
            "b": torch.randn(4, dtype=torch.float32),
        }
        out = compress_layer_state_dict(sd, compression=None)
        assert out is sd  # same object, no copy

    def test_none_compression_returns_original_keys(self):
        from airllm.utils import compress_layer_state_dict

        sd = {"a": torch.zeros(2, 2), "b": torch.ones(3)}
        out = compress_layer_state_dict(sd, compression=None)
        assert set(out.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# layer_tensor_names / load_layer_subset
#
# NOTE: these functions receive layer_name WITHOUT trailing dot, then append
# ".safetensors" internally.  e.g. "model.layers.0" -> "model.layers.0.safetensors"
# ---------------------------------------------------------------------------

class TestSafetensorIO:
    """Write a tiny safetensors shard and read it back via the helpers."""

    def test_layer_tensor_names(self, tmp_layer_shard):
        from airllm.utils import layer_tensor_names

        # layer_name does NOT include trailing dot — the function appends ".safetensors"
        names = layer_tensor_names(str(tmp_layer_shard), "model.layers.0")
        assert "model.layers.0.self_attn.weight" in names
        assert "model.layers.0.self_attn.bias" in names

    def test_load_layer_subset(self, tmp_layer_shard):
        from airllm.utils import load_layer_subset

        subset = load_layer_subset(
            str(tmp_layer_shard),
            "model.layers.0",
            ["model.layers.0.self_attn.weight"],
        )
        assert set(subset.keys()) == {"model.layers.0.self_attn.weight"}
        assert subset["model.layers.0.self_attn.weight"].shape == (4, 8)


# ---------------------------------------------------------------------------
# link_or_copy_file
# ---------------------------------------------------------------------------

class TestLinkOrCopyFile:
    def test_hardlink_preferred(self, tmp_path):
        from airllm.utils import link_or_copy_file

        src = tmp_path / "src.bin"
        src.write_bytes(b"hello")
        dst = tmp_path / "dst.bin"

        method = link_or_copy_file(src, dst)
        assert dst.read_bytes() == b"hello"
        # hardlink or symlink depending on fs
        assert method in ("hardlink", "symlink", "copy")

    def test_overwrites_existing(self, tmp_path):
        from airllm.utils import link_or_copy_file

        src = tmp_path / "src.bin"
        src.write_bytes(b"new")
        dst = tmp_path / "dst.bin"
        dst.write_bytes(b"old")

        link_or_copy_file(src, dst)
        assert dst.read_bytes() == b"new"


# ---------------------------------------------------------------------------
# split_and_save_layers (offline, tiny fixture)
# ---------------------------------------------------------------------------

class TestSplitAndSaveLayers:
    """split_and_save_layers with a tiny synthetic checkpoint (no network)."""

    def test_split_creates_layer_files(self, tmp_safetensors):
        from airllm.utils import split_and_save_layers

        result = split_and_save_layers(checkpoint_path=tmp_safetensors)
        assert os.path.exists(result)
        # Should contain the layer shard + done marker
        files = os.listdir(result)
        safetensor_files = [f for f in files if f.endswith("safetensors")]
        done_markers = [f for f in files if f.endswith(".done")]
        assert len(safetensor_files) >= 1
        assert len(done_markers) >= 1

    def test_split_idempotent(self, tmp_safetensors):
        """Running split twice should return the same path without error."""
        from airllm.utils import split_and_save_layers

        r1 = split_and_save_layers(checkpoint_path=tmp_safetensors)
        r2 = split_and_save_layers(checkpoint_path=tmp_safetensors)
        assert r1 == r2


# ---------------------------------------------------------------------------
# NotEnoughSpaceException
# ---------------------------------------------------------------------------

class TestNotEnoughSpaceException:
    def test_is_exception(self):
        from airllm.utils import NotEnoughSpaceException
        e = NotEnoughSpaceException("disk full")
        assert str(e) == "disk full"
        assert isinstance(e, Exception)
