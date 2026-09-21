"""Shared fixtures for CPU-only, network-free tests."""

import pytest


@pytest.fixture
def tmp_safetensors(tmp_path):
    """Create a tiny single-layer safetensors shard on disk and return its path.

    The file contains two float32 tensors:
      - model.layers.0.self_attn.weight  (4, 8)
      - model.layers.0.self_attn.bias    (8,)
    plus a model.safetensors.index.json weight-map so ``split_and_save_layers``
    can exercise its normal code path without downloading anything.
    """
    import json
    import torch
    from safetensors.torch import save_file

    sd = {
        "model.layers.0.self_attn.weight": torch.randn(4, 8, dtype=torch.float32),
        "model.layers.0.self_attn.bias": torch.randn(8, dtype=torch.float32),
    }
    save_file(sd, tmp_path / "model.safetensors")

    weight_map = {k: "model.safetensors" for k in sd}
    index = {"metadata": {"total_size": sum(v.numel() * 4 for v in sd.values())},
             "weight_map": weight_map}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return tmp_path


@pytest.fixture
def tmp_layer_shard(tmp_path):
    """Create a tiny per-layer safetensors shard (post-split) and return its path.

    After split_and_save_layers runs, layers are saved with trailing-dot naming:
    ``model.layers.0.safetensors`` (from layer_name ``model.layers.0.`` + ``safetensors``).

    The loading functions (layer_tensor_names, load_layer_subset, load_model)
    receive layer_name WITHOUT trailing dot and append ``.safetensors``.
    Both conventions produce the same filename: ``model.layers.0.safetensors``.

    This fixture creates that file directly.
    """
    import torch
    from safetensors.torch import save_file

    sd = {
        "model.layers.0.self_attn.weight": torch.randn(4, 8, dtype=torch.float32),
        "model.layers.0.self_attn.bias": torch.randn(8, dtype=torch.float32),
    }
    save_file(sd, tmp_path / "model.layers.0.safetensors")
    return tmp_path
