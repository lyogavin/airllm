"""CPU-only tests for the persister subsystem.

Exercises SafetensorModelPersister save/load/exists roundtrip using temp files.

Conventions:
- persist_model and model_persist_exist expect layer_name WITH trailing dot
  (e.g. "model.layers.0.").
- load_model expects layer_name WITHOUT trailing dot
  (e.g. "model.layers.0"), then appends ".safetensors" internally.
Both produce the same filename: model.layers.0.safetensors.
"""

import torch
from pathlib import Path


class TestSafetensorModelPersister:
    """Roundtrip save -> exists -> load via SafetensorModelPersister."""

    def test_persist_and_load_roundtrip(self, tmp_path):
        from airllm.persist.safetensor_model_persister import SafetensorModelPersister

        persister = SafetensorModelPersister()
        saving_path = Path(tmp_path)
        # persist_model uses layer_name with trailing dot
        save_layer_name = "model.layers.0."
        # load_model uses layer_name without trailing dot
        load_layer_name = "model.layers.0"

        sd = {
            "model.layers.0.weight": torch.randn(4, 4, dtype=torch.float32),
            "model.layers.0.bias": torch.randn(4, dtype=torch.float32),
        }

        # Not yet saved
        assert not persister.model_persist_exist(save_layer_name, saving_path)

        # Save
        persister.persist_model(sd, save_layer_name, saving_path)

        # Now exists
        assert persister.model_persist_exist(save_layer_name, saving_path)

        # Load and verify
        loaded = persister.load_model(load_layer_name, saving_path)
        assert set(loaded.keys()) == set(sd.keys())
        for k in sd:
            assert torch.equal(loaded[k], sd[k])

    def test_done_marker_created(self, tmp_path):
        from airllm.persist.safetensor_model_persister import SafetensorModelPersister

        persister = SafetensorModelPersister()
        saving_path = Path(tmp_path)
        layer_name = "model.layers.0."

        sd = {"model.layers.0.weight": torch.ones(2, 2)}
        persister.persist_model(sd, layer_name, saving_path)

        done_file = saving_path / (layer_name + "safetensors.done")
        assert done_file.exists()

    def test_model_persist_exist_returns_false_for_missing(self, tmp_path):
        from airllm.persist.safetensor_model_persister import SafetensorModelPersister

        persister = SafetensorModelPersister()
        assert not persister.model_persist_exist("nonexistent.layer.", Path(tmp_path))


class TestModelPersisterSingleton:
    """ModelPersister.get_model_persister() should return a consistent instance."""

    def test_returns_same_instance(self):
        from airllm.persist.model_persister import ModelPersister

        p1 = ModelPersister.get_model_persister()
        p2 = ModelPersister.get_model_persister()
        assert p1 is p2
