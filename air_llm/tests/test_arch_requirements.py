"""Tests for the up-front architecture requirements check (issue #389).

Kimi K3's remote code only builds on transformers 4.56.x, but a user on 5.x used to see nothing
but "Could not instantiate the model on meta from config" plus a pile of unrelated Auto-class
errors. These tests pin the detection logic and the injection points, with no torch, transformers
or checkpoint download involved.
"""

import json
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / 'airllm'

# Bind the package without running airllm/__init__.py: that pulls in the whole torch/transformers
# stack, which neither module under test needs.
if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = _pkg

from airllm.arch_requirements import (_checkpoint_keys, missing_requirements,  # noqa: E402
                                      read_config_json, unsupported_reason)
from airllm.auto_model import AutoModel  # noqa: E402

# The relevant part of moonshotai/Kimi-K3's config.json.
K3_CONFIG = {
    "architectures": ["KimiK3ForConditionalGeneration"],
    "auto_map": {
        "AutoConfig": "configuration_kimi_k3.KimiK3Config",
        "AutoModel": "modeling_kimi_k3.KimiK3ForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_kimi_k3.KimiK3ForConditionalGeneration",
    },
    "model_type": "kimi_k3",
    "text_config": {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
    },
    "vision_config": {"_attn_implementation": "flash_attention_2"},
    "transformers_version": "4.56.2",
}

LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
}


def _installed(*available):
    """Pretend only ``available`` modules are importable."""
    return lambda module: module in available


def _write_checkpoint(root, config):
    root = Path(root)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


class _TempDir:
    """A temporary directory that lives next to this test file.

    ``tempfile.TemporaryDirectory`` ends up under ``$TMPDIR``, which is outside the checkout in
    sandboxed CI environments; then the config writes are denied. A scratch dir beside the test
    keeps everything inside the workspace. No cleanup try/except: on Windows a denied delete
    raises inside ``__exit__`` and would mask the actual failure.
    """

    def __enter__(self):
        self.path = Path(__file__).resolve().parent / f'.tmp_{uuid.uuid4().hex}'
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, exc_type, exc, tb):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


class TestCheckpointKeys(unittest.TestCase):
    def test_nested_model_types_are_collected(self):
        keys = _checkpoint_keys(K3_CONFIG)
        self.assertIn('kimi_k3', keys)
        self.assertIn('kimi_linear', keys)
        self.assertIn('kimik3forconditionalgeneration', keys)

    def test_empty_or_malformed_input(self):
        self.assertEqual(_checkpoint_keys({}), set())
        self.assertEqual(_checkpoint_keys(None), set())


class TestUnsupportedReason(unittest.TestCase):
    def test_k3_on_transformers_5_is_rejected_with_the_fix(self):
        reason = unsupported_reason({'kimi_k3'}, version='5.22.2')
        self.assertIsNotNone(reason)
        self.assertIn('5.22.2', reason)
        self.assertIn('transformers==4.56.2', reason)
        self.assertIn('issues/389', reason)

    def test_k3_on_transformers_4_is_accepted(self):
        self.assertIsNone(unsupported_reason({'kimi_k3'}, version='4.56.2'))

    def test_undeterminable_version_is_not_guessed(self):
        self.assertIsNone(unsupported_reason({'kimi_k3'}, version='not-a-version'))

    def test_unrelated_architectures_are_never_affected(self):
        for version in ('4.56.2', '5.0.0', '5.22.2'):
            self.assertIsNone(
                unsupported_reason({'llama', 'llamaforcausalmodel'}, version=version))


class TestMissingRequirements(unittest.TestCase):
    def test_k3_lists_the_mxfp4_stack(self):
        self.assertEqual(
            missing_requirements({'kimi_k3'}, is_installed=_installed()),
            ['flash_attn', 'compressed_tensors'])

    def test_nothing_missing_when_installed(self):
        self.assertEqual(
            missing_requirements({'kimi_k3'},
                                 is_installed=_installed('flash_attn', 'compressed_tensors')),
            [])

    def test_other_models_have_no_extras(self):
        self.assertEqual(missing_requirements({'llama'}, is_installed=_installed()), [])


class TestReadConfigJson(unittest.TestCase):
    def test_reads_directory_and_file(self):
        with _TempDir() as tmp:
            root = _write_checkpoint(tmp, K3_CONFIG)
            self.assertEqual(read_config_json(root), K3_CONFIG)
            self.assertEqual(read_config_json(str(root / 'config.json')), K3_CONFIG)

    def test_missing_or_broken_config_is_none(self):
        self.assertIsNone(read_config_json('/nonexistent/checkpoint'))
        with _TempDir() as tmp:
            (Path(tmp) / 'config.json').write_text('{not json', encoding='utf-8')
            self.assertIsNone(read_config_json(tmp))


class TestAutoModelEntryPoint(unittest.TestCase):
    """The check must fire before the (expensive) split, and go quiet for everything else."""

    def _patch_major(self, major):
        import airllm.arch_requirements as arch

        original = arch.transformers_major
        arch.transformers_major = lambda: major

        def restore():
            arch.transformers_major = original

        self.addCleanup(restore)

    def test_k3_on_transformers_5_fails_before_loading(self):
        self._patch_major(5)
        with _TempDir() as tmp:
            root = _write_checkpoint(tmp, K3_CONFIG)
            with self.assertRaises(RuntimeError) as ctx:
                AutoModel.from_pretrained(str(root))
        self.assertIn('transformers==4.56.2', str(ctx.exception))

    def test_k3_on_transformers_4_is_not_rejected(self):
        self._patch_major(4)
        with _TempDir() as tmp:
            root = _write_checkpoint(tmp, K3_CONFIG)
            self._assert_past_the_check(root)

    def test_non_k3_checkpoint_is_untouched(self):
        self._patch_major(5)
        with _TempDir() as tmp:
            root = _write_checkpoint(tmp, LLAMA_CONFIG)
            self._assert_past_the_check(root)

    def _assert_past_the_check(self, root):
        """Call the loader and return only if the requirement check let the checkpoint through."""
        try:
            AutoModel.from_pretrained(str(root))
        except RuntimeError as e:
            if 'transformers==' in str(e):
                self.fail(f"checkpoint was rejected by the requirements check: {e}")
            # anything else is the loader failing later on this fake checkpoint (no weights)
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
