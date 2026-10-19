"""Lightweight unit tests for the CLI / chat REPL features.

These tests run without a GPU, without downloading any model, and without
importing heavy dependencies like torch.  They exercise:

  * /clear and /system slash-commands
  * Model-alias resolution
  * `airllm --help` smoke-test
  * Safety of `remove_model_cache()` (rejects arbitrary paths)
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Alias resolution (models.py)
# ---------------------------------------------------------------------------

from airllm.models import resolve_model_name, MODEL_ALIASES


class TestAliasResolution:
    """resolve_model_name should map known aliases and pass unknowns through."""

    def test_known_alias(self):
        assert resolve_model_name("llama3:70b") == "meta-llama/Meta-Llama-3-70B-Instruct"

    def test_known_alias_case_insensitive(self):
        assert resolve_model_name("LLAMA3:70B") == "meta-llama/Meta-Llama-3-70B-Instruct"

    def test_unknown_passes_through(self):
        assert resolve_model_name("my-org/my-custom-model") == "my-org/my-custom-model"

    def test_whitespace_stripped(self):
        assert resolve_model_name("  deepseek-r1:8b  ") == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    def test_all_aliases_resolve_to_nonempty_string(self):
        for alias, repo in MODEL_ALIASES.items():
            resolved = resolve_model_name(alias)
            assert resolved == repo, f"Alias '{alias}' resolved to '{resolved}', expected '{repo}'"


# ---------------------------------------------------------------------------
# /clear  and  /system  slash-commands (chat.py)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal stand-in so InteractiveChatSession can be constructed."""
    chat_template = None

    def apply_chat_template(self, *a, **kw):
        raise NotImplementedError

    def encode(self, text, **kw):
        return text.split()

    def __call__(self, *a, **kw):
        raise NotImplementedError


class _FakeModel:
    """Minimal stand-in — no real inference needed."""
    tokenizer = _FakeTokenizer()
    max_seq_len = 128

    def generate(self, **kw):
        raise NotImplementedError


# Import after fakes so we can construct sessions without torch/GPU
from airllm.chat import InteractiveChatSession


class TestSlashClear:
    def test_clear_resets_to_system_only(self):
        session = InteractiveChatSession(
            model=_FakeModel(),
            model_name="test",
            system_prompt="Be concise.",
        )
        # Simulate a user turn
        session.messages.append({"role": "user", "content": "hello"})
        session.messages.append({"role": "assistant", "content": "hi"})
        assert len(session.messages) == 3  # system + user + assistant

        result = session.step("/clear")

        assert result is True  # session continues
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "system"
        assert session.messages[0]["content"] == "Be concise."

    def test_clear_with_no_system_prompt(self):
        session = InteractiveChatSession(
            model=_FakeModel(),
            model_name="test",
            system_prompt="",
        )
        session.messages.append({"role": "user", "content": "hello"})

        session.step("/clear")

        assert session.messages == []


class TestSlashSystem:
    def test_system_updates_prompt_and_resets(self):
        session = InteractiveChatSession(
            model=_FakeModel(),
            model_name="test",
            system_prompt="Old prompt.",
        )
        session.messages.append({"role": "user", "content": "hi"})

        result = session.step("/system Be brief and precise.")

        assert result is True
        assert session.system_prompt == "Be brief and precise."
        # History should be reset to just the new system prompt
        assert len(session.messages) == 1
        assert session.messages[0]["content"] == "Be brief and precise."

    def test_system_without_arg_shows_current(self, capsys):
        session = InteractiveChatSession(
            model=_FakeModel(),
            model_name="test",
            system_prompt="Current prompt.",
        )

        session.step("/system")

        captured = capsys.readouterr()
        assert "Current prompt." in captured.out


# ---------------------------------------------------------------------------
# airllm --help  smoke test (cli.py)
# ---------------------------------------------------------------------------

from airllm.cli import build_parser


class TestCliHelp:
    def test_help_exits_zero(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_subcommand_help_exits_zero(self):
        parser = build_parser()
        for sub in ("run", "pull", "list", "show", "rm", "aliases"):
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([sub, "--help"])
            assert exc_info.value.code == 0, f"{sub} --help exited with {exc_info.value.code}"


# ---------------------------------------------------------------------------
# remove_model_cache safety (models.py)
# ---------------------------------------------------------------------------

from airllm.models import remove_model_cache, _is_safe_cache_path, get_hf_hub_cache_dir


class TestRemoveModelCacheSafety:
    """Ensure remove_model_cache never deletes outside the HF hub cache."""

    def test_arbitrary_path_not_deleted(self, tmp_path):
        """Passing an arbitrary filesystem path must NOT delete it.

        The old code had a fallback that would shutil.rmtree() any path that
        existed as a directory.  After the fix, the function only looks for a
        ``models--`` prefixed folder inside the HF cache, so a raw path like
        ``/tmp`` or ``C:\\Windows`` is never touched.
        """
        victim = tmp_path / "should_survive"
        victim.mkdir()
        (victim / "important_file.txt").write_text("do not delete")

        # remove_model_cache should not delete the victim directory
        result = remove_model_cache(str(victim))

        assert result is False
        assert victim.exists(), "Arbitrary directory was deleted — safety regression!"
        assert (victim / "important_file.txt").read_text() == "do not delete"

    def test_safe_path_inside_cache(self, tmp_path):
        """A properly-formed models-- dir inside the cache should be accepted."""
        fake_hub = tmp_path / "hub"
        fake_hub.mkdir()
        model_dir = fake_hub / "models--test-org--test-model"
        model_dir.mkdir()

        with mock.patch("airllm.models.get_hf_hub_cache_dir", return_value=fake_hub):
            result = remove_model_cache("test-org/test-model")

        assert result is True
        assert not model_dir.exists()

    def test_nonexistent_model_returns_false(self, tmp_path):
        """Deleting a model that isn't cached should return False, not crash."""
        fake_hub = tmp_path / "hub"
        fake_hub.mkdir()

        with mock.patch("airllm.models.get_hf_hub_cache_dir", return_value=fake_hub):
            result = remove_model_cache("nonexistent-org/nonexistent-model")

        assert result is False

    def test_is_safe_rejects_cache_root_itself(self, tmp_path):
        """Deleting the cache root itself must not be allowed."""
        fake_hub = tmp_path / "hub"
        fake_hub.mkdir()

        with mock.patch("airllm.models.get_hf_hub_cache_dir", return_value=fake_hub):
            assert _is_safe_cache_path(fake_hub) is False
