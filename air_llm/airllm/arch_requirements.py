"""Per-architecture runtime requirements that can be checked before the expensive split.

Some checkpoints ship remote model code that only works on one transformers major, and some force
an attention backend regardless of what is requested. Left unchecked, the failure surfaces deep
inside model construction, after the second Auto class has been tried and with an error that names
neither the real cause nor the fix (see issue #389, where Kimi K3 on transformers 5.x reported four
unrelated "Unrecognized configuration class" errors before the actual one).

The checks here are static: they only look at the checkpoint's ``config.json`` (and the installed
versions) and never import torch or transformers, so they are cheap enough to run up front and can
be tested in isolation.
"""
import json
import os

# ``model_type`` -> highest transformers major that can load the checkpoint.
#
# Kimi K3's remote code hard-forces ``flash_attention_2`` in ``KimiLinearModel.__init__`` (it
# overwrites ``config._attn_implementation`` whatever was requested) and it only declares the
# pre-5.0 ``_supports_flash_attn_2`` flag, which transformers >= 5 no longer consults when picking
# a backend. It therefore builds on 4.56.x and fails on 5.x. The weights are MXFP4, so loading them
# also needs ``compressed-tensors`` and the flash-attn kernels themselves.
_TRANSFORMERS_MAJORS = {
    "kimi_k3": 4,
}

_EXTRA_REQUIREMENTS = {
    "kimi_k3": ("flash_attn", "compressed_tensors"),
}

_SUMMARIES = {
    "kimi_k3": "KimiK3ForConditionalGeneration (Kimi K3)",
}

_HINTS = {
    "kimi_k3": "pip install \"transformers==4.56.2\" compressed-tensors flash-attn",
}


def read_config_json(config_path: str) -> dict | None:
    """Return the parsed ``config.json`` for a local checkpoint, or ``None``.

    ``config_path`` may be the directory itself or its ``config.json``. Anything unreadable or not
    shaped like a config yields ``None``; this helper must never be the reason a load fails.
    """
    config_json = (os.path.join(config_path, "config.json")
                   if os.path.isdir(config_path) else config_path)
    try:
        with open(config_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _checkpoint_keys(config):
    """All ``model_type``/architecture signals in the config, nested sub-configs included.

    Kimi K3 keeps ``model_type: kimi_k3`` at the top level, but other wrappers only declare it on
    ``text_config``, so walk the dict rather than trusting the top level alone.
    """
    keys = set()
    stack = [config]
    seen = 0
    while stack and seen < 64:
        node = stack.pop()
        seen += 1
        if not isinstance(node, dict):
            continue
        model_type = node.get("model_type")
        if isinstance(model_type, str):
            keys.add(model_type.lower())
        archs = node.get("architectures")
        if isinstance(archs, list):
            for arch in archs:
                if isinstance(arch, str):
                    keys.add(arch.lower())
        for value in node.values():
            if isinstance(value, dict):
                stack.append(value)
    return keys


def _major(version):
    try:
        return int(str(version).split(".")[0])
    except (TypeError, ValueError):
        return None


def transformers_major():
    """Major version of the installed transformers, or ``None`` when it cannot be determined."""
    try:
        import transformers
    except ImportError:
        return None
    return _major(getattr(transformers, "__version__", None))


def _is_installed(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def unsupported_reason(keys, version=None, is_installed=None):
    """Return a one-line, actionable message when transformers ``version`` cannot load ``keys``.

    ``keys`` is the ``model_type`` / architecture set from ``_checkpoint_keys``. ``version`` is the
    transformers version string (the installed one when omitted); an undeterminable version is
    never guessed at.
    """
    for model_type, required_major in _TRANSFORMERS_MAJORS.items():
        if model_type not in keys:
            continue
        major = _major(version) if version is not None else transformers_major()
        if major is None or major <= required_major:
            return None
        summary = _SUMMARIES[model_type]
        installed = version if version is not None else f"{major}.x"
        return (
            f"{summary} needs transformers {required_major}.56.x, but transformers {installed} "
            f"is installed. Its remote model code hard-codes flash attention using the pre-5.0 "
            f"`_supports_flash_attn_2` flag, which transformers {required_major + 1} no longer "
            f"checks, so every Auto class fails while building the model on meta "
            f"(https://github.com/lyogavin/airllm/issues/389). "
            f"Fix: {_HINTS[model_type]}"
        )
    return None


def missing_requirements(keys, is_installed=None):
    """Modules the checkpoint needs beyond the base install (e.g. Kimi K3's MXFP4 stack)."""
    is_installed = is_installed if is_installed is None else is_installed
    missing = []
    for model_type, modules in _EXTRA_REQUIREMENTS.items():
        if model_type not in keys:
            continue
        missing.extend(mod for mod in modules if not is_installed(mod))
    return missing
