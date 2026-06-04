"""Architecture detection and dispatch for AirLLM model wrappers.

The :class:`AutoModel` factory picks the right per-architecture wrapper class
(``AirLLMLlama2``, ``AirLLMQWen2``, ...) based on the model config returned by
Hugging Face Transformers.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Tuple

from transformers import AutoConfig

from .utils import is_on_mac_os

if is_on_mac_os:
    from .airllm_llama_mlx import AirLLMLlamaMlx  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)

# Mapping of substring patterns found in ``config.architectures[0]`` to the
# airllm class that should handle them. Order matters: the first match wins,
# so put the most specific patterns first (e.g. "Qwen2ForCausalLM" before "QWen").
_ARCHITECTURE_DISPATCH: Tuple[Tuple[str, str], ...] = (
    ("Qwen2ForCausalLM", "AirLLMQWen2"),
    ("QWen",            "AirLLMQWen"),
    ("Baichuan",        "AirLLMBaichuan"),
    ("ChatGLM",         "AirLLMChatGLM"),
    ("InternLM",        "AirLLMInternLM"),
    ("Mixtral",         "AirLLMMixtral"),
    ("Mistral",         "AirLLMMistral"),
    ("Llama",           "AirLLMLlama2"),
)


class UnknownArchitectureError(ValueError):
    """Raised when the model's architecture is not supported by AirLLM."""


class AutoModel:
    """Factory that returns a per-architecture AirLLM model wrapper.

    Always use :meth:`AutoModel.from_pretrained` — direct instantiation is
    disabled.
    """

    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @staticmethod
    def _load_config(pretrained_model_name_or_path: str, **kwargs) -> object:
        token = kwargs.get("hf_token")
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            token=token,
        )

    @classmethod
    def get_module_class(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ) -> Tuple[str, str]:
        """Return ``(module_name, class_name)`` for the model at ``pretrained_model_name_or_path``."""
        config = cls._load_config(pretrained_model_name_or_path, **kwargs)

        architecture = (config.architectures or [""])[0]
        if not architecture:
            raise UnknownArchitectureError(
                f"Could not determine model architecture: 'architectures' "
                f"is empty in the config of {pretrained_model_name_or_path!r}."
            )

        for needle, class_name in _ARCHITECTURE_DISPATCH:
            if needle in architecture:
                logger.debug(
                    "Detected architecture %r -> %s", architecture, class_name
                )
                return "airllm", class_name

        supported = ", ".join(name for _, name in _ARCHITECTURE_DISPATCH)
        raise UnknownArchitectureError(
            f"Unsupported architecture {architecture!r} for "
            f"{pretrained_model_name_or_path!r}. "
            f"Supported architectures: {supported}. "
            f"Open an issue at https://github.com/lyogavin/airllm/issues "
            f"if you need support for this model."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *args, **kwargs)

        module_name, class_name = cls.get_module_class(
            pretrained_model_name_or_path, *args, **kwargs
        )
        module = importlib.import_module(module_name)
        wrapper = getattr(module, class_name)
        return wrapper(pretrained_model_name_or_path, *args, **kwargs)
