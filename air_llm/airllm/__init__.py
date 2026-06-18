from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

def _optional_import(name, attr):
    # Family modules pull in family-specific deps (sentencepiece for
    # Baichuan, tiktoken for Qwen, etc.) at import time. A missing
    # optional dep should disable that single family, not break the
    # whole `import airllm`. auto_model.py handles a missing family
    # at lookup time.
    try:
        module = __import__(f"airllm.{name}", fromlist=[attr])
        globals()[attr] = getattr(module, attr)
    except ImportError as e:
        import warnings
        warnings.warn(f"airllm: {attr} unavailable ({e}); install its extras to enable.")


if is_on_mac_os:
    _optional_import("airllm_llama_mlx", "AirLLMLlamaMlx")
    from .auto_model import AutoModel
else:
    _optional_import("airllm", "AirLLMLlama2")
    _optional_import("airllm_chatglm", "AirLLMChatGLM")
    _optional_import("airllm_qwen", "AirLLMQWen")
    _optional_import("airllm_qwen2", "AirLLMQWen2")
    _optional_import("airllm_baichuan", "AirLLMBaichuan")
    _optional_import("airllm_internlm", "AirLLMInternLM")
    _optional_import("airllm_mistral", "AirLLMMistral")
    _optional_import("airllm_mixtral", "AirLLMMixtral")
    from .airllm_base import AirLLMBaseModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException

