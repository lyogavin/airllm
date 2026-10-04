"""Load the optional compression backend only when compressed weights are used."""
import importlib


def require_bitsandbytes():
    """Return bitsandbytes, preserving the cause if its native backend cannot load."""
    try:
        return importlib.import_module('bitsandbytes')
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError(
            'AirLLM compression requires a working bitsandbytes installation compatible '
            'with your PyTorch build and GPU backend (CUDA or ROCm). '
            'For uncompressed checkpoints, use compression=None without bitsandbytes. '
            'Already-compressed layer shards still require bitsandbytes.'
        ) from exc
