"""
Device abstraction utilities for AirLLM.

Adds support for:
  - NVIDIA CUDA  (device="cuda:0")
  - Intel / AMD integrated GPU via DirectML on Windows  (device="privateuseone:0")
  - Apple Silicon via MPS  (device="mps")   [MLX path handles this separately]
  - CPU fallback  (device="cpu")
"""

import torch


# ---------------------------------------------------------------------------
# DirectML detection
# ---------------------------------------------------------------------------

try:
    import torch_directml  # pip install torch-directml

    _directml_available = True
except ImportError:
    _directml_available = False


def is_directml_available() -> bool:
    return _directml_available


def get_directml_device(index: int = 0):
    """Return a torch-directml device handle, or None if not available."""
    if _directml_available:
        import torch_directml
        return torch_directml.device(index)
    return None


# ---------------------------------------------------------------------------
# Device type helpers
# ---------------------------------------------------------------------------

def get_device_type(device: str) -> str:
    """
    Normalise a device string to one of:
      "cuda" | "directml" | "mps" | "cpu"
    """
    d = str(device).lower()
    if d.startswith("cuda"):
        return "cuda"
    # torch-directml registers as "privateuseone" internally
    if d.startswith("privateuseone") or d.startswith("dml") or d.startswith("directml"):
        return "directml"
    if d.startswith("mps"):
        return "mps"
    return "cpu"


def is_cuda_device(device: str) -> bool:
    return get_device_type(device) == "cuda"


def is_directml_device(device: str) -> bool:
    return get_device_type(device) == "directml"


def can_pin_memory(device: str) -> bool:
    """
    pin_memory() is only meaningful when copying to CUDA.
    It's a no-op (and sometimes errors) for DirectML / MPS / CPU targets.
    """
    return is_cuda_device(device)


# ---------------------------------------------------------------------------
# Device-agnostic cache clearing
# ---------------------------------------------------------------------------

def empty_cache(device: str) -> None:
    """Free unused memory on the given device."""
    dtype = get_device_type(device)
    if dtype == "cuda":
        torch.cuda.empty_cache()
    elif dtype == "mps":
        # torch.mps.empty_cache() is available in PyTorch >= 2.0
        # but calling it on a non-Mac machine raises RuntimeError
        if hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass
    # DirectML and CPU: nothing to do


# ---------------------------------------------------------------------------
# Device-agnostic free-memory query
# ---------------------------------------------------------------------------

def get_free_memory_bytes(device: str) -> int:
    """
    Return free device memory in bytes, or -1 if unavailable.
    """
    dtype = get_device_type(device)
    if dtype == "cuda":
        try:
            free, _ = torch.cuda.mem_get_info()
            return free
        except Exception:
            return -1
    # MPS / DirectML / CPU: no reliable API yet
    return -1


# ---------------------------------------------------------------------------
# Compression support check
# ---------------------------------------------------------------------------

def supports_bitsandbytes(device: str) -> bool:
    """
    bitsandbytes only works on NVIDIA CUDA devices.
    Integrated GPUs (DirectML / MPS) must skip compression.
    """
    return is_cuda_device(device)
