# AMD GPUs with ROCm

AirLLM uses ROCm through PyTorch's `torch.cuda` API. Use `device="cuda:0"`
on AMD GPUs. Uncompressed inference does not require bitsandbytes.

## Installation

Install a compatible AMD driver and ROCm-enabled PyTorch build for your GPU
and OS using the [AMD ROCm installation guide](https://rocm.docs.amd.com/en/latest/install/rocm.html)
and [PyTorch guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html),
or [TheRock's release instructions](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).
WSL also requires the [ROCDXG bridge](https://rocm.docs.amd.com/en/latest/install/rocm.html#install-rocdxg-and-amd-smi-for-wsl).

Install AirLLM into the same environment from a checkout:

```bash
python -m pip install -e ./air_llm
```

Use this package installation rather than the repository-root `requirements.txt`,
which contains legacy notebook dependencies. Verify the
ROCm PyTorch build and GPU access after installation:

```python
import torch
assert torch.version.hip, "A ROCm PyTorch build is required"
assert torch.cuda.is_available(), "PyTorch cannot access the GPU"
x = torch.ones((16, 16), device="cuda:0")
assert torch.isfinite(x @ x).all()
torch.cuda.synchronize()
```

## Inference and compression

```python
import torch
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "/path/to/uncompressed-checkpoint",
    device="cuda:0",
    dtype=torch.bfloat16,
    compression=None,
)
```

For `compression="4bit"` (NF4) or `compression="8bit"`, install a bitsandbytes
build with working kernels for your PyTorch/GPU backend. Existing compressed
shards also require bitsandbytes. AirLLM decompresses weights before layer
execution, and compression disables prefetching.

New 8-bit shards preserve the source dtype. Legacy shards remain readable with
an FP16 fallback, which can lose BF16 range. To preserve the original dtype,
**delete and re-split 8bit caches**: remove the model's `splitted_model.8bit`
directory and reload the original checkpoint with `compression="8bit"`.
Alternatively, use a fresh `layer_shards_saving_path`. Re-splitting requires the
original checkpoint weights; missing dtype metadata cannot be recovered from
old caches. NF4 already records its dtype.

## Tests

Run the CPU suite from the checkout:

```bash
python -m unittest discover -v
```

GPU tests are opt-in and require a working CUDA/ROCm GPU and bitsandbytes backend.
They check streaming parity against a resident model, compression round trips,
BF16 range, and legacy 8-bit compatibility without downloading pretrained models:

```bash
AIRLLM_GPU_TESTS=1 python -m unittest discover -v
```

In PowerShell:

```powershell
$env:AIRLLM_GPU_TESTS = "1"
python -m unittest discover -v
```
