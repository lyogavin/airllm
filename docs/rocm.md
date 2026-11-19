# AMD GPUs with ROCm

AirLLM uses ROCm through PyTorch's `torch.cuda` API. Use `device="cuda:0"`
on AMD GPUs; `rocm` and `hip` are not PyTorch device types. Uncompressed
inference does not require bitsandbytes.

## Installation

Install the AMD driver and ROCm release for your GPU and OS using the
[official AMD ROCm installation page](https://rocm.docs.amd.com/en/latest/install/rocm.html),
then install ROCm-enabled PyTorch using the
[AMD PyTorch guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
Alternatively, follow [TheRock's release instructions](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
for ROCm/PyTorch builds. Verify GPU access in that environment:

```python
import torch
assert torch.version.hip, "A ROCm PyTorch build is required"
assert torch.cuda.is_available(), "PyTorch cannot access the GPU"
x = torch.ones((16, 16), device="cuda:0")
assert torch.isfinite(x @ x).all()
torch.cuda.synchronize()
print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))
```

From an AirLLM checkout, install into the same environment:

```bash
python -m pip install -e ./air_llm
```

Keep the working ROCm PyTorch installation and recheck `torch.version.hip`
after installation. The repository-root `requirements.txt` contains legacy
notebook dependencies, rather than the package's installation requirements.

WSL also needs a compatible Windows AMD driver and ROCDXG bridge; follow
[AMD's ROCDXG setup instructions](https://rocm.docs.amd.com/en/latest/install/rocm.html#install-rocdxg-and-amd-smi-for-wsl).
Missing `librocdxg.so` or `hsaKmtOpenKFD` errors occur before AirLLM inference.
A process-local `LD_LIBRARY_PATH` can expose an existing bridge installation
without changing system loader configuration.

## Inference and compression

```python
import torch
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "/path/to/uncompressed-checkpoint",
    device="cuda:0",
    dtype=torch.bfloat16,
    compression=None,
    prefetching=True,
)
```

AirLLM loads bitsandbytes when compression is requested or compressed shards
are loaded.
For `compression="4bit"` (NF4) or `compression="8bit"` (blockwise 8-bit),
install a build with working kernels for your PyTorch/GPU backend:

```bash
python -m pip install bitsandbytes==0.50.2
```

Existing compressed shards still require bitsandbytes, even if the current call
does not request new compression. A successful import does not prove that the
GPU kernels work. AirLLM decompresses shards into floating-point weights before
layer execution; these paths do not use `Linear4bit` or `LLM.int8` GEMM kernels.
Compression disables AirLLM prefetching.

New 8-bit shards preserve the source dtype. Legacy shards remain readable with
the existing FP16 fallback, matching their previous behavior. To preserve the
original dtype, **delete and re-split 8bit caches**: remove the model's existing
`splitted_model.8bit` directory, then load the original checkpoint again with
`compression="8bit"`. Alternatively, use a fresh `layer_shards_saving_path` to
keep the old cache while generating new shards. Re-splitting requires the
original checkpoint weights; restore them first if they were deleted. Missing
dtype metadata cannot be recovered from an old cached shard, so loading a legacy
BF16 cache still reconstructs FP16 weights and can lose BF16 range. NF4 already
records its dtype.

## Strix Halo validation

The original focused regression suite was validated on AMD Strix Halo, Radeon 8060S
(`gfx1151`), Ubuntu 24.04 on WSL2, with:

| Component | Tested version |
| --- | --- |
| PyTorch | 2.13.0+rocm10.0.0 |
| ROCm SDK | 10.0.0 |
| HIP component (`torch.version.hip`) | 7.15.26333 |
| ROCDXG | 1.2.2 |
| Transformers / Accelerate | 4.57.6 / 1.15.0 |
| bitsandbytes | 0.50.2, native `libbitsandbytes_rocm715.so` |

The PyTorch wheel was `torch[device-gfx1151]==2.13.0+rocm10.0.0` from
`https://stable.repo.amd.com/rocm/whl-next/`. The standalone environment used
ROCm SDK core/libraries/device packages at 10.0.0 and a process-local ROCDXG
library path. HIP's component version is distinct from the ROCm distribution
version. No custom bitsandbytes build or GPU-architecture override was used.

The tests now use Python's built-in unittest runner. From the checkout, run the
optional-backend regression tests without enabling GPU work:

```bash
python -m unittest discover -s air_llm/tests -p 'test_optional_compression.py' -v
```

The CPU end-to-end suite runs by default under unittest discovery on Linux and
Windows, where AirLLM uses its PyTorch backend. It creates a tiny local Llama
checkpoint and verifies streamed logits, repeated generation, and parameter
eviction with prefetch disabled and enabled. It blocks bitsandbytes imports
throughout AirLLM loading and inference, and requires no model downloads:

```bash
python -m unittest discover -s air_llm/tests -p 'test_streaming_cpu.py' -v
```

macOS skips this PyTorch CPU suite because its `AutoModel` entry point uses MLX.

In a configured CUDA or ROCm environment, explicitly opt in to the GPU suites:

```bash
AIRLLM_GPU_TESTS=1 python -m unittest discover -s air_llm/tests -p 'test_streaming_parity.py' -v
AIRLLM_GPU_TESTS=1 python -m unittest discover -s air_llm/tests -p 'test_compression_backend.py' -v
```

For the original validation recorded above, all 15 cases passed. The locally
generated small Llama checkpoint checks FP16
and BF16 streaming against resident Transformers, prefetch off/on, cached and
repeated generation, parameter eviction to meta, and retained allocations.
The optional-backend cases exercise missing/broken native imports. Seven GPU
compression cases check real NF4/8-bit kernels, persisted shard reconstruction,
numerical error, BF16 values outside FP16 range, and legacy 8-bit compatibility.
These tests require no pretrained model downloads.

The same original 15 cases also passed on an NVIDIA GeForce RTX 4070 Laptop GPU
(8 GB), Ubuntu 24.04, with driver 580.126.09, PyTorch 2.14.1+cu132,
CUDA 13.2, Transformers 4.57.6, Accelerate 1.15.0, and bitsandbytes 0.50.2
using native `libbitsandbytes_cuda132.so`. There were no skips, and `pip check`
passed. Both FP16 and BF16 generated tokens matched the resident reference.

### Qwen3.8-27B text inference

An additional text inference check used `Qwen/Qwen3.8-27B` in BF16 on the
same Strix Halo / ROCm 10.0 system, with Transformers 5.18.0, SDPA,
`MIOPEN_FIND_MODE=FAST`, `compression=None`, and prefetch enabled.
Greedy generation answered `Paris` to the capital-of-France prompt and
matched on repetition. Tokens and checked prefill/cached-step logits matched
the previously recorded native Transformers + Accelerate offload reference
on the same GPU.

Peak PyTorch GPU allocation was **3.50 GiB**, with **3.55 GiB reserved**,
under a 4 GiB PyTorch allocator limit. The uncompressed layer shards totaled
54.71 GB. These figures cover the short text inference check and measure
PyTorch's GPU allocator; they exclude host memory, file cache, and native
allocations outside that allocator. Vision-input inference was not tested.

GPU tests are opt-in through `AIRLLM_GPU_TESTS=1` and skip when no GPU is
available; a skipped run is not GPU validation. The hardware results above
record the original suite before its conversion to unittest.
This change does not establish support for all architectures, pretrained
quantization formats, large-model memory bounds, or throughput.
PyTorch describes the API convention in
[HIP semantics](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/hip.rst).
