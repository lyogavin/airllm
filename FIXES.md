# Compatibility Fixes for transformers 5.x, bitsandbytes 0.49+, PyTorch 2.10+

## Getting a 70B LLM running on a single 8GB GPU — and the rabbit hole of dependency fixes that got us there

AirLLM lets you run massive language models on consumer hardware by loading one layer at a time into GPU memory. Instead of needing 40GB+ VRAM for a 70B model, you stream each of the 80+ transformer layers through your GPU sequentially.

The library hadn't been updated since mid-2024, and the Python ML ecosystem had moved on. Here's what broke and how it was fixed.

### 1. Optional dependency gatekeeping the entire library

`optimum.bettertransformer` was imported unconditionally at the top of the module. If you didn't have it installed, you couldn't even `from airllm import AutoModel`. The fix: wrap it in a try/except and track availability with a flag, since BetterTransformer is actually deprecated in favour of PyTorch's native SDPA attention anyway.

### 2. Missing class attribute breaks generation pipeline

Transformers 5.x added a `_is_stateful` property check inside `GenerationMixin._supports_default_dynamic_cache()`. AirLLM's base model class inherits from `GenerationMixin` but never defined this attribute. One line fix: `_is_stateful = False` on the class. Small change, total showstopper without it.

### 3. KV cache API completely changed

The old transformers used plain Python tuples for past key values — you could do `past_key_values[layer_idx]` to get `(key, value)` for a layer. Transformers 5.x replaced this with a `DynamicCache` object that isn't subscriptable. The code had multiple places indexing into past_key_values as if it were a list of tuples. Fix: detect `DynamicCache` instances and use `.get_seq_length()` instead, and neutralise the cache in the forward pass since AirLLM's layer-by-layer approach doesn't benefit from it.

### 4. Quantization loading API was renamed and restructured

The bitsandbytes integration in transformers went through a major refactor:
- `check_quantized_param()` -> `param_needs_quantization()`
- `create_quantized_param()` -> removed entirely
- `update_torch_dtype()` -> `update_dtype()`

The old `create_quantized_param` handled loading pre-quantized 4-bit weights. The new API has `get_quantize_ops().convert()`, but that's designed for quantizing float weights from scratch — it fails on already-quantized uint8 data with "Blockwise 4bit quantization only supports 16/32-bit floats, but got torch.uint8".

The fix was to bypass the new quantization pipeline entirely for pre-quantized weights: reconstruct `bnb.functional.QuantState` from the stored metadata tensors, create `Params4bit` directly with `bnb_quantized=True`, and set the weight on the module manually (avoiding `accelerate`'s `set_module_tensor_to_device` which would strip the `bnb_quantized` flag and trigger re-quantization).

### 5. Decoder layers no longer return tuples

This was the subtlest bug. In older transformers, `LlamaDecoderLayer.forward()` returned a tuple `(hidden_states, ...)`. The code did `layer(seq, **kwargs)[0]` to extract hidden states.

In transformers 5.x, it returns a plain tensor. So `[0]` was indexing dimension 0 of the tensor itself — silently slicing off the batch dimension. The model would process layer 0 fine with shape `[1, 9, 8192]`, then layer 1 would receive `[9, 8192]`, and the rotary embedding would fail with a cryptic dimension mismatch: "size of tensor a (64) must match size of tensor b (128)". The 64 came from the head dimension being miscomputed after the batch dim was dropped.

Fix: check `isinstance(result, torch.Tensor)` before indexing.

### 6. Rotary embeddings moved out of attention layers

Transformers 5.x moved rotary position embedding computation out of individual attention layers. Instead of each layer computing its own cos/sin from `position_ids`, a shared `rotary_emb` module on the model produces `position_embeddings` that get passed through. AirLLM was passing `position_ids` to each layer but not `position_embeddings`, resulting in `None` being unpacked as `cos, sin = position_embeddings`.

Fix: grab the model's `rotary_emb` module and compute `position_embeddings` dynamically for each layer call with the correct position IDs for the current sequence slice.

---

**End result:** Llama 3.1 70B (4-bit quantized) running inference on a single RTX 3070 with 8GB VRAM. 83 layers streamed through in ~36 seconds.

### Environment
- Python 3.11
- PyTorch 2.10.0+cu126
- transformers 5.3.0
- bitsandbytes 0.49.2
- Windows 11
