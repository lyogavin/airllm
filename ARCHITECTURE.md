# AirLLM Architecture: How It Works

## Overview

AirLLM enables running large language models (70B+ parameters) on consumer-grade GPUs with as little as 4GB of VRAM. It achieves this by fundamentally changing how models are loaded and executed during inference.

## The Core Problem

Traditional LLM inference loads the entire model into GPU memory at once:
- A 70B parameter model in FP16 requires ~140GB of VRAM
- Consumer GPUs typically have 4-24GB of VRAM
- This makes large models inaccessible without expensive hardware

## The Core Solution: Layer-by-Layer Inference

AirLLM's breakthrough is **layer sharding with sequential execution**:

```
Traditional Approach:          AirLLM Approach:
┌─────────────────┐           ┌─────────────────┐
│  Load All Layers │          │  Load Layer 1    │
│  into GPU Memory │          │  Process Input   │
│                  │          │  Free Layer 1    │
│  Process Input   │   VS     │  Load Layer 2    │
│  Through Network │          │  Process Output  │
│                  │          │  Free Layer 2    │
│  Return Output   │          │  ...             │
└─────────────────┘           │  Load Layer N    │
                               │  Return Output   │
                               └─────────────────┘
```

### Key Insight

Neural network inference is naturally sequential - each layer processes the output of the previous layer. AirLLM exploits this by:

1. **Loading only one layer at a time** into GPU memory
2. **Processing the input** through that layer
3. **Storing intermediate activations** in RAM
4. **Freeing the layer** from GPU
5. **Repeating** for the next layer

This trades GPU memory for:
- **Disk I/O time** (loading layers from storage)
- **RAM capacity** (storing intermediate activations)
- **Slightly slower inference** (due to loading overhead)

## Architecture Components

### 1. Model Splitter (`utils.py::split_and_save_layers`)

**Purpose**: Decompose a standard model into layer-wise shards

**Process**:
```python
# Pseudocode of splitting process
for each layer in model:
    layer_state_dict = extract_parameters(layer)
    if compression:
        layer_state_dict = quantize(layer_state_dict)
    save_to_disk(layer_state_dict, f"layer_{i}.safetensors")
```

**What gets split**:
- Embedding layer (`model.embed_tokens`)
- Each transformer layer (`model.layers.0`, `model.layers.1`, ...)
- Normalization layer (`model.norm`)
- Language model head (`lm_head`)

**File format**: SafeTensors (efficient, safe serialization format)

**Compression options**:
- None (FP16): Default, no quality loss
- 8-bit: ~2x smaller, minimal quality loss
- 4-bit (NF4): ~4x smaller, slight quality loss

### 2. Base Model Class (`airllm_base.py::AirLLMBaseModel`)

**Purpose**: Core inference engine that orchestrates layer-by-layer execution

**Key data structures**:
```python
self.layers = [embed, layer_0, ..., layer_n, norm, lm_head]  # Model structure
self.layer_names = ["model.embed_tokens", ...]               # Layer identifiers
self.checkpoint_path = "/path/to/splitted_model"            # Where shards are stored
```

**Initialization steps**:
1. Download/locate the original model
2. Split into layers (if not already split)
3. Create empty model skeleton with `init_empty_weights()`
4. Load only model buffers (positional embeddings, etc.)
5. Keep layers on "meta" device (not materialized)

### 3. Forward Pass (`airllm_base.py::forward`)

**The heart of AirLLM** - how inference actually works:

```python
def forward(self, input_ids, ...):
    # 1. Initialize
    batch = [input_ids.to(device) for input_ids in input_ids]
    kv_cache = [] if use_cache else None

    # 2. Process each layer sequentially
    for layer_name, layer in zip(self.layer_names, self.layers):
        # 2a. Load layer weights from disk
        state_dict = load_layer(checkpoint_path, layer_name)

        # 2b. Move to GPU
        move_layer_to_device(state_dict)

        # 2c. Process input through layer
        for j, seq in enumerate(batch):
            batch[j] = layer(seq, attention_mask=..., ...)

        # 2d. Free GPU memory
        layer.to("meta")  # Move to meta device (deallocate)
        clean_memory()    # Force garbage collection

    # 3. Return final output
    return logits
```

**Key optimizations**:

#### A. Prefetching (Overlapping I/O and Computation)
```python
# Load layer N+1 while processing layer N
with ThreadPoolExecutor() as executor:
    future = executor.submit(load_layer, layer_names[i+1])

    # Process current layer
    process_layer(current_layer, input)

    # Wait for next layer to finish loading
    next_layer_state = future.result()
```

**Impact**: ~10% speedup by hiding disk I/O latency

#### B. Memory Pinning (Faster CPU→GPU Transfer)
```python
# Pin memory on CPU for faster DMA transfers
for param in state_dict.values():
    param.pin_memory()  # Lock in RAM, enable async GPU copy
```

#### C. Flash Attention / SDPA
```python
# Use optimized attention implementations
model = BetterTransformer.transform(model)  # Flash attention
# OR
config.attn_implementation = "sdpa"  # Scaled dot-product attention
```

### 4. KV Cache Management

For generation tasks, AirLLM maintains key-value caches to avoid recomputing:

```python
kv_cache_list = [
    (keys_layer_0, values_layer_0),    # Cache for layer 0
    (keys_layer_1, values_layer_1),    # Cache for layer 1
    ...
]

# During generation:
# 1st token: Compute and cache all KVs
# 2nd token onward: Reuse cached KVs, only compute new token
```

**Memory trade-off**:
- Caches stored in RAM (not GPU)
- Enables fast generation without reprocessing full sequence
- Essential for multi-turn conversations

### 5. Compression System (`utils.py::compress_layer_state_dict`)

**4-bit Quantization** (NF4 - Normal Float 4-bit):
```python
# Compress to 4-bit using block-wise quantization
for param_name, param in layer_state_dict.items():
    quantized, quant_state = bnb.functional.quantize_nf4(
        param.cuda(),
        blocksize=64  # Quantize in 64-element blocks
    )
    # Save: quantized values + quantization metadata
```

**8-bit Quantization**:
```python
# Compress to 8-bit
quantized, quant_state = bnb.functional.quantize_blockwise(
    param.cuda(),
    blocksize=2048
)
# Save: quantized values + absmax (for dequantization)
```

**Why block-wise quantization?**
- Handles outliers better than global quantization
- Each block has its own scale factor
- Minimal accuracy loss (~0.1-0.5% on benchmarks)

### 6. Multi-Architecture Support (`auto_model.py`)

AirLLM supports multiple model architectures through architecture detection:

```python
config = AutoConfig.from_pretrained(model_path)

if "Llama" in config.architectures[0]:
    return AirLLMLlama2(...)
elif "Qwen2" in config.architectures[0]:
    return AirLLMQWen2(...)
elif "Mistral" in config.architectures[0]:
    return AirLLMMistral(...)
# ... etc
```

Each architecture class inherits from `AirLLMBaseModel` and customizes:
- Layer naming conventions
- Position embedding handling
- Attention mechanisms
- Tokenizer configurations

## Performance Characteristics

### Memory Usage

| Component | GPU Memory | RAM | Disk |
|-----------|-----------|-----|------|
| Single layer (70B) | ~2-3GB | - | - |
| Intermediate activations | - | ~1-2GB | - |
| KV cache | - | ~500MB-2GB | - |
| Model shards | - | - | ~140GB (FP16) |
| Model shards (4-bit) | - | - | ~35GB |

**Total GPU usage**: 3-4GB for a 70B model

### Speed

Compared to standard inference on high-memory GPU:
- **No compression**: ~10-15x slower (disk I/O bottleneck)
- **With compression**: ~3-5x slower (smaller files, faster loading)
- **With prefetching**: ~10% faster than without

Typical 70B model on 4GB GPU:
- First token: ~10-30 seconds
- Subsequent tokens: ~2-5 seconds each

## System Requirements

### Minimum
- **GPU**: 4GB VRAM (for 70B models)
- **RAM**: 8GB+
- **Disk**: 150GB free (for model shards + original)
- **Disk type**: SSD strongly recommended

### Recommended
- **GPU**: 8GB+ VRAM (for 405B models or faster inference)
- **RAM**: 16GB+
- **Disk**: NVMe SSD (5-10x faster loading than HDD)

## Limitations & Trade-offs

### What AirLLM Sacrifices
1. **Speed**: Significantly slower than full GPU inference
2. **Disk space**: Requires storing split model (can delete original)
3. **Batch processing**: Less efficient with large batches

### What AirLLM Preserves
1. **Model quality**: No accuracy loss (without compression)
2. **Full model capabilities**: All layers, no pruning/distillation
3. **Flexibility**: Same interface as standard transformers

## When to Use AirLLM

**Good use cases**:
- Running 70B+ models on consumer hardware
- Prototyping with large models
- Low-throughput applications (chatbots, research)
- GPU memory constrained environments

**Not ideal for**:
- High-throughput production serving
- Real-time applications requiring <100ms latency
- Batch inference on many samples
- When high-memory GPUs are available

## Advanced Features

### Profiling Mode
```python
model = AutoModel.from_pretrained(..., profiling_mode=True)
# Outputs detailed timing for:
# - Disk loading time
# - GPU transfer time
# - Compression/decompression time
# - Computation time
```

### Custom Layer Shards Path
```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-70b",
    layer_shards_saving_path="/fast/nvme/disk"  # Use fastest disk
)
```

### MacOS Support (Apple Silicon)
```python
# Automatically uses MLX backend on MacOS
# Leverages unified memory architecture
model = AutoModel.from_pretrained(...)
```

## Implementation Highlights

### 1. Memory Management
```python
def clean_memory():
    gc.collect()                              # Python garbage collection
    ctypes.CDLL("libc.so.6").malloc_trim(0)  # Release libc caches
    torch.cuda.empty_cache()                  # Clear PyTorch cache
```

Called after each layer to ensure memory is actually freed.

### 2. Meta Device Pattern
```python
# Create model structure without allocating memory
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Layers exist but have no actual tensors
# Later, load weights one layer at a time
set_module_tensor_to_device(model, param_name, device, value=tensor)
```

This is the key to initializing large models without OOM errors.

### 3. Safe Model Splitting
```python
# Check disk space before splitting
total_size = sum(os.path.getsize(f) for f in model_files)
free_space = shutil.disk_usage(path).free

if free_space < total_size:
    raise NotEnoughSpaceException(...)
```

Prevents partial splits that corrupt the model.

## Conclusion

AirLLM demonstrates that with clever engineering, we can overcome hardware limitations. By understanding the sequential nature of neural network inference and carefully managing memory, disk I/O, and computation, it makes state-of-the-art LLMs accessible to anyone with a consumer GPU.

The key insights:
1. **Sequential layer loading** instead of full model loading
2. **RAM for activations**, **GPU for computation**, **Disk for storage**
3. **Compression** to reduce I/O bottleneck
4. **Prefetching** to overlap I/O and compute
5. **Clean abstractions** to support multiple architectures

This architecture pattern can be applied to other memory-constrained deep learning scenarios beyond LLMs.
