# Building Memory-Efficient LLM Inference Systems

A comprehensive guide to creating systems like AirLLM that run large models on limited hardware.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [System Design Principles](#system-design-principles)
3. [Implementation Guide](#implementation-guide)
4. [Optimization Techniques](#optimization-techniques)
5. [Common Pitfalls](#common-pitfalls)
6. [Testing & Benchmarking](#testing--benchmarking)
7. [Production Considerations](#production-considerations)

## Core Concepts

### The Memory Hierarchy

Understanding the memory hierarchy is crucial for building efficient systems:

```
┌─────────────────────────────────────────────┐
│ GPU VRAM (Fastest, Smallest)                │
│ - 4-80GB typical                             │
│ - ~1TB/s bandwidth                          │
│ - Use for: Active layer computation         │
├─────────────────────────────────────────────┤
│ System RAM (Fast, Moderate)                 │
│ - 16-256GB typical                          │
│ - ~50-100GB/s bandwidth                     │
│ - Use for: Intermediate activations, KV     │
├─────────────────────────────────────────────┤
│ NVMe SSD (Medium, Large)                    │
│ - 500GB-4TB typical                         │
│ - ~3-7GB/s read speed                       │
│ - Use for: Model layer shards               │
├─────────────────────────────────────────────┤
│ SATA SSD (Slower, Large)                    │
│ - ~500MB/s read speed                       │
│ - Use for: Model storage (slower)           │
├─────────────────────────────────────────────┤
│ HDD (Slowest, Largest)                      │
│ - ~100-200MB/s read speed                   │
│ - Avoid for active inference if possible    │
└─────────────────────────────────────────────┘
```

### The Fundamental Trade-off

```python
# Traditional inference
speed = HIGH
memory_usage = ENTIRE_MODEL  # Problem for large models
quality = FULL

# Memory-efficient inference
speed = MEDIUM_TO_LOW
memory_usage = SINGLE_LAYER  # Fits on small GPUs
quality = FULL (with optional compression trade-off)
```

**Key insight**: You're trading GPU memory for disk I/O time and RAM capacity.

## System Design Principles

### Principle 1: Layer Granularity

Split your model at the **layer level**, not arbitrary chunks:

```python
# GOOD: Layer-level splitting
layers = [
    "embedding",
    "transformer_layer_0",
    "transformer_layer_1",
    ...
    "transformer_layer_n",
    "output_head"
]

# BAD: Arbitrary splitting
chunks = [
    "first_25_percent_of_model",
    "second_25_percent_of_model",
    ...
]
```

**Why?** Neural networks have natural boundaries at layer transitions. This:
- Simplifies state management (clear inputs/outputs)
- Enables layer-wise optimizations
- Makes debugging easier
- Supports architecture-specific customizations

### Principle 2: Minimize Data Movement

**Every data transfer has a cost:**

```python
# Cost hierarchy (from expensive to cheap)
1. Disk → RAM → GPU → Compute → GPU → RAM → Disk (EXPENSIVE)
2. RAM → GPU → Compute → GPU → RAM (BETTER)
3. GPU → Compute → GPU (BEST)

# Strategy: Keep intermediate results in fastest possible tier
activations = []  # Keep in RAM between layers
current_layer = load_to_gpu(layer_n)  # Only current layer on GPU
output = compute(current_layer, activations[-1])
activations.append(output.to('cpu'))  # Move result to RAM
```

### Principle 3: Overlap I/O and Computation

**Never let the GPU idle waiting for data:**

```python
# SEQUENTIAL (wasteful)
load_layer_n()      # GPU idle :(
compute_layer_n()   # I/O idle
load_layer_n_plus_1()  # GPU idle :(
compute_layer_n_plus_1()

# PIPELINED (efficient)
load_layer_n()
load_layer_n_plus_1_async()  # Start loading next
compute_layer_n()             # While computing, next layer loads
wait(layer_n_plus_1)          # Should be ready or nearly ready
compute_layer_n_plus_1()
```

**Implementation**: Use threading, async I/O, or CUDA streams.

### Principle 4: Compression When Appropriate

**Not all precision is equal:**

```python
# Full precision
weights_fp32 = 4 bytes/param × 70B params = 280GB

# Half precision (minimal quality loss)
weights_fp16 = 2 bytes/param × 70B params = 140GB

# 8-bit quantization (slight quality loss)
weights_int8 = 1 byte/param × 70B params = 70GB

# 4-bit quantization (noticeable but acceptable loss)
weights_int4 = 0.5 bytes/param × 70B params = 35GB
```

**Rule of thumb**: Use the lowest precision that maintains acceptable quality for your use case.

## Implementation Guide

### Step 1: Model Inspection & Splitting

**Goal**: Decompose a standard model into loadable shards.

```python
import torch
from pathlib import Path
from safetensors.torch import save_file, load_file
import json

def inspect_model_structure(model_path):
    """Understand the model's layer structure."""
    # Load the model index
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index['weight_map']

    # Group by layer
    layers = {}
    for param_name, shard_file in weight_map.items():
        # Extract layer identifier
        # Example: "model.layers.0.attention.query.weight" → "model.layers.0"
        layer_id = extract_layer_id(param_name)

        if layer_id not in layers:
            layers[layer_id] = []
        layers[layer_id].append(param_name)

    return layers

def extract_layer_id(param_name):
    """Extract layer identifier from parameter name."""
    parts = param_name.split('.')

    # Common patterns:
    if 'embed' in param_name:
        return 'embedding'
    elif 'layers' in parts:
        # "model.layers.12.attn.weight" → "model.layers.12"
        layer_idx = parts.index('layers')
        return '.'.join(parts[:layer_idx+2])
    elif 'lm_head' in param_name or 'output' in param_name:
        return 'lm_head'
    elif 'norm' in param_name:
        return 'norm'
    else:
        return 'other'

def split_model(model_path, output_dir):
    """Split model into layer-wise shards."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    layers = inspect_model_structure(model_path)

    # Load original model shards
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    # Group parameters by source file
    shard_files = set(index['weight_map'].values())

    for shard_file in shard_files:
        # Load shard
        shard_path = Path(model_path) / shard_file
        state_dict = load_file(shard_path, device='cpu')

        # Distribute to layer files
        for layer_id, param_names in layers.items():
            layer_state = {
                name: state_dict[name]
                for name in param_names
                if name in state_dict
            }

            if layer_state:
                output_path = output_dir / f"{layer_id}.safetensors"
                save_file(layer_state, output_path)

        # Free memory
        del state_dict
        torch.cuda.empty_cache()

    print(f"Split into {len(layers)} layer shards at {output_dir}")
```

### Step 2: Layer Loader

**Goal**: Efficiently load individual layers into GPU memory.

```python
class LayerLoader:
    """Handles loading layer shards with caching and prefetching."""

    def __init__(self, checkpoint_path, prefetch=True):
        self.checkpoint_path = Path(checkpoint_path)
        self.prefetch = prefetch
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def load_layer(self, layer_name):
        """Load a layer's parameters from disk."""
        layer_path = self.checkpoint_path / f"{layer_name}.safetensors"

        # Load to CPU first
        state_dict = load_file(layer_path, device='cpu')

        # Pin memory for faster GPU transfer
        for key in state_dict:
            state_dict[key] = state_dict[key].pin_memory()

        return state_dict

    def prefetch_layer(self, layer_name):
        """Asynchronously start loading next layer."""
        if self.prefetch:
            self.future = self.executor.submit(self.load_layer, layer_name)

    def get_prefetched_layer(self):
        """Wait for prefetched layer to finish loading."""
        if self.future is not None:
            state_dict = self.future.result()
            self.future = None
            return state_dict
        return None

    def load_to_device(self, state_dict, model, device):
        """Move layer parameters to device and attach to model."""
        from accelerate.utils import set_module_tensor_to_device

        for param_name, param in state_dict.items():
            set_module_tensor_to_device(
                model,
                param_name,
                device,
                value=param
            )
```

### Step 3: Inference Engine

**Goal**: Orchestrate layer-by-layer forward pass.

```python
class MemoryEfficientModel:
    """Memory-efficient model that loads layers on-demand."""

    def __init__(self, model_path, checkpoint_path, device='cuda:0'):
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        self.device = device
        self.checkpoint_path = checkpoint_path

        # Create empty model (no memory allocation)
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config)

        # Identify layers
        self.layer_names = self._identify_layers()

        # Setup loader
        self.loader = LayerLoader(checkpoint_path, prefetch=True)

    def _identify_layers(self):
        """Get ordered list of layer names."""
        # This depends on model architecture
        # For Llama-style models:
        n_layers = len(self.model.model.layers)
        return (
            ['model.embed_tokens'] +
            [f'model.layers.{i}' for i in range(n_layers)] +
            ['model.norm', 'lm_head']
        )

    def forward(self, input_ids, attention_mask=None):
        """Execute forward pass layer-by-layer."""
        # Current activation
        hidden_states = input_ids

        # Prefetch first layer
        self.loader.prefetch_layer(self.layer_names[0])

        for i, layer_name in enumerate(self.layer_names):
            # Get current layer (from prefetch if available)
            state_dict = self.loader.get_prefetched_layer()
            if state_dict is None:
                state_dict = self.loader.load_layer(layer_name)

            # Load to GPU
            self.loader.load_to_device(state_dict, self.model, self.device)

            # Prefetch next layer
            if i + 1 < len(self.layer_names):
                self.loader.prefetch_layer(self.layer_names[i + 1])

            # Execute layer
            layer = self._get_layer_module(layer_name)
            hidden_states = self._execute_layer(
                layer,
                hidden_states,
                attention_mask
            )

            # Free GPU memory
            self._unload_layer(layer)

        return hidden_states

    def _get_layer_module(self, layer_name):
        """Get the layer module from model."""
        module = self.model
        for part in layer_name.split('.'):
            module = getattr(module, part)
        return module

    def _execute_layer(self, layer, hidden_states, attention_mask):
        """Run forward pass through a single layer."""
        if isinstance(layer, torch.nn.Embedding):
            return layer(hidden_states)
        elif isinstance(layer, torch.nn.Linear):
            return layer(hidden_states)
        else:
            # Transformer layer
            return layer(
                hidden_states,
                attention_mask=attention_mask
            )[0]

    def _unload_layer(self, layer):
        """Remove layer from GPU memory."""
        layer.to('meta')  # Move to meta device (deallocates)
        torch.cuda.empty_cache()
        gc.collect()
```

### Step 4: Generation Support

**Goal**: Support text generation with KV caching.

```python
class GenerationMixin:
    """Add generation capabilities to memory-efficient model."""

    def generate(self, input_ids, max_new_tokens=20, **kwargs):
        """Generate text autoregressively."""
        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                generated,
                past_key_values=past_key_values,
                use_cache=True
            )

            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Update sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Update KV cache
            past_key_values = outputs.past_key_values

            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return generated

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        """Forward pass with KV cache support."""
        hidden_states = input_ids
        new_kv_cache = [] if use_cache else None

        # Process only new tokens if using cache
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]  # Only last token

        for i, layer_name in enumerate(self.layer_names):
            # Load layer...
            state_dict = self._load_layer(layer_name)
            layer = self._get_layer_module(layer_name)

            # Execute with cache
            if 'layers' in layer_name and use_cache:
                # Pass previous KV cache for this layer
                past_kv = past_key_values[i] if past_key_values else None

                outputs = layer(
                    hidden_states,
                    past_key_value=past_kv,
                    use_cache=True
                )

                hidden_states = outputs[0]
                new_kv = outputs[1]  # (key, value) tuple
                new_kv_cache.append(new_kv)
            else:
                hidden_states = self._execute_layer(layer, hidden_states)

            # Unload layer...
            self._unload_layer(layer)

        return ModelOutput(
            logits=hidden_states,
            past_key_values=tuple(new_kv_cache) if use_cache else None
        )
```

## Optimization Techniques

### 1. Compression Implementation

```python
def quantize_layer_4bit(state_dict):
    """Quantize layer to 4-bit using NF4."""
    import bitsandbytes as bnb

    quantized = {}
    for name, param in state_dict.items():
        # Quantize to 4-bit
        quant_param, quant_state = bnb.functional.quantize_nf4(
            param.cuda(),
            blocksize=64  # 64-element blocks
        )

        # Store quantized param
        quantized[name] = quant_param.cpu()

        # Store quantization metadata
        quantized[f"{name}.quant_state"] = {
            'absmax': quant_state.absmax.cpu(),
            'code': quant_state.code.cpu(),
            'blocksize': quant_state.blocksize,
            'quant_type': quant_state.quant_type,
        }

    return quantized

def dequantize_layer_4bit(quantized_dict):
    """Dequantize 4-bit layer back to FP16."""
    import bitsandbytes as bnb

    dequantized = {}
    for name, param in quantized_dict.items():
        if '.quant_state' not in name:
            # Reconstruct quant state
            quant_state_dict = quantized_dict[f"{name}.quant_state"]
            quant_state = bnb.functional.QuantState(
                absmax=quant_state_dict['absmax'].cuda(),
                code=quant_state_dict['code'].cuda(),
                blocksize=quant_state_dict['blocksize'],
                quant_type=quant_state_dict['quant_type']
            )

            # Dequantize
            dequantized[name] = bnb.functional.dequantize_nf4(
                param.cuda(),
                quant_state
            )

    return dequantized
```

### 2. Disk I/O Optimization

```python
import os

def optimize_disk_reads(checkpoint_path):
    """Optimize file system for sequential reads."""
    # Use direct I/O to bypass OS cache (for large files)
    # This prevents polluting system cache with model weights

    import fcntl

    def read_with_direct_io(filepath):
        fd = os.open(filepath, os.O_RDONLY | os.O_DIRECT)
        # Read in aligned blocks
        block_size = 4096
        data = b''
        while True:
            chunk = os.read(fd, block_size)
            if not chunk:
                break
            data += chunk
        os.close(fd)
        return data

def use_memory_mapped_files(filepath):
    """Use mmap for faster access to layer files."""
    import mmap

    with open(filepath, 'r+b') as f:
        # Memory-map the file
        mm = mmap.mmap(f.fileno(), 0)

        # OS will page in data as needed
        # Much faster for random access patterns
        return mm

def optimize_ssd_scheduling():
    """Optimize SSD scheduler for throughput."""
    # On Linux, use 'noop' or 'none' scheduler for NVMe
    # This reduces latency for direct I/O

    # Example: echo "none" > /sys/block/nvme0n1/queue/scheduler
    pass
```

### 3. Memory Management

```python
def aggressive_memory_cleanup():
    """Aggressively free memory after each layer."""
    import gc
    import ctypes

    # Python garbage collection
    gc.collect()

    # Linux: release glibc memory back to OS
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

    # PyTorch: clear CUDA cache
    torch.cuda.empty_cache()

    # Force synchronization
    torch.cuda.synchronize()

def monitor_memory_usage():
    """Track memory usage for debugging."""
    import psutil

    process = psutil.Process()

    gpu_mem = torch.cuda.memory_allocated() / 1e9
    ram = process.memory_info().rss / 1e9

    print(f"GPU: {gpu_mem:.2f}GB | RAM: {ram:.2f}GB")
```

### 4. Attention Optimization

```python
def use_flash_attention(model):
    """Enable Flash Attention for 2-4x speedup."""
    try:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
        print("Using Flash Attention via BetterTransformer")
    except:
        # Fallback to SDPA
        if hasattr(model.config, 'attn_implementation'):
            model.config.attn_implementation = 'sdpa'
            print("Using PyTorch SDPA attention")

    return model

def optimize_kv_cache_memory(kv_cache):
    """Compress KV cache to save RAM."""
    # KV cache can be large in long contexts
    # Options:
    # 1. Quantize to int8 (2x smaller)
    # 2. Use sliding window (discard old tokens)
    # 3. Compress with low-rank approximation

    compressed_cache = []
    for key, value in kv_cache:
        # Quantize to int8
        key_int8 = (key * 127).to(torch.int8)
        value_int8 = (value * 127).to(torch.int8)
        compressed_cache.append((key_int8, value_int8))

    return compressed_cache
```

## Common Pitfalls

### Pitfall 1: Not Handling Model Variants

```python
# WRONG: Assumes all models have same structure
layer_names = [f'model.layers.{i}' for i in range(32)]

# RIGHT: Detect model structure dynamically
def get_layer_names(config):
    if config.model_type == 'llama':
        n = config.num_hidden_layers
        return [f'model.layers.{i}' for i in range(n)]
    elif config.model_type == 'gpt2':
        n = config.n_layer
        return [f'transformer.h.{i}' for i in range(n)]
    # ... handle more architectures
```

### Pitfall 2: Ignoring Batch Processing

```python
# WRONG: Only handles single input
def forward(self, input_ids):
    # Assumes input_ids is [1, seq_len]
    ...

# RIGHT: Handle batches properly
def forward(self, input_ids):
    batch_size = input_ids.shape[0]
    # Process each item in batch
    # OR vectorize operations
```

### Pitfall 3: Memory Leaks

```python
# WRONG: Holds references to old layers
self.layers_cache = {}
for layer_name in layer_names:
    layer = load_layer(layer_name)
    self.layers_cache[layer_name] = layer  # LEAK!
    process(layer)

# RIGHT: Explicitly free memory
for layer_name in layer_names:
    layer = load_layer(layer_name)
    process(layer)
    del layer  # Free reference
    torch.cuda.empty_cache()
```

### Pitfall 4: Inefficient Serialization

```python
# WRONG: Use pickle (slow, unsafe)
torch.save(state_dict, 'layer.bin')

# RIGHT: Use safetensors (fast, safe)
from safetensors.torch import save_file
save_file(state_dict, 'layer.safetensors')
```

## Testing & Benchmarking

### Correctness Testing

```python
def test_output_correctness():
    """Verify layer-by-layer gives same results as standard inference."""
    from transformers import AutoModelForCausalLM

    # Reference model (full GPU)
    reference_model = AutoModelForCausalLM.from_pretrained(
        "model_name",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Our implementation
    efficient_model = MemoryEfficientModel("model_name", "checkpoint")

    # Test inputs
    input_ids = torch.randint(0, 1000, (1, 128))

    # Compare outputs
    with torch.no_grad():
        ref_output = reference_model(input_ids).logits
        our_output = efficient_model(input_ids)

    # Check similarity
    diff = (ref_output - our_output).abs().max()
    assert diff < 1e-3, f"Outputs differ by {diff}"

    print("✓ Outputs match reference implementation")
```

### Performance Benchmarking

```python
import time

def benchmark_inference():
    """Measure inference speed."""
    model = MemoryEfficientModel("model_name", "checkpoint")
    input_ids = torch.randint(0, 1000, (1, 128))

    # Warmup
    _ = model(input_ids)

    # Benchmark
    times = []
    for _ in range(10):
        start = time.time()
        _ = model(input_ids)
        times.append(time.time() - start)

    print(f"Average time: {sum(times)/len(times):.2f}s")
    print(f"Tokens/sec: {128 / (sum(times)/len(times)):.2f}")

def profile_bottlenecks():
    """Identify performance bottlenecks."""
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    model = MemoryEfficientModel("model_name", "checkpoint")
    model(torch.randint(0, 1000, (1, 128)))

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

## Production Considerations

### 1. Error Handling

```python
class RobustLayerLoader:
    """Layer loader with retry logic and error recovery."""

    def load_layer(self, layer_name, max_retries=3):
        for attempt in range(max_retries):
            try:
                return self._load_layer_impl(layer_name)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Load failed (attempt {attempt+1}), retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

    def verify_checkpoint_integrity(self, checkpoint_path):
        """Verify all layer files are present and valid."""
        missing = []
        corrupted = []

        for layer_name in self.layer_names:
            path = checkpoint_path / f"{layer_name}.safetensors"

            if not path.exists():
                missing.append(layer_name)
            else:
                try:
                    # Try to load
                    _ = load_file(path, device='cpu')
                except:
                    corrupted.append(layer_name)

        if missing or corrupted:
            raise ValueError(
                f"Missing: {missing}, Corrupted: {corrupted}"
            )
```

### 2. Multi-GPU Support

```python
class MultiGPUModel:
    """Distribute computation across multiple GPUs."""

    def __init__(self, model_path, checkpoint_path, devices=['cuda:0', 'cuda:1']):
        self.devices = devices
        self.current_device_idx = 0

    def forward(self, input_ids):
        hidden_states = input_ids

        for i, layer_name in enumerate(self.layer_names):
            # Round-robin across GPUs
            device = self.devices[i % len(self.devices)]

            # Load to specific GPU
            layer = self.load_layer(layer_name, device)
            hidden_states = hidden_states.to(device)
            hidden_states = layer(hidden_states)

        return hidden_states
```

### 3. Monitoring & Logging

```python
import logging
from dataclasses import dataclass
from typing import List

@dataclass
class InferenceMetrics:
    """Track inference metrics."""
    layer_load_times: List[float]
    layer_compute_times: List[float]
    peak_gpu_memory: float
    peak_ram_memory: float
    total_time: float

class MonitoredModel:
    """Model with comprehensive monitoring."""

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.metrics = InferenceMetrics([], [], 0, 0, 0)

    def forward(self, input_ids):
        start_time = time.time()

        for layer_name in self.layer_names:
            # Monitor loading
            load_start = time.time()
            layer = self.load_layer(layer_name)
            load_time = time.time() - load_start
            self.metrics.layer_load_times.append(load_time)

            # Monitor computation
            compute_start = time.time()
            output = self.execute_layer(layer, input_ids)
            compute_time = time.time() - compute_start
            self.metrics.layer_compute_times.append(compute_time)

            # Monitor memory
            gpu_mem = torch.cuda.memory_allocated()
            self.metrics.peak_gpu_memory = max(
                self.metrics.peak_gpu_memory,
                gpu_mem
            )

            self.logger.debug(
                f"Layer {layer_name}: "
                f"load={load_time:.3f}s, "
                f"compute={compute_time:.3f}s, "
                f"gpu_mem={gpu_mem/1e9:.2f}GB"
            )

        self.metrics.total_time = time.time() - start_time
        return output
```

## Advanced Topics

### Dynamic Batching

```python
class DynamicBatcher:
    """Batch multiple requests for efficiency."""

    def __init__(self, model, max_batch_size=8, timeout=0.1):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.queue = []

    async def infer(self, input_ids):
        """Add request to batch queue."""
        future = asyncio.Future()
        self.queue.append((input_ids, future))

        # Trigger batch if full
        if len(self.queue) >= self.max_batch_size:
            await self._process_batch()

        return await future

    async def _process_batch(self):
        """Process accumulated batch."""
        batch_inputs = [x[0] for x in self.queue]
        futures = [x[1] for x in self.queue]
        self.queue = []

        # Pad to same length
        max_len = max(x.shape[1] for x in batch_inputs)
        padded = [
            F.pad(x, (0, max_len - x.shape[1]))
            for x in batch_inputs
        ]

        # Batch inference
        batch = torch.cat(padded, dim=0)
        outputs = self.model(batch)

        # Distribute results
        for i, future in enumerate(futures):
            future.set_result(outputs[i])
```

### Model Streaming

```python
class StreamingModel:
    """Stream model from cloud storage during inference."""

    def __init__(self, s3_bucket, model_prefix):
        import boto3
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket
        self.prefix = model_prefix
        self.cache_dir = Path("/tmp/model_cache")

    def load_layer(self, layer_name):
        """Load layer from S3 if not cached locally."""
        cache_path = self.cache_dir / f"{layer_name}.safetensors"

        if not cache_path.exists():
            # Download from S3
            s3_key = f"{self.prefix}/{layer_name}.safetensors"
            self.s3.download_file(
                self.bucket,
                s3_key,
                str(cache_path)
            )

        return load_file(cache_path, device='cpu')
```

## Conclusion

Building memory-efficient LLM inference systems requires:

1. **Understanding the memory hierarchy** and optimizing data movement
2. **Layer-granular splitting** for clean abstractions
3. **Prefetching** to overlap I/O and computation
4. **Compression** to reduce I/O bottleneck
5. **Careful memory management** to avoid leaks
6. **Robust error handling** for production use
7. **Comprehensive testing** for correctness and performance

The techniques in this guide can be applied to any memory-constrained deep learning scenario. The key is understanding your hardware constraints and making intelligent trade-offs between speed, memory, and quality.

### Further Reading

- [Accelerate documentation](https://huggingface.co/docs/accelerate) - For model sharding
- [SafeTensors](https://github.com/huggingface/safetensors) - For efficient serialization
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - For quantization
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - For attention optimization
- [vLLM](https://github.com/vllm-project/vllm) - For production serving (different approach)

### Contributing

If you build a system using these techniques, consider:
- Open-sourcing your implementation
- Sharing benchmarks and optimizations
- Contributing back improvements

The goal is to make large language models accessible to everyone, regardless of hardware budget.
