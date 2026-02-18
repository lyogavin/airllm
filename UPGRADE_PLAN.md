# AirLLM Upgrade Plan for Qwen3.5 Support

**Goal:** Update AirLLM to work with latest dependencies and support Qwen3.5-397B-A17B (MoE architecture)

---

## Model Information

**Model:** `Qwen3.5-397B-A17B`
- **Location:** `D:\AI-Models\Qwen3.5-397B-A17B`
- **Architecture:** `Qwen3_5MoeForConditionalGeneration`
- **Model Type:** `qwen3_5_moe`
- **Required transformers:** `4.57.0+` (per config.json)
- **Features:** Multimodal MoE with 512 experts, 10 experts per token, 60 layers

---

## Package Update Summary

### Current vs Required Versions

| Package | AirLLM Original | Working (Old) | Required (New) | Reason |
|---------|-----------------|---------------|----------------|--------|
| `transformers` | git+latest (2023) | 4.40.0 | **>=4.57.0** | Qwen3.5 MoE support |
| `optimum` | (implicit) | 1.16.0 | **>=2.1.0** | Latest version |
| `accelerate` | git+v0.20.3 | latest | **>=0.26.0** | Modern features |
| `torch` | (implicit) | >=2.0.0 | **>=2.0.0** | CUDA 12.x support |
| `safetensors` | (implicit) | latest | **>=0.4.0** | Performance |
| `bitsandbytes` | 0.39.0 | latest | **>=0.43.0** | Latest quantization |
| `huggingface-hub` | (implicit) | latest | **>=0.20.0** | Snapshot download |
| `scipy` | (implicit) | latest | **>=1.10.0** | Dependency |
| `tqdm` | (implicit) | latest | **latest** | No change |
| `sentencepiece` | 0.1.99 | latest | **>=0.2.0** | Tokenizer support |

---

## Code Changes Required

### 1. `airllm_base.py` - BetterTransformer Import Fix (CRITICAL)

**Problem:** Line 18 has hard import `from optimum.bettertransformer import BetterTransformer`
- optimum 2.x **removed** the bettertransformer module entirely
- This causes `ModuleNotFoundError` on import

**Solution:** Make import optional with graceful fallback

```python
# Change line 18 from:
from optimum.bettertransformer import BetterTransformer

# To:
try:
    from optimum.bettertransformer import BetterTransformer
    BETTERTRANSFORMER_AVAILABLE = True
except ImportError:
    BetterTransformer = None
    BETTERTRANSFORMER_AVAILABLE = False
```

**Also update `init_model()` method (~line 187):**
```python
if self.get_use_better_transformer() and BETTERTRANSFORMER_AVAILABLE:
    # ... existing BetterTransformer code
```

---

### 2. `auto_model.py` - Add Qwen3.5 MoE Support

**Problem:** `get_module_class()` doesn't recognize Qwen3.5 MoE architecture

**Current architectures handled:**
- `Qwen2ForCausalLM` → `AirLLMQWen2`
- `QWen` → `AirLLMQWen`

**Qwen3.5 architectures to add:**
- `Qwen3_5MoeForConditionalGeneration` → NEW class needed
- `Qwen3MoeForCausalLM` → NEW class needed  
- `Qwen3ForCausalLM` → Can use `AirLLMQWen2` base

**Solution:** Add detection for Qwen3/Qwen3.5 variants

---

### 3. NEW: `airllm_qwen3_moe.py` - Qwen3.5 MoE Handler

Create new handler class for Qwen3.5 MoE models with:
- Proper layer name mapping for MoE architecture
- Handling for 512 experts, 10 experts per token
- Vision encoder support (multimodal)
- Linear attention + full attention hybrid layers

---

### 4. `setup.py` - Update Dependencies

**Current (outdated):**
```python
install_requires=[
    'tqdm',
    'torch',
    'transformers',
    'accelerate',
    'safetensors',
    'optimum',
    'huggingface-hub',
    'scipy',
]
```

**Updated:**
```python
install_requires=[
    'tqdm',
    'torch>=2.0.0',
    'transformers>=4.57.0',
    'accelerate>=0.26.0',
    'safetensors>=0.4.0',
    'optimum>=2.1.0',
    'huggingface-hub>=0.20.0',
    'scipy>=1.10.0',
    'sentencepiece>=0.2.0',
]
```

Also update version to `2.12.0`

---

### 5. `__init__.py` - Export New Classes

Add imports for new Qwen3.5 classes:
```python
from .airllm_qwen3_moe import AirLLMQwen3Moe
```

---

## Implementation Order

1. **Fix `airllm_base.py`** - BetterTransformer import (unblocks all other work)
2. **Update `setup.py`** - New dependency versions
3. **Create `airllm_qwen3_moe.py`** - Qwen3.5 MoE handler
4. **Update `auto_model.py`** - Add Qwen3.5 detection
5. **Update `__init__.py`** - Export new classes
6. **Test locally** - Import and basic load test
7. **Build wheel** - Create airllm-2.12.0 wheel
8. **Update Docker** - Test in container with Qwen3.5-397B-A17B

---

## Testing Checklist

- [ ] `from airllm import AutoModel` works without error
- [ ] `AutoModel.from_pretrained("D:/AI-Models/Qwen3.5-397B-A17B")` loads config
- [ ] Model splits layers correctly for MoE architecture
- [ ] Basic inference produces output
- [ ] Docker container works with updated wheel

---

## Notes

- **Qwen3.5-397B-A17B is multimodal** - has vision encoder, may need special handling
- **MoE with 512 experts** - layer splitting needs to handle expert weights
- **Linear attention layers** - hybrid architecture with full attention every 4th layer
- **Data dir:** `D:\qwen-data`

---

*Created: 2026-02-18*
