import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.quantizers import AutoHfQuantizer

from .airllm_base import AirLLMBaseModel
from .utils import clean_memory

def _build_qwen35_moe_skeleton(config):
    """
    Directly instantiate Qwen3_5MoeForConditionalGeneration.
    AutoModelForCausalLM.from_config fails because the outer multimodal config
    does not expose vocab_size at the top level (it lives on config.text_config).
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
    )
    return Qwen3_5MoeForConditionalGeneration(config)


def _is_qwen35_moe_multimodal(config_or_path):
    """Returns True for Qwen3_5MoeForConditionalGeneration (397B multimodal wrapper).
    Accepts either a config object or a filesystem path string."""
    if isinstance(config_or_path, str):
        import json, os
        cfg_path = os.path.join(config_or_path, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                data = json.load(f)
            archs = data.get('architectures', []) or []
        else:
            return False
    else:
        archs = getattr(config_or_path, 'architectures', []) or []
    return any('Qwen3_5Moe' in a for a in archs)


class AirLLMQwen3Moe(AirLLMBaseModel):
    """
    AirLLM handler for Qwen3 MoE models. Supports two variants:
    - Qwen3MoeForCausalLM (e.g. Qwen3-30B-A3B): standard model.layers.* layout
    - Qwen3_5MoeForConditionalGeneration (e.g. Qwen3.5-397B): multimodal wrapper,
      layers nested under model.language_model.*, includes SSM (linear_attn) layers
    """

    def __init__(self, *args, **kwargs):
        super(AirLLMQwen3Moe, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def set_experts_implementation(self, implementation: str):
        pass

    def get_generation_config(self):
        return GenerationConfig()

    def forward(self, input_ids=None, **kwargs):
        """
        Override base forward to keep the model skeleton alive between tokens.
        Base class handles dynamic n_layers_in_gpu recalculation from VRAM.
        """
        result = super().forward(input_ids=input_ids, **kwargs)
        self._skip_model_reinit = True
        return result

    def init_model(self):
        self.model = None
        self.hf_quantizer = None

        if _is_qwen35_moe_multimodal(self.config):
            # Qwen3.5-397B: multimodal wrapper — AutoModelForCausalLM.from_config fails
            # because vocab_size lives on config.text_config, not at top level.
            print("Qwen3.5 MoE: building multimodal conditional-generation skeleton...")
            try:
                with init_empty_weights():
                    self.model = _build_qwen35_moe_skeleton(self.config)
            except Exception as e:
                clean_memory()
                raise RuntimeError(
                    f"Failed to build Qwen3.5 MoE model skeleton: {e}\n"
                    "Ensure transformers>=4.57.0 is installed."
                ) from e
        else:
            # Qwen3-30B-A3B and similar: standard CausalLM, use base init
            print("Qwen3 MoE: building standard CausalLM skeleton...")
            from airllm.airllm_base import AirLLMBaseModel
            AirLLMBaseModel.init_model(self)
            # Move rotary_emb to device so get_pos_emb_args can call it
            try:
                rotary_emb = self.model.model.rotary_emb
                rotary_emb.to(device=self.running_device, dtype=self.running_dtype)
            except Exception:
                pass
            return

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        self.model.eval()
        self.model.tie_weights()
        self.set_layers_from_layer_names()

        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.running_device,
                                        value=buffer, dtype=self.running_dtype)

        # Move rotary_emb to device so get_pos_emb_args can call it
        try:
            rotary_emb = self.model.model.language_model.rotary_emb
            rotary_emb.to(device=self.running_device, dtype=self.running_dtype)
        except Exception:
            pass

    def _get_rotary_emb(self):
        """Return the rotary embedding module regardless of model nesting."""
        try:
            return self.model.model.language_model.rotary_emb  # 397B multimodal
        except AttributeError:
            pass
        try:
            return self.model.model.rotary_emb  # 30B standard
        except AttributeError:
            return None

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        # Both Qwen3 MoE variants handle causal masking internally.
        # AirLLM's 4D causal mask causes shape errors — return empty.
        return {}

    def get_pos_emb_args(self, len_p, len_s):
        # Transformers 5.x requires position_embeddings=(cos, sin).
        # Must call rotary_emb.forward() — manual computation ignores rope_type/scaling.
        try:
            rotary_emb = self._get_rotary_emb()
            if rotary_emb is None:
                return {}
            head_dim = rotary_emb.inv_freq.shape[0] * 2
            x = torch.zeros(1, len_s, head_dim,
                            dtype=self.running_dtype, device=self.running_device)
            position_ids = torch.arange(len_p, len_p + len_s,
                                        dtype=torch.long, device=self.running_device).unsqueeze(0)
            with torch.no_grad():
                cos, sin = rotary_emb(x, position_ids)
            return {'position_embeddings': (cos, sin)}
        except Exception as e:
            print(f"get_pos_emb_args failed: {e}", flush=True)
            return {}

    def run_layer(self, layer, seq, **kwargs):
        kwargs.pop('use_cache', None)
        kwargs.pop('position_ids', None)

        # SSM (linear_attn) layers only exist in Qwen3.5-397B.
        # They are numerically unstable in fp16/bf16 — run in float32.
        layer_type = getattr(layer, 'layer_type', None)
        is_ssm = (layer_type == 'linear_attention')
        if is_ssm:
            orig_dtype = seq.dtype
            layer.float()
            seq = seq.float()
            if 'position_embeddings' in kwargs and kwargs['position_embeddings'] is not None:
                cos, sin = kwargs['position_embeddings']
                kwargs['position_embeddings'] = (cos.float(), sin.float())
            with torch.no_grad():
                out = layer(seq, **kwargs)
            layer.to(orig_dtype)
            if isinstance(out, torch.Tensor):
                out = out.to(orig_dtype)
                return out, (out,)
            out0 = out[0].to(orig_dtype)
            return out0, out
        else:
            out = layer(seq, **kwargs)
            if isinstance(out, torch.Tensor):
                return out, (out,)
            return out[0], out

    def move_layer_to_device(self, state_dict):
        """
        Override to handle transformers 5.x fused Qwen3MoeExperts.

        Cached shards are pre-fused (gate_up_proj, down_proj) so we inject
        them directly onto the Qwen3MoeExperts module, bypassing
        set_module_tensor_to_device which can't traverse the fused module.
        Non-expert keys are delegated to the base implementation.
        """
        import re

        fused_expert_re = re.compile(
            r'^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$'
        )

        remaining = {}
        expert_tensors = {}  # experts_path -> {attr: tensor}

        for k, v in state_dict.items():
            m = fused_expert_re.match(k)
            if m:
                experts_path, attr = m.group(1), m.group(2)
                expert_tensors.setdefault(experts_path, {})[attr] = v
            else:
                remaining[k] = v

        for experts_path, attrs in expert_tensors.items():
            module = self.model
            for part in experts_path.split('.'):
                module = getattr(module, part)

            for attr, tensor in attrs.items():
                setattr(module, attr, torch.nn.Parameter(
                    tensor.to(device=self.running_device, dtype=self.running_dtype),
                    requires_grad=False
                ))

        if remaining:
            return super().move_layer_to_device(remaining)
        return []

    def set_layer_names_dict(self):
        # self.config is not yet set at call time — use the raw path instead
        path = getattr(self, 'model_local_path_or_repo_id', None) or ''
        if _is_qwen35_moe_multimodal(path):
            # Qwen3.5-397B: layers nested under language_model
            self.layer_names_dict = {
                'embed': 'model.language_model.embed_tokens',
                'layer_prefix': 'model.language_model.layers',
                'norm': 'model.language_model.norm',
                'lm_head': 'lm_head',
            }
        else:
            # Qwen3-30B-A3B: standard flat layout
            self.layer_names_dict = {
                'embed': 'model.embed_tokens',
                'layer_prefix': 'model.layers',
                'norm': 'model.norm',
                'lm_head': 'lm_head',
            }


class AirLLMQwen3(AirLLMBaseModel):
    """
    AirLLM handler for Qwen3 dense models (non-MoE).
    
    Supports:
    - Qwen3ForCausalLM
    """

    def __init__(self, *args, **kwargs):
        super(AirLLMQwen3, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        # Use SDPA instead of BetterTransformer for Qwen3
        return False

    def get_generation_config(self):
        return GenerationConfig()

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }
