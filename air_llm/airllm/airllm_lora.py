"""Layer-wise LoRA training with AirLLM disk offload.

Hugging Face Trainer / PEFT keep a full autograd graph (or 4-bit weights of the
whole 27B) on the GPU. This module instead:

* reuses the per-layer shards already built for inference
* gathers embed rows and lm_head rows from CPU
* runs one decoder layer at a time, with frozen base weights streamed from disk
* keeps LoRA A/B + Adam on the GPU
* holds the autograd graph's hidden states on CPU so 64 layers do not occupy VRAM

v1 is text-only Qwen3.5 / Qwen3.8. Vision stays on meta. Dropout is 0 so the
backward recompute matches the forward.
"""

from accelerate.utils.modeling import set_module_tensor_to_device
import torch
import torch.nn.functional as F

from .airllm_qwen3_5 import AirLLMQwen3_5
from .airllm_qwen4_exp import AirLLMQwen4Exp
from .chunked_ce import chunked_linear_cross_entropy
from .lora_linear import (
    DEFAULT_LORA_TARGETS,
    FLASH_NEXT_LORA_TARGETS,
    inject_lora,
    inject_packed_expert_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
)
from .utils import clean_memory


class _StreamedModule(torch.autograd.Function):
    """One module's forward under no_grad; recompute with grad in backward.

    Inputs and outputs of ``apply`` are CPU tensors so the 64-layer graph does
    not pin 64 hidden states on the GPU. LoRA grads are written on the inner
    ``out.backward(g)`` and are *not* Function outputs.
    """

    trainer = None
    kind = None
    extras = None

    @staticmethod
    def forward(ctx, hidden):
        trainer = _StreamedModule.trainer
        kind = _StreamedModule.kind
        extras = _StreamedModule.extras
        ctx.trainer = trainer
        ctx.kind = kind
        ctx.extras = extras
        h_cpu = hidden.detach().cpu() if hidden.device.type != "cpu" else hidden.detach()
        with torch.no_grad():
            out = trainer._run_streamed(kind, h_cpu, extras, grad=False)
        ctx.save_for_backward(h_cpu)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        trainer = ctx.trainer
        h = ctx.saved_tensors[0].to(trainer.device).detach().requires_grad_(True)
        g = grad_output.to(device=trainer.device, dtype=h.dtype)
        out = trainer._run_streamed(ctx.kind, h, ctx.extras, grad=True)
        if not torch.is_tensor(out):
            out = out[0]
        try:
            out.backward(g)
        finally:
            stream_idx = trainer._stream_idx_for_kind(ctx.kind)
            trainer._evict(stream_idx)
            trainer._prefetch_stream(stream_idx - 1)
        grad_hidden = h.grad if h.grad is not None else torch.zeros_like(h)
        return grad_hidden.cpu()


class AirLLMLoRA(AirLLMQwen3_5):
    """Streamed LoRA trainer for Qwen3.5 / Qwen3.8 dense VL (text-only)."""

    def set_layer_names_dict(self):
        super().set_layer_names_dict()
        self.layer_names_dict["resident"] = []
        self.layer_names_dict["cpu_resident"] = [
            self.layer_names_dict["embed"],
            self.layer_names_dict["lm_head"],
        ]

    def __init__(
        self,
        model_local_path_or_repo_id,
        device="cuda:0",
        dtype=None,
        max_seq_len=512,
        layer_shards_saving_path=None,
        profiling_mode=False,
        compression=None,
        hf_token=None,
        prefetching=True,
        delete_original=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=None,
        lr=1e-4,
        ce_chunk_size=4096,
        weight_decay=0.0,
    ):
        super().__init__(
            model_local_path_or_repo_id,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
            layer_shards_saving_path=layer_shards_saving_path,
            profiling_mode=profiling_mode,
            compression=compression,
            hf_token=hf_token,
            prefetching=prefetching,
            delete_original=delete_original,
            install_hooks=False,
            load_resident=False,
        )
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.ce_chunk_size = ce_chunk_size
        self.target_modules = tuple(target_modules or DEFAULT_LORA_TARGETS)

        self.model.config.use_cache = False
        text_cfg = getattr(self.config, "text_config", None)
        if text_cfg is not None:
            text_cfg.use_cache = False
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.model.eval()

        self._finish_lora_setup(
            lora_r, lora_alpha, lora_dropout, lr, weight_decay, packed_experts=False)

    def _finish_lora_setup(self, lora_r, lora_alpha, lora_dropout, lr, weight_decay,
                            packed_experts=False):
        n = inject_lora(
            self._decoder_container(),
            target_modules=self.target_modules,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=self.device,
            dtype=self.running_dtype,
        )
        n_exp = 0
        if packed_experts:
            n_exp = inject_packed_expert_lora(
                self._decoder_container(),
                r=lora_r,
                alpha=lora_alpha,
                device=self.device,
                dtype=self.running_dtype,
            )
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        trainable = lora_parameters(self.model)
        n_params = sum(p.numel() for p in trainable)
        extra = f", packed-expert modules {n_exp}" if packed_experts else ""
        print(
            f"AirLLM LoRA: wrapped {n} linears{extra}, rank {lora_r}, "
            f"{n_params / 1e6:.1f}M trainable params "
            f"({sum(p.numel() * p.element_size() for p in trainable) / 1e6:.0f}MB)"
        )
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    def _module(self, dotted):
        mod = self.model
        for attr in dotted.split("."):
            mod = getattr(mod, attr)
        return mod

    def _decoder_container(self):
        return self._module(self.layer_names_dict["layer_prefix"])

    def _text_model(self):
        embed_path = self.layer_names_dict["embed"]
        return self._module(embed_path.rsplit(".", 1)[0])

    def _n_decoder_layers(self):
        return len(self.layers) - 3

    def _materialize(self, stream_idx):
        if self.prefetching and self._prefetch_future is not None and self._prefetched_idx == stream_idx:
            state = self._prefetch_future.result()
            self._prefetch_future = None
            self._prefetched_idx = None
        else:
            if self._prefetch_future is not None:
                self._prefetch_future.result()
                self._prefetch_future = None
            state = self.load_layer_to_cpu(self.layer_names[stream_idx])
        module = self.layers[stream_idx]
        module._airllm_moved = self.move_layer_to_device(state)

    def _evict(self, stream_idx):
        module = self.layers[stream_idx]
        for param_name in getattr(module, "_airllm_moved", []):
            set_module_tensor_to_device(self.model, param_name, "meta")
        module._airllm_moved = []
        clean_memory()

    def _prefetch_stream(self, stream_idx):
        if not self.prefetching or self._executor is None:
            return
        if stream_idx < 1 or stream_idx > len(self.layer_names) - 2:
            return
        self._prefetch_future = self._executor.submit(
            self.load_layer_to_cpu, self.layer_names[stream_idx])
        self._prefetched_idx = stream_idx

    def _stream_idx_for_kind(self, kind):
        if kind[0] == "decoder":
            return 1 + kind[1]
        if kind[0] == "norm":
            return len(self.layers) - 2
        raise ValueError(f"unknown streamed kind {kind}")

    def _run_decoder_layer(self, layer_i, hidden, extras):
        layer = self.layers[1 + layer_i]
        mask_map = extras.get("mask_map")
        text_cfg = getattr(self.config, "text_config", self.config)
        layer_types = getattr(text_cfg, "layer_types", None)
        if isinstance(mask_map, dict) and layer_types is not None:
            attn_mask = mask_map[layer_types[layer_i]]
        else:
            attn_mask = mask_map
        out = layer(
            hidden,
            position_embeddings=extras["position_embeddings"],
            attention_mask=attn_mask,
            position_ids=extras["position_ids"],
            past_key_values=None,
            use_cache=False,
        )
        if torch.is_tensor(out):
            return out
        return out[0]

    def _run_streamed(self, kind, hidden, extras, grad):
        """Run one streamed module. When ``grad`` is True the caller must evict after backward."""
        h = hidden.to(device=self.device, dtype=self.running_dtype)
        stream_idx = self._stream_idx_for_kind(kind)
        self._materialize(stream_idx)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            if kind[0] == "decoder":
                out = self._run_decoder_layer(kind[1], h, extras)
            elif kind[0] == "norm":
                out = self.layers[-2](h)
            else:
                raise ValueError(f"unknown streamed kind {kind}")
        if grad:
            return out
        self._evict(stream_idx)
        self._prefetch_stream(stream_idx + 1)
        return out.detach().cpu()

    def _apply_streamed(self, hidden, kind, extras):
        _StreamedModule.trainer = self
        _StreamedModule.kind = kind
        _StreamedModule.extras = extras
        try:
            return _StreamedModule.apply(hidden)
        finally:
            _StreamedModule.trainer = None
            _StreamedModule.kind = None
            _StreamedModule.extras = None

    def _make_masks(self, hidden_shape, attention_mask, text_position_ids):
        try:
            from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
        except ImportError:
            return None
        dummy = torch.empty(
            hidden_shape, device=self.device, dtype=self.running_dtype)
        cfg = getattr(self.config, "text_config", self.config)
        mask_kwargs = {
            "config": cfg,
            "inputs_embeds": dummy,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        try:
            return {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
        except Exception as e:  # noqa: BLE001 - fall back to the layer's is_causal path
            print(f"AirLLM LoRA: could not build Qwen3.5 masks ({e}); using None")
            return None

    def _embed_and_context(self, input_ids, attention_mask):
        embed = self.layers[0]
        hidden = F.embedding(input_ids.cpu(), embed.weight)
        hidden = hidden.to(dtype=self.running_dtype)
        hidden.requires_grad_(True)

        batch, seq = input_ids.shape
        pos = torch.arange(seq, device=self.device)
        pos4 = pos.view(1, 1, -1).expand(4, batch, -1)
        text_position_ids = pos4[0]
        rope_position_ids = pos4[1:]
        probe = torch.zeros(1, 1, hidden.shape[-1], device=self.device, dtype=self.running_dtype)
        position_embeddings = self._text_model().rotary_emb(probe, rope_position_ids)
        mask_map = self._make_masks(hidden.shape, attention_mask, text_position_ids)
        extras_base = {
            "position_embeddings": position_embeddings,
            "position_ids": text_position_ids,
            "mask_map": mask_map,
        }
        return hidden, extras_base

    def _vram_gb(self):
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / 1024 ** 3

    def train_step(self, input_ids, labels=None, attention_mask=None, verbose=False):
        """One SFT step. ``input_ids`` is ``[B, S]``. Returns a Python float loss."""
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [batch, seq]")
        if labels is None:
            labels = input_ids
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        hidden, extras = self._embed_and_context(input_ids, attention_mask)
        n_dec = self._n_decoder_layers()
        if verbose:
            print(f"fwd start seq={input_ids.shape[1]} peak_vram={self._vram_gb():.2f}GB", flush=True)
        for i in range(n_dec):
            hidden = self._apply_streamed(hidden, ("decoder", i), extras)
            if verbose and (i % 8 == 0 or i + 1 == n_dec):
                print(f"fwd layer {i}/{n_dec} peak_vram={self._vram_gb():.2f}GB", flush=True)
        hidden = self._apply_streamed(hidden, ("norm",), extras)
        if verbose:
            print(f"ce start peak_vram={self._vram_gb():.2f}GB", flush=True)

        shift_h = hidden[:, :-1, :]
        shift_y = labels[:, 1:]
        if attention_mask is not None:
            shift_y = shift_y.masked_fill(attention_mask[:, 1:] == 0, -100)

        weight = self.layers[-1].weight
        loss = chunked_linear_cross_entropy(
            shift_h, weight, shift_y, chunk_size=self.ce_chunk_size)
        if verbose:
            print(f"backward start loss={float(loss.detach().cpu()):.4f} peak_vram={self._vram_gb():.2f}GB", flush=True)
        loss.backward()
        self.optimizer.step()
        if verbose:
            print(f"step done peak_vram={self._vram_gb():.2f}GB", flush=True)
        return float(loss.detach().cpu())

    def save_adapter(self, path):
        torch.save(lora_state_dict(self.model), path)

    def load_adapter(self, path):
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        load_lora_state_dict(self.model, state)


class AirLLMLoRAQwen4Exp(AirLLMQwen4Exp):
    """Streamed LoRA for Qwen3.8-Flash-Next (125B MoE + 51B PLE).

    A packed MoE layer is ~5GB of bf16, so this is not a 4GB recipe — peak is
    one MoE layer plus LoRA/Adam. The n-gram table stays file-mapped on the host.
    Vision stays on meta (text-only).
    """

    def set_layer_names_dict(self):
        super().set_layer_names_dict()
        self.layer_names_dict["resident"] = []
        self.layer_names_dict["cpu_resident"] = [
            self.layer_names_dict["embed"],
            self.layer_names_dict["lm_head"],
        ]

    def __init__(
        self,
        model_local_path_or_repo_id,
        device="cuda:0",
        dtype=None,
        max_seq_len=512,
        layer_shards_saving_path=None,
        profiling_mode=False,
        compression=None,
        hf_token=None,
        prefetching=True,
        delete_original=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=None,
        lr=1e-4,
        ce_chunk_size=4096,
        weight_decay=0.0,
        packed_experts=False,
    ):
        super().__init__(
            model_local_path_or_repo_id,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
            layer_shards_saving_path=layer_shards_saving_path,
            profiling_mode=profiling_mode,
            compression=compression,
            hf_token=hf_token,
            prefetching=prefetching,
            delete_original=delete_original,
            install_hooks=False,
            load_resident=False,
        )
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.ce_chunk_size = ce_chunk_size
        self.target_modules = tuple(target_modules or FLASH_NEXT_LORA_TARGETS)

        self.model.config.use_cache = False
        text_cfg = getattr(self.config, "text_config", None)
        if text_cfg is not None:
            text_cfg.use_cache = False
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.model.eval()

        # transformers defaults MoE to grouped_mm. On a 4090 that is the Python
        # fallback, whose autograd always allocates a full-size dW for the packed
        # expert tensors (~3.4GB) even though those weights are frozen. Eager
        # F.linear on routed slices does not.
        if hasattr(self.model, "set_experts_implementation"):
            self.model.set_experts_implementation("eager")
            print("AirLLM LoRA Flash-Next: experts_implementation=eager (skip packed-expert dW)")

        # Packed-expert LoRA is off by default: 48 layers × 512 experts × r=16 is ~5.5GB of
        # adapters and ~22GB of fp32 Adam. Shared-expert / attention / hyper / PLE Linears
        # still train. Pass packed_experts=True only if you have the VRAM.
        self._finish_lora_setup(
            lora_r, lora_alpha, lora_dropout, lr, weight_decay, packed_experts=packed_experts)

    _finish_lora_setup = AirLLMLoRA._finish_lora_setup
    _module = AirLLMLoRA._module
    _decoder_container = AirLLMLoRA._decoder_container
    _text_model = AirLLMLoRA._text_model
    _n_decoder_layers = AirLLMLoRA._n_decoder_layers
    _materialize = AirLLMLoRA._materialize
    _evict = AirLLMLoRA._evict
    _prefetch_stream = AirLLMLoRA._prefetch_stream
    _stream_idx_for_kind = AirLLMLoRA._stream_idx_for_kind
    _run_streamed = AirLLMLoRA._run_streamed
    _apply_streamed = AirLLMLoRA._apply_streamed
    _vram_gb = AirLLMLoRA._vram_gb
    train_step = AirLLMLoRA.train_step
    save_adapter = AirLLMLoRA.save_adapter
    load_adapter = AirLLMLoRA.load_adapter

    def _hc_count(self):
        cfg = getattr(self.config, "text_config", self.config)
        return int(getattr(cfg, "hc_count", 1) or 1)

    def _make_masks(self, hidden_shape, attention_mask, text_position_ids):
        try:
            from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
        except ImportError:
            return None
        # Masks are built from token-width embeds (hidden), not the 4-stream hyper state.
        dummy = torch.empty(
            hidden_shape, device=self.device, dtype=self.running_dtype)
        cfg = getattr(self.config, "text_config", self.config)
        mask_kwargs = {
            "config": cfg,
            "inputs_embeds": dummy,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": text_position_ids,
            "allow_is_causal_skip": False,
        }
        try:
            return {
                "qwen_sparse_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
        except Exception as e:  # noqa: BLE001
            print(f"AirLLM LoRA Flash-Next: could not build masks ({e}); using None")
            return None

    def _embed_and_context(self, input_ids, attention_mask):
        embed = self.layers[0]
        hidden = F.embedding(input_ids.cpu(), embed.weight)
        hidden = hidden.to(dtype=self.running_dtype)
        hidden = hidden.repeat(1, 1, self._hc_count())
        hidden.requires_grad_(True)

        batch, seq = input_ids.shape
        pos = torch.arange(seq, device=self.device)
        pos4 = pos.view(1, 1, -1).expand(4, batch, -1)
        text_position_ids = pos4[0]
        rope_position_ids = pos4[1:]
        probe = torch.zeros(
            1, 1, embed.weight.shape[-1], device=self.device, dtype=self.running_dtype)
        position_embeddings = self._text_model().rotary_emb(probe, rope_position_ids)
        token_shape = (batch, seq, embed.weight.shape[-1])
        mask_map = self._make_masks(token_shape, attention_mask, text_position_ids)
        conv_mask = None
        if isinstance(mask_map, dict):
            conv_mask = mask_map.get("linear_attention")
        ple_ids = input_ids
        if conv_mask is not None:
            cfg = getattr(self.config, "text_config", self.config)
            eos = getattr(cfg, "eos_token_id", None)
            if isinstance(eos, (list, tuple)):
                eos = eos[0] if eos else None
            if eos is not None:
                try:
                    ple_ids = torch.where(conv_mask.bool(), input_ids, eos)
                except RuntimeError:
                    ple_ids = input_ids
        return hidden, {
            "position_embeddings": position_embeddings,
            "position_ids": text_position_ids,
            "mask_map": mask_map,
            "conv_mask": conv_mask,
            "ple_input_ids": ple_ids,
        }

    def _run_decoder_layer(self, layer_i, hidden, extras):
        layer = self.layers[1 + layer_i]
        mask_map = extras.get("mask_map")
        if isinstance(mask_map, dict):
            attn_mask = mask_map.get("qwen_sparse_attention")
        else:
            attn_mask = mask_map
        out = layer(
            hidden,
            position_embeddings=extras["position_embeddings"],
            attention_mask=attn_mask,
            conv_mask=extras.get("conv_mask"),
            past_key_values=None,
            ple_input_ids=extras.get("ple_input_ids"),
            use_cache=False,
        )
        if torch.is_tensor(out):
            return out
        return out[0]
