from .airllm_base import AirLLMBaseModel
from .utils import _wrap_forward_int64_scatter


def _ensure_qwen4_exp_indexer_scatter_int64():
    """transformers qwen4_exp writes int32 indexer indices; torch.scatter requires int64."""
    try:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer
    except ImportError:
        return
    if getattr(Qwen4ExpTextQSAIndexer.forward, '_airllm_int64_scatter', False):
        return
    Qwen4ExpTextQSAIndexer.forward = _wrap_forward_int64_scatter(Qwen4ExpTextQSAIndexer.forward)
    Qwen4ExpTextQSAIndexer.forward._airllm_int64_scatter = True


class AirLLMQwen4Exp(AirLLMBaseModel):
    """Qwen3.8-Flash-Next / Qwen4-Exp (``Qwen4ExpForConditionalGeneration``).

    Flash-Next is not the 27B dense VL. It is a 48-layer hybrid MoE (512 experts, 10 routed + 1
    shared) with Gated DeltaNet, Qwen Sparse Attention, hyper-connections, and a ~51B n-gram
    embedding (PLE) injected at one decoder layer. Transformers stores that table as
    ``ngram_embedding.shard_N.weight`` and concatenates it at load time; the modeling code already
    gathers rows on whatever device the table lives on (``_no_placement_params``).

    AirLLM therefore:

    * streams the decoder under ``model.language_model`` (there is no final RMSNorm; the output
      mix is ``hyper_connection_mixer``)
    * keeps ``model.visual`` resident, same as Qwen3.8-27B
    * constructs that ``nn.Embedding`` on the meta device (accelerate would otherwise allocate
      ~191GB of empty fp32 on CPU before moving it) then peels the table out of its decoder
      layer and file-mmaps it on the host, so ~102GB of bf16 never lands in GPU VRAM or
      anonymous RAM
    * drops ``mtp.*`` (transformers ignores the Multi-Token Prediction head)
    * works around a transformers indexer bug that scatters ``int32`` indices (torch wants
      ``int64``) so sparse-attention layers can run

    Packed MoE experts are 3D tensors on a single module, not a ``ModuleList``, so Kimi-style
    per-expert module hooks do not apply. A whole MoE layer is a few GB of bf16; that is what
    streams today.

    Needs a transformers build that includes in-tree ``qwen4_exp`` (the class is not remote code).
    Optional ``fla`` / ``causal-conv1d`` speed up DeltaNet; transformers falls back to PyTorch
    when they are missing.
    """

    def set_layer_names_dict(self):
        _ensure_qwen4_exp_indexer_scatter_int64()
        self.layer_names_dict = {
            'embed': 'model.language_model.embed_tokens',
            'layer_prefix': 'model.language_model.layers',
            'norm': 'model.language_model.hyper_connection_mixer',
            'lm_head': 'lm_head',
            'resident': [
                'model.visual',
            ],
            # Matches ``model.language_model.layers.{i}.ple.ple_embedding.ngram_embedding`` for
            # whatever layer ``ple_layer_ids`` selected. The splitter peels those tensors out of
            # the parent decoder layer so streaming that layer cannot drag ~102GB onto the GPU.
            'cpu_resident_marker': 'ple.ple_embedding.ngram_embedding',
        }
