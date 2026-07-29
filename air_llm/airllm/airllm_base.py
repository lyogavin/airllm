
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from .profiler import LayeredProfiler

from .utils import clean_memory, load_layer, layer_tensor_names, load_layer_subset, \
    find_or_create_local_splitted_path
from .persist import ModelPersister

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False


# Helpers that transformers 5.0 moved out of transformers.utils.generic. Remote model code is
# routinely written against an older transformers and still imports them from the old location,
# which makes such models fail to import at all. Re-exporting is enough to load them.
_RELOCATED_TRANSFORMERS_SYMBOLS = {
    'OutputRecorder': 'transformers.utils.output_capturing',
    'check_model_inputs': 'transformers.utils.output_capturing',
}


def restore_relocated_transformers_symbols():
    """Re-export moved transformers helpers under their old names, where they are missing."""
    import importlib
    import transformers.utils.generic as generic

    for name, new_home in _RELOCATED_TRANSFORMERS_SYMBOLS.items():
        if hasattr(generic, name):
            continue
        try:
            module = importlib.import_module(new_home)
        except ImportError:
            continue
        symbol = getattr(module, name, None)
        if symbol is not None:
            setattr(generic, name, symbol)


class AirLLMBaseModel:
    """
    Memory-frugal wrapper around a Hugging Face ``*ForCausalLM`` model.

    The checkpoint is split into per-layer shards on disk. The real transformers model is
    instantiated on the ``meta`` device (no memory used) and owns the full forward / generation
    logic. AirLLM only attaches forward hooks to each big module (embeddings, every decoder
    layer, the final norm and the lm_head) to stream that module's weights disk -> GPU right
    before it runs and free them right after, prefetching the next module on a worker thread.

    Because transformers drives the forward pass, AirLLM no longer needs to track per-architecture
    attention/rotary/cache details: new model architectures work as soon as transformers supports
    them.
    """

    # Upper bound on how much pinned (page-locked) host memory a single prefetched layer may use.
    # Layers larger than this are loaded into ordinary pageable memory instead.
    max_pinned_layer_bytes = 2 * 1024 ** 3

    # Subclasses override this to point at non-standard module names.
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                                 'layer_prefix': 'model.layers',
                                 'norm': 'model.norm',
                                 'lm_head': 'lm_head'}

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False):
        """
        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            runtime dtype; defaults to the model's own config.torch_dtype (usually bfloat16 for
            modern models). float16 has too narrow a range for very deep models and overflows to
            inf/NaN, which silently corrupts the output, so we don't force it.
        max_seq_len : int, optional
            max seq length, by default 512
        layer_shards_saving_path : str, optional
            optional path to save the splitted shards, by default next to the model cache
        profiling_mode : bool, optional
            whether to profile the model loading time, default False
        compression: str, optional
            '4bit' or '8bit' to enable block-wise quantization of the on-disk shards
        hf_token: str, optional
            huggingface api token
        prefetching: bool, optional
            overlap the next layer's disk load with the current layer's compute
        delete_original: bool, optional
            delete the original downloaded checkpoint after splitting to save disk space
        """

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self.hf_quantizer = None

        if compression is not None and not bitsandbytes_installed:
            raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. '
                              'To use compression, please install bitsandbytes: `pip install bitsandbytes`')

        self.compression = compression
        self.hf_token = hf_token

        restore_relocated_transformers_symbols()

        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=compression,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original)

        self.running_device = device
        self.device = torch.device(self.running_device)

        # Prefer transformers' native implementation; only trust the model's bundled remote code when
        # transformers doesn't recognize the architecture. Vendored remote code is frequently pinned
        # to an old transformers and breaks against the current cache/generation APIs (e.g.
        # DeepSeek-V2's modeling_deepseek.py calls the long-removed DynamicCache.seen_tokens).
        token_kwargs = {'token': hf_token} if hf_token is not None else {}
        try:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=False, **token_kwargs)
            self.trust_remote_code = False
        except Exception:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=True, **token_kwargs)
            self.trust_remote_code = True

        # Default to the model's native dtype (bf16 for most modern models). Forcing fp16 overflows
        # on deep models (e.g. Qwen3-235B's 94 layers) and produces garbage; bf16's wider range
        # avoids it. Users can still override via dtype=.
        if dtype is None:
            cfg_dtype = getattr(self.config, "torch_dtype", None)
            if isinstance(cfg_dtype, str):
                cfg_dtype = getattr(torch, cfg_dtype, None)
            dtype = cfg_dtype if isinstance(cfg_dtype, torch.dtype) else torch.float16
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        self.generation_config = self.get_generation_config()
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        # prefetch executor / state
        self.prefetching = prefetching
        if self.compression is not None and self.prefetching:
            print("prefetching is not supported together with compression for now; disabling prefetching.")
            self.prefetching = False
        self._executor = ThreadPoolExecutor(max_workers=1) if self.prefetching else None
        self._prefetch_future = None
        self._prefetched_idx = None

        self.init_model()

        # compute layer count from the instantiated model
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        layers_count = len(model_attr)

        self.layer_names = [self.layer_names_dict['embed']] + \
                           [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.set_layers_from_layer_names()
        self._load_resident_modules()
        self._install_streaming_hooks()

    # ---- customization hooks for subclasses -------------------------------------------------

    def get_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    # ---- model construction -----------------------------------------------------------------

    def _propagate_attn_implementation(self, impl):
        """Push the attention choice down into nested sub-configs.

        Multimodal wrappers keep the real decoder under a sub-config -- Kimi K3 uses ``text_config``
        -- and transformers records the request only on the config it was handed. The sub-model then
        reads an unset value and falls through to a flash-attention path, which fails outright on a
        machine without flash-attn installed.
        """
        from transformers import PretrainedConfig

        def walk(cfg, depth=0):
            if depth > 2:
                return
            for sub in vars(cfg).values():
                if isinstance(sub, PretrainedConfig):
                    sub._attn_implementation = impl
                    walk(sub, depth + 1)

        walk(self.config)

    def init_model(self):
        # Build the real model on meta (no memory). include_buffers=False so non-persistent
        # buffers such as rotary inv_freq are actually computed (they aren't in the checkpoint).
        self.model = None
        try:
            self._propagate_attn_implementation("sdpa")
            with init_empty_weights(include_buffers=False):
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="sdpa", trust_remote_code=self.trust_remote_code)
        except (ValueError, TypeError) as e:
            print(f"attn_implementation='sdpa' not available ({e}), falling back to eager attention")
            self.model = None
        if self.model is None:
            # Some (often remote-code) architectures don't support sdpa and also default to it, so we
            # must request eager explicitly; otherwise transformers re-selects sdpa and errors again.
            self._propagate_attn_implementation("eager")
            with init_empty_weights(include_buffers=False):
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="eager", trust_remote_code=self.trust_remote_code)

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is None:
            # Nested multimodal configs (Kimi K3) keep it under text_config.
            quantization_config = getattr(getattr(self.config, "text_config", None),
                                          "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)
            # compressed-tensors registers a hook that expands every packed module on the first
            # forward. That undoes per-expert streaming (a K3 layer becomes ~56GB) and is also
            # unnecessary: we decompress each expert ourselves as it loads. Remove the hook.
            hook = getattr(self.model, "ct_decompress_hook", None)
            if hook is not None:
                hook.remove()
                delattr(self.model, "ct_decompress_hook")

        self.model.eval()
        self.model.tie_weights()
        self.model.generation_config = self.generation_config

        # Move all (already-materialized) buffers to the running device, preserving their dtype.
        # This includes rotary inv_freq, which transformers computes once at the model level and
        # passes down to every decoder layer.
        for buffer_name, buffer in self.model.named_buffers():
            if buffer is not None and buffer.device.type != 'meta':
                set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer)

        # Force the model to report the running (cuda) device even though its parameters live on
        # meta between layer executions, so transformers' generation utilities place inputs/cache
        # tensors on the right device.
        self._patch_device_property()

    def _patch_device_property(self):
        running_device = torch.device(self.running_device)
        running_dtype = self.running_dtype
        base_cls = type(self.model)

        # transformers >= 4.50 removed GenerationMixin from PreTrainedModel, so model classes that
        # predate that change (or ship as remote code, like Kimi K3's multimodal wrapper) no longer
        # have .generate(). They still define prepare_inputs_for_generation, so mixing the class
        # back in restores generation. It must come after the model class, per transformers.
        extra_bases = () if isinstance(self.model, GenerationMixin) else (GenerationMixin,)
        if extra_bases:
            print(f"{base_cls.__name__} does not inherit GenerationMixin; mixing it in so "
                  f"generate() works.")

        class _AirLLMRuntimeModel(base_cls, *extra_bases):
            @property
            def device(self):
                return running_device

            @property
            def dtype(self):
                return running_dtype

        self.model.__class__ = _AirLLMRuntimeModel

    def set_layers_from_layer_names(self):
        self.layers = []

        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    # ---- weight streaming -------------------------------------------------------------------

    def load_layer_to_cpu(self, layer_name):
        t = time.time()
        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time
            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)
            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        if self.prefetching and torch.cuda.is_available():
            # pin_memory() returns a pinned copy rather than pinning in place, so the result has to
            # be kept for the faster host->device copy to actually happen. Pinned memory can't be
            # paged out, so we only spend it on layers small enough to be safe: a frontier MoE
            # checkpoint has ~17GB layers, and with prefetching two are in flight at once, which
            # would lock up ~34GB of RAM for a copy that is dwarfed by the disk read anyway.
            total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
            if total_bytes <= self.max_pinned_layer_bytes:
                try:
                    for k in state_dict.keys():
                        state_dict[k] = state_dict[k].pin_memory()
                except RuntimeError:
                    # Out of pinned memory: fall back to pageable, which is slower but always works.
                    pass

        return state_dict

    def _restore_plain_weight_modules(self, state_dict):
        """Undo CT's packed-parameter layout when the checkpoint still ships a plain ``weight``.

        Some checkpoints (Kimi K3) list residual/router Linears under the MXFP4 target list but
        store them as bf16. After ``preprocess_model`` those modules expose ``weight_packed`` and
        reject the real ``weight`` tensor. Restore a meta ``weight`` Parameter so the shard can load.
        """
        plain = {k[:-len('.weight')] for k in state_dict if k.endswith('.weight')}
        packed = {k[:-len('.weight_packed')] for k in state_dict if k.endswith('.weight_packed')}
        for prefix in plain - packed:
            try:
                module = self.model.get_submodule(prefix)
            except AttributeError:
                continue
            names = list(module._parameters.keys())
            if 'weight' in names or 'weight_packed' not in names:
                continue
            weight = state_dict[f'{prefix}.weight']
            for name in names:
                if name == 'weight' or name.startswith('weight_'):
                    module._parameters.pop(name, None)
            module.register_parameter(
                'weight',
                torch.nn.Parameter(torch.empty(weight.shape, device='meta', dtype=weight.dtype),
                                   requires_grad=False),
            )
            for attr in ('quantization_scheme', 'quantization_status', 'quantization_format'):
                if hasattr(module, attr):
                    delattr(module, attr)

    def _decompress_state_dict(self, state_dict):
        """Expand packed payloads into the plain weights the modules expect.

        compressed-tensors has no quantized compute kernels: it registers a hook that decompresses
        the whole model before the first forward, after which every quantized module wants a plain
        ``weight``. Our shards still hold ``weight_packed``/``weight_scale``, so we expand them here.

        The expansion happens on the GPU, after transferring the *packed* bytes. For MXFP4 that
        moves 4x less data across PCIe than transferring an already-expanded weight would.
        """
        if self.hf_quantizer is None:
            return state_dict

        packed_prefixes = {k[: -len('.weight_packed')]
                           for k in state_dict if k.endswith('.weight_packed')}
        if not packed_prefixes:
            return state_dict

        from compressed_tensors.compressors.base import BaseCompressor
        try:
            from compressed_tensors.linear.compressed_linear import CompressedLinear
        except ImportError:
            CompressedLinear = None

        out = dict(state_dict)
        for prefix in packed_prefixes:
            try:
                module = self.model.get_submodule(prefix)
            except AttributeError:
                continue
            # CompressedLinear's own forward consumes the packed payload, so leave it packed.
            if CompressedLinear is not None and isinstance(module, CompressedLinear):
                continue
            scheme = getattr(module, 'quantization_scheme', None)
            if scheme is None or getattr(scheme, 'format', None) is None:
                continue

            local = {k[len(prefix) + 1:]: v.to(self.running_device)
                     for k, v in state_dict.items() if k.startswith(prefix + '.')}
            # scheme.format is an enum on some compressed-tensors versions and a plain str on others.
            fmt = getattr(scheme.format, 'value', scheme.format)
            compressor = BaseCompressor.get_value_from_registry(fmt)
            decompressed = compressor.decompress(local, scheme)

            for k in local:
                out.pop(f'{prefix}.{k}', None)
            if 'weight' in decompressed:
                # Once expanded, the module reads only `weight`; the scales that came back merely
                # describe how the checkpoint stored it, and keeping them would waste VRAM.
                out[f'{prefix}.weight'] = decompressed['weight']
                self._expose_plain_weight(module, decompressed['weight'])
            else:
                for k, v in decompressed.items():
                    out[f'{prefix}.{k}'] = v
        return out

    def _expose_plain_weight(self, module, weight):
        """Swap a module's packed parameters for the plain ``weight`` its forward reads.

        compressed-tensors patches a ``quantized_forward`` onto quantized Linears that reads
        ``self.weight``; the packed parameters only describe how the checkpoint stores the value.
        Marking the module COMPRESSED tells that forward the weight is already on the quantization
        grid, so it skips a fake-quantize that would only reproduce what we just decompressed.
        """
        existing = module._parameters.get('weight')
        if existing is None or existing.shape != weight.shape:
            for name in [n for n in list(module._parameters) if n.startswith('weight')]:
                module._parameters.pop(name, None)
            module.register_parameter(
                'weight',
                torch.nn.Parameter(torch.empty(weight.shape, device='meta', dtype=weight.dtype),
                                   requires_grad=False),
            )
        try:
            from compressed_tensors.quantization import QuantizationStatus
        except ImportError:
            return
        module.quantization_status = QuantizationStatus.COMPRESSED

    def _adopt_checkpoint_shape(self, param_name, value):
        """Resize a meta placeholder whose shape disagrees with the checkpoint.

        A model class builds its parameters from the config, and that construction can disagree
        with the weights actually shipped. Kimi K3 sizes ``A_log`` from ``num_heads`` while its
        checkpoint stores one entry per head channel, which would abort the load. For a parameter
        still on meta -- i.e. one we have never materialised -- the checkpoint is the source of
        truth, so adopt its shape.
        """
        module_path, _, attr = param_name.rpartition('.')
        try:
            module = self.model.get_submodule(module_path) if module_path else self.model
        except AttributeError:
            return
        current = module._parameters.get(attr)
        if current is None or current.device.type != 'meta' or current.shape == value.shape:
            return

        if not hasattr(self, '_shape_adoption_warned'):
            self._shape_adoption_warned = set()
        if attr not in self._shape_adoption_warned:
            self._shape_adoption_warned.add(attr)
            print(f"{attr}: checkpoint ships {tuple(value.shape)} but the model class builds "
                  f"{tuple(current.shape)}; using the checkpoint shape.")

        module.register_parameter(
            attr,
            torch.nn.Parameter(torch.empty(value.shape, device='meta', dtype=current.dtype),
                               requires_grad=False),
        )

    def move_layer_to_device(self, state_dict):
        self._restore_plain_weight_modules(state_dict)
        state_dict = self._decompress_state_dict(state_dict)
        moved = []
        for param_name in self._param_names_from_state_dict(state_dict):
            if self.hf_quantizer is not None and self._needs_quantization(param_name):
                # On-the-fly-quantizing schemes (e.g. bitsandbytes) reconstruct the param from the
                # weight plus companion quant-state tensors carried in state_dict.
                self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name,
                                                         self.running_device, state_dict)
            else:
                # Normal load. Only ordinary high-precision tensors get cast to the runtime dtype;
                # pre-quantized payloads must be placed verbatim (see _should_load_verbatim).
                value = state_dict[param_name]
                self._adopt_checkpoint_shape(param_name, value)
                if self._should_load_verbatim(param_name, value):
                    set_module_tensor_to_device(self.model, param_name, self.running_device, value=value)
                else:
                    set_module_tensor_to_device(self.model, param_name, self.running_device,
                                                value=value, dtype=self.running_dtype)
            moved.append(param_name)
        return moved

    # Suffixes of the companion tensors that pre-quantized checkpoints ship alongside a weight
    # (fp8 block scales, compressed-tensors/MXFP4 packed payloads and their scales, GPTQ indices).
    _QUANT_COMPANION_SUFFIXES = ("_scale", "_scale_inv", "_packed", "_zero_point", "_g_idx", "_shape")

    def _should_load_verbatim(self, param_name, value):
        """Whether a checkpoint tensor must be placed on the device without a dtype cast.

        Casting a pre-quantized payload to the runtime dtype destroys it: an fp8 weight loses its
        quantization, and a 4-bit MXFP4 ``weight_packed`` tensor (stored as packed integers) becomes
        meaningless floats. Equally important for very large models, decompressing on load would
        multiply a layer's footprint by ~4x, which is what keeps Kimi-K3-class checkpoints from
        fitting on a single GPU. So we keep anything that isn't a plain high-precision float as-is.
        """
        if not value.is_floating_point():
            # Packed 4-bit payloads, zero points, g_idx, shape metadata.
            return True
        if value.element_size() == 1:
            # Any 8-bit float: fp8 e4m3/e5m2 weights, and the e8m0 scales MXFP4 uses.
            return True
        return param_name.endswith(self._QUANT_COMPANION_SUFFIXES)

    def _needs_quantization(self, param_name):
        q = self.hf_quantizer
        # transformers renamed check_quantized_param -> param_needs_quantization.
        if hasattr(q, "param_needs_quantization"):
            return q.param_needs_quantization(self.model, param_name)
        return q.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})

    def _param_names_from_state_dict(self, state_dict):
        names = []
        for param_name in state_dict.keys():
            # bitsandbytes stores a weight plus companion quant-state tensors named
            # "<weight>.4bit.*" / "<weight>.8bit.*"; those are reconstructed together via
            # create_quantized_param, so collapse them down to the base weight name. Everything
            # else (including fp8 weight + weight_scale_inv pairs) is kept as distinct params.
            if '.4bit.' in param_name or '.8bit.' in param_name:
                base = param_name.split('.4bit.')[0].split('.8bit.')[0]
                if base not in names:
                    names.append(base)
            elif param_name not in names:
                names.append(param_name)
        return names

    def _load_resident_modules(self):
        """Load modules that sit outside the streamed embed -> layers -> norm -> lm_head sequence.

        Multimodal checkpoints carry a vision tower and projector, and some architectures add
        extra top-level norms. They never get a streaming hook, so without this they would stay on
        the meta device and fail the moment they run. They are small (well under a GB), so we load
        them once and leave them resident.
        """
        for name in self.layer_names_dict.get('resident', []):
            try:
                state_dict = self.load_layer_to_cpu(name)
            except FileNotFoundError:
                # Not every checkpoint of a given architecture ships every optional module.
                continue
            self.move_layer_to_device(state_dict)

    def _install_streaming_hooks(self):
        # Modules execute in this order during a forward: embed -> layers -> norm -> lm_head.
        n = len(self.layer_names)

        # Detect tied input/output embeddings. When tied, lm_head shares the embedding weight, so
        # there is no separate lm_head shard. We keep the embedding resident on the GPU (it is the
        # only copy and such models are small) and re-tie lm_head to it, then stream only the
        # decoder layers and the final norm.
        self.tie_word_embeddings = bool(getattr(self.config, "tie_word_embeddings", False))

        if self.tie_word_embeddings:
            embed_state = self.load_layer_to_cpu(self.layer_names[0])
            self.move_layer_to_device(embed_state)
            self.model.tie_weights()
            self._streamed_indices = list(range(1, n - 1))  # decoder layers + final norm
        else:
            self._streamed_indices = list(range(n))

        self._streamed_set = set(self._streamed_indices)

        self._setup_expert_streaming()

        for idx in self._streamed_indices:
            module = self.layers[idx]
            module._airllm_idx = idx
            module.register_forward_pre_hook(self._pre_hook)
            module.register_forward_hook(self._post_hook)

    # ---- per-expert streaming ---------------------------------------------------------------

    def _setup_expert_streaming(self):
        """Stream individual MoE experts instead of whole decoder layers, where that is possible.

        A sparse MoE layer holds hundreds of experts but routes each token to a handful of them.
        Materialising the whole layer is therefore enormously wasteful: for Kimi K3 a layer's
        experts are ~55GB expanded, of which a token touches ~1GB. Because the model calls each
        selected expert as its own module (and skips unselected ones), a forward hook per expert
        loads exactly the experts that actually run.

        This needs `expert_prefix` in layer_names_dict and safetensors shards, since it relies on
        reading individual tensors out of a shard.
        """
        self._expert_streaming = False
        self._expert_keys = {}
        self._non_expert_keys = {}

        expert_prefix = self.layer_names_dict.get('expert_prefix')
        if not expert_prefix:
            return
        if type(ModelPersister.get_model_persister()).__name__ != 'SafetensorModelPersister':
            return

        layer_prefix = self.layer_names_dict['layer_prefix']
        hooked = 0

        for idx in self._streamed_indices:
            layer_name = self.layer_names[idx]
            if not layer_name.startswith(layer_prefix + '.'):
                continue
            try:
                names = layer_tensor_names(self.checkpoint_path, layer_name)
            except Exception:
                continue

            marker = f'.{expert_prefix}.'
            per_expert = {}
            others = []
            for key in names:
                pos = key.find(marker)
                if pos == -1:
                    others.append(key)
                    continue
                rest = key[pos + len(marker):]
                head = rest.split('.', 1)[0]
                if not head.isdigit():
                    others.append(key)
                    continue
                per_expert.setdefault(int(head), []).append(key)

            if not per_expert:
                continue

            layer_module = self.layers[idx]
            experts_container = layer_module
            try:
                for attr in expert_prefix.split('.'):
                    experts_container = getattr(experts_container, attr)
            except AttributeError:
                continue

            self._non_expert_keys[idx] = others
            self._expert_keys[idx] = per_expert

            for expert_idx, keys in per_expert.items():
                if expert_idx >= len(experts_container):
                    continue
                expert_module = experts_container[expert_idx]
                expert_module._airllm_expert = (idx, expert_idx)
                expert_module.register_forward_pre_hook(self._expert_pre_hook)
                expert_module.register_forward_hook(self._expert_post_hook)
                hooked += 1

        if hooked:
            self._expert_streaming = True
            n_layers = len(self._expert_keys)
            print(f"per-expert streaming enabled: {hooked} experts across {n_layers} layers "
                  f"load on demand, so only the experts a token routes to are materialised.")

    def _expert_pre_hook(self, module, args):
        layer_idx, expert_idx = module._airllm_expert
        keys = self._expert_keys[layer_idx][expert_idx]
        state_dict = load_layer_subset(self.checkpoint_path, self.layer_names[layer_idx], keys)
        module._airllm_moved = self.move_layer_to_device(state_dict)

    def _expert_post_hook(self, module, args, output):
        for param_name in getattr(module, '_airllm_moved', []):
            set_module_tensor_to_device(self.model, param_name, 'meta')
        module._airllm_moved = []
        return output

    def _next_streamed_idx(self, idx):
        nxt = idx + 1
        return nxt if nxt in self._streamed_set else None

    def _load_streamed_layer(self, idx):
        """Load one streamed module's weights. Experts are excluded when they stream themselves."""
        keys = self._non_expert_keys.get(idx) if getattr(self, '_expert_streaming', False) else None
        if keys is None:
            return self.load_layer_to_cpu(self.layer_names[idx])
        return load_layer_subset(self.checkpoint_path, self.layer_names[idx], keys)

    def _pre_hook(self, module, args):
        idx = module._airllm_idx

        if self.prefetching and self._prefetch_future is not None and self._prefetched_idx == idx:
            state_dict = self._prefetch_future.result()
            self._prefetch_future = None
        else:
            state_dict = self._load_streamed_layer(idx)

        module._airllm_moved = self.move_layer_to_device(state_dict)

        if self.prefetching:
            nxt = self._next_streamed_idx(idx)
            if nxt is not None:
                self._prefetch_future = self._executor.submit(self._load_streamed_layer, nxt)
                self._prefetched_idx = nxt

    def _post_hook(self, module, args, output):
        # module.to('meta') would also evict the experts, which manage their own lifetime, so with
        # expert streaming we only release exactly what this hook placed.
        if self.hf_quantizer is not None or getattr(self, '_expert_streaming', False):
            for param_name in getattr(module, '_airllm_moved', []):
                set_module_tensor_to_device(self.model, param_name, 'meta')
        else:
            module.to('meta')
        clean_memory()
        return output

    # ---- delegation to the underlying transformers model ------------------------------------

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
