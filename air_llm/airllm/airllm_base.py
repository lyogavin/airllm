
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, LlamaForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights

from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer, HfQuantizer

from .profiler import LayeredProfiler

try:
    from optimum.bettertransformer import BetterTransformer
    BETTERTRANSFORMER_AVAILABLE = True
except ImportError:
    BetterTransformer = None
    BETTERTRANSFORMER_AVAILABLE = False

from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path, calculate_n_layers_in_gpu

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False



try:
    from transformers.cache_utils import Cache, DynamicCache

    cache_utils_installed = True
    print('>>>> cache_utils installed')
except ImportError:
    cache_utils_installed = False






class AirLLMBaseModel(GenerationMixin):

    _is_stateful = False  # required by transformers 5.x GenerationMixin

    # customize layer names here
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                       'layer_prefix': 'model.layers',
                       'norm': 'model.norm',
                       'lm_head': 'lm_head',}



    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False, n_layers_in_gpu=None):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM.

        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        max_seq_len : int, optional
            max seq lenght, by default 512
        layer_shards_saving_path : str, optional
            optional path to save layered shards model file, by default just save to the local cache of model, subdir named splitted_model will be saved
        profiling_mode : book, optional
            if to profile the model loading time, default to False
        compression: str, optinal
            setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
        hf_token: str, optional
            huggingface api token could be provided, by default None
        """


        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self._supports_cache_class = False
        self.hf_quantizer = None

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. To use compression, please install bitsandbytes: `pip install bitsandbytes`')


        self.compression = compression
        self.hf_token = hf_token

        # Save parameters
        self.model_local_path_or_repo_id = model_local_path_or_repo_id

        self.set_layer_names_dict()


        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(model_local_path_or_repo_id,
                                                                                         layer_shards_saving_path,
                                                                                         compression=compression,
                                                                                         layer_names=self.layer_names_dict,
                                                                                         hf_token=hf_token,
                                                                                         delete_original=delete_original)
        self.running_device = device
        self.device = torch.device(self.running_device)
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        # Create model
        if hf_token is not None:
            self.config = AutoConfig.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)

        self.generation_config = self.get_generation_config()
        #print(f"using generation_config: {self.generation_config}")

        self.tokenizer = self.get_tokenizer(hf_token=hf_token)


        self.init_model()

        # get layer count:
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        layers_count = len(model_attr)


        self.layer_names = [self.layer_names_dict['embed']] + [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in
                                                               range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.main_input_name = "input_ids"

        # Auto-detect or use provided n_layers_in_gpu
        if n_layers_in_gpu is not None:
            self.n_layers_in_gpu = max(1, int(n_layers_in_gpu))
            print(f"Using manually specified n_layers_in_gpu={self.n_layers_in_gpu}")
        else:
            self.n_layers_in_gpu = calculate_n_layers_in_gpu(
                self.checkpoint_path, self.layer_names, device=device,
                layer_shards_saving_path=layer_shards_saving_path
            )

        # Fraction of free VRAM to use when dynamically recalculating per forward pass.
        # Can be overridden by callers (e.g. server.py) via model._vram_fraction = 0.75
        self._vram_fraction = 0.75

        # model weights prefetch cuda stream
        self.prefetching = prefetching

        if self.compression is not None:
            self.prefetching = False
            print(f"not support prefetching for compression for now. loading with no prepetching mode.")

        if self.n_layers_in_gpu > 1:
            self.prefetching = False
            print(f"Prefetching disabled: using multi-layer mode ({self.n_layers_in_gpu} layers per pass).")

        # this operation should run only if gpu is available
        if self.prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

    # if derived class needs to create generation config differently, like Mistrial, this function can be overridden
    def get_generation_config(self):
        # protective on generation config

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception as e:
            return GenerationConfig()

    def _update_n_layers_in_gpu(self):
        """Recalculate n_layers_in_gpu from current free VRAM before each forward
        pass. Reflects GPU memory consumed by other processes at call time.
        Uses self._vram_fraction (default 0.75) as a safety margin.
        """
        import glob
        import os as _os

        if not torch.cuda.is_available():
            return

        try:
            free_bytes, _ = torch.cuda.mem_get_info(0)
        except Exception:
            return

        # Find layer shard files in the checkpoint path
        shard_pattern = _os.path.join(self.checkpoint_path, 'model.layers.*.safetensors')
        shard_files = glob.glob(shard_pattern)
        if not shard_files:
            return

        sizes = sorted(_os.path.getsize(f) for f in shard_files)
        median_bytes = sizes[len(sizes) // 2]
        if median_bytes == 0:
            return

        usable_bytes = int(free_bytes * self._vram_fraction)
        n = max(1, int(usable_bytes // median_bytes))

        if n != self.n_layers_in_gpu:
            print(
                f"[AirLLM] VRAM: {free_bytes/1e9:.1f}GB free "
                f"({self._vram_fraction*100:.0f}% usable) "
                f"→ n_layers_in_gpu {self.n_layers_in_gpu} → {n}",
                flush=True
            )
            self.n_layers_in_gpu = n

    # a chance to customize tokenizer
    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    def get_use_better_transformer(self):
        return True

    def init_model(self):

        # try way 1 better transformers...
        # Load meta model (no memory used)
        self.model = None

        if self.get_use_better_transformer() and BETTERTRANSFORMER_AVAILABLE:
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
                    self.model = BetterTransformer.transform(self.model)  # enable flash attention
            except ValueError as ve:
                del self.model
                clean_memory()
                self.model = None

            if self.model is None:
                # try way 2.
                try:

                    print(f"new version of transfomer, no need to use BetterTransformer, try setting attn impl to sdpa...")
                    self.config.attn_implementation = "sdpa"

                    with init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(self.config, attn_implementation="sdpa", trust_remote_code=True)
                    print(f"attn imp: {type(self.model.model.layers[3].self_attn)}")

                except TypeError as ve:
                    del self.model
                    clean_memory()
                    self.model = None

        # fallback to original way
        if self.model is None:
            print(f"either BetterTransformer or attn_implementation='sdpa' is available, creating model directly")
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)

        quantization_config = getattr(self.config, "quantization_config", None)

        if quantization_config is not None:
            try:
                self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
                device_map = self.hf_quantizer.update_device_map(None)
                self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)
            except Exception as e:
                print(f"WARNING: quantizer setup failed ({e}), continuing without quantizer preprocessing. "
                      "Weights will be loaded as-is from shards.")
                self.hf_quantizer = None

        self.model.eval()
        self.model.tie_weights()

        self.set_layers_from_layer_names()

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer,
                                        dtype=self.running_dtype)

        if 'rotary_pos_emb' in self.layer_names_dict:
            # for glm keep rotary_pos_emb in gpu
            self.load_rotary_pos_emb_to_device()

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

    def load_rotary_pos_emb_to_device(self):
        state_dict = load_layer(self.checkpoint_path, self.layer_names_dict['rotary_pos_emb'])
        self.move_layer_to_device(state_dict)

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

        # pin memory:
        if self.prefetching:
            t = time.time()
            if torch.cuda.is_available():  # Check if CUDA is available
                for k in state_dict.keys():
                    state_dict[k].pin_memory()
            else:
                # For CPU, no action is needed, but you could optionally add a log or message
                print("Prefetching is enabled, but no pin_memory operation is needed for CPU.")

            elapsed_time = time.time() - t
            if self.profiling_mode:
                self.profiler.add_profiling_time('pin_memory_to_trigger_load', elapsed_time)

        return state_dict

    def move_layer_to_device(self, state_dict):
        layers = []
        for param_name, param in state_dict.items():
            if self.hf_quantizer is None:
                layers.append(param_name)
            else:
                if '.weight' in param_name:
                    layer_name = param_name[:param_name.index(".weight") + len(".weight")]
                    if layer_name not in layers:
                        layers.append(layer_name)

        for param_name in layers:
            if (self.hf_quantizer is None or
                not self.hf_quantizer.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})
               ):
                set_module_tensor_to_device(self.model, param_name, self.running_device, value=state_dict[param_name],
                                            dtype=self.running_dtype,
                                            )
            else:
                torch_dtype = self.hf_quantizer.update_torch_dtype(None)
                self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name, self.running_device, state_dict)
        return layers

    # make GenerationMixin happy
    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # AirLLM does not support the DynamicCache / Cache API introduced in transformers 4.36+
        # Treat any Cache object as None so we always run full forward passes
        if cache_utils_installed and past_key_values is not None:
            from transformers.cache_utils import Cache
            if isinstance(past_key_values, Cache):
                past_key_values = None

        if past_key_values is not None:
            past_length = self.get_past_key_values_cache_seq_len(past_key_values) #[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_past_key_values_cache_seq_len(self, past_key_values):
        return past_key_values[0][0].shape[2]
    def get_sequence_len(self, seq):
        return seq.shape[1]

    def get_pos_emb_args(self, len_p, len_s):
        return {}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {'past_key_value': (k_cache, v_cache)}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        return {'attention_mask': full_attention_mask[:, :, -len_s:, -len_p - len_s:]}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):

        return {'position_ids': full_position_ids[:, len_p:len_p + len_s]}


    def run_layer(self, layer, seq, **kwargs):
        """Run a transformer layer and return (hidden_states, layer_output_tuple).
        Override in subclasses if the layer returns a plain tensor instead of a tuple."""
        out = layer(seq, **kwargs)
        if isinstance(out, torch.Tensor):
            return out, (out,)
        return out[0], out

    def run_lm_head(self, layer, seq):
        return layer(seq).float()

    def run_norm(self, layer, seq):
        return layer(seq)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if cache_utils_installed:
            # we don't support kv cache for new version yet
            use_cache = False
            past_key_values = None  # DynamicCache from transformers 5.x is not subscriptable

        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()

        # Recalculate n_layers_in_gpu from current free VRAM before each pass
        self._update_n_layers_in_gpu()

        # Reboot the model to make sure buffers are loaded and memory is clean
        # Subclasses can set _skip_model_reinit=True to keep the skeleton alive
        # between tokens (avoids rebuilding empty skeleton on every decode step).
        if not getattr(self, '_skip_model_reinit', False):
            del self.model
            clean_memory()
            self.init_model()
        else:
            clean_memory()

        print(f"[AirLLM] forward() called — input shape: {input_ids.shape}, device: {input_ids.device}", flush=True)

        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]

        kv_cache_list = [] if use_cache else None
        if use_cache:
            for x in self.layers:
                kv_cache_list.append(([], []))
        all_hidden_states = [] * len(self.layers) if output_hidden_states else None
        all_self_attns = [] * len(self.layers) if output_attentions else None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            # Build list of (index, layer_name, layer) chunks
            layer_triples = list(zip(range(len(self.layer_names)), self.layer_names, self.layers))
            chunks = [layer_triples[s:s + self.n_layers_in_gpu]
                      for s in range(0, len(layer_triples), self.n_layers_in_gpu)]

            # Prefetch first chunk (single-layer mode only)
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])

            n_chunks = len(chunks)
            print(f"[AirLLM] Starting inference: {len(self.layer_names)} layers, {n_chunks} chunk(s), {self.n_layers_in_gpu} layer(s)/chunk", flush=True)

            for chunk_idx, chunk in enumerate(tqdm(chunks,
                              desc=f'running layer chunks({self.running_device})',
                              total=n_chunks)):

                # --- Load all layers in this chunk into GPU ---
                chunk_layer_names = [ln for _, ln, _ in chunk]
                print(f"[AirLLM] Chunk {chunk_idx+1}/{n_chunks}: loading {chunk_layer_names}", flush=True)
                chunk_moved = []  # list of (layer, moved_layer_names)
                for ci, (i, layer_name, layer) in enumerate(chunk):
                    if self.prefetching:
                        if self.profiling_mode:
                            t = time.time()
                        state_dict = future.result()
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('load_safe_tensor_cpu_wait', elapsed_time)

                        if self.profiling_mode:
                            t = time.time()
                        moved_layers = self.move_layer_to_device(state_dict)
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('create_layer_from_state_dict', elapsed_time)

                        # Kick off next layer
                        next_flat_idx = chunk[0][0] + ci + 1
                        if next_flat_idx < len(self.layer_names):
                            if self.profiling_mode:
                                t = time.time()
                            future = executor.submit(self.load_layer_to_cpu, self.layer_names[next_flat_idx])
                            if self.profiling_mode:
                                elapsed_time = time.time() - t
                                self.profiler.add_profiling_time('kick_off_load_cpu', elapsed_time)
                    else:
                        state_dict = self.load_layer_to_cpu(layer_name)
                        if self.profiling_mode:
                            t = time.time()
                        moved_layers = self.move_layer_to_device(state_dict)
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('create_layer_from_safe_tensor', elapsed_time)

                    chunk_moved.append((i, layer_name, layer, moved_layers))

                print(f"[AirLLM] Chunk {chunk_idx+1}/{n_chunks}: running layers", flush=True)
                # --- Run batch through all layers in this chunk ---
                for (i, layer_name, layer, moved_layers) in chunk_moved:
                    for j, seq in enumerate(batch):

                        if layer_name == self.layer_names_dict['embed']:
                            batch[j] = layer(seq)
                        elif layer_name == self.layer_names_dict['norm']:
                            batch[j] = self.run_norm(layer, seq)

                            if output_attentions:
                                all_hidden_states[i].append(batch[j])
                        elif layer_name == self.layer_names_dict['lm_head']:
                            batch[j] = self.run_lm_head(layer, seq)
                        else:
                            # NaN check on input
                            if torch.isnan(seq).any() or torch.isinf(seq).any():
                                print(f"[AirLLM] NaN/Inf DETECTED IN INPUT to {layer_name} (chunk {chunk_idx+1})", flush=True)

                            if output_attentions:
                                all_hidden_states[i].append(new_seq)

                            if past_key_values is not None:
                                k_cache, v_cache = past_key_values[i - 1]
                                len_p = self.get_past_key_values_cache_seq_len(past_key_values)
                                len_s = self.get_sequence_len(seq)

                                position_ids_args = self.get_position_ids_args(position_ids, len_p, len_s)
                                attention_mask_args = self.get_attention_mask_args(attention_mask, len_p, len_s)
                                past_key_value_args = self.get_past_key_value_args(k_cache, v_cache)

                                kwargs = {'use_cache': True}
                                pos_embed_args = self.get_pos_emb_args(len_p, len_s)
                                kwargs = {**kwargs, **past_key_value_args, **pos_embed_args,
                                          **attention_mask_args, **position_ids_args}

                                layer_outputs = layer(seq, **kwargs)
                                new_seq = layer_outputs[0]

                                if output_attentions:
                                    all_self_attns[i].append(layer_outputs[1])

                                if use_cache:
                                    (k_cache, v_cache) = layer_outputs[2 if output_attentions else 1]
                                    kv_cache_list[i][0].append(k_cache)
                                    kv_cache_list[i][1].append(v_cache)

                            else:
                                len_seq = self.get_sequence_len(seq)

                                pos_embed_args = self.get_pos_emb_args(0, len_seq)
                                attention_mask_args = self.get_attention_mask_args(attention_mask, 0, len_seq)
                                position_ids_args = self.get_position_ids_args(position_ids, 0, len_seq)

                                if not use_cache:
                                    kwargs = {'use_cache': False}
                                    kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
                                    new_seq, _ = self.run_layer(layer, seq, **kwargs)
                                else:
                                    kwargs = {'use_cache': True}
                                    kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
                                    layer_out = layer(seq, **kwargs)
                                    new_seq, (k_cache, v_cache) = layer_out
                                    kv_cache_list[i][0].append(k_cache)
                                    kv_cache_list[i][1].append(v_cache)

                            # NaN check on output
                            if torch.isnan(new_seq).any() or torch.isinf(new_seq).any():
                                nan_pct = torch.isnan(new_seq).float().mean().item() * 100
                                inf_pct = torch.isinf(new_seq).float().mean().item() * 100
                                print(f"[AirLLM] NaN/Inf DETECTED IN OUTPUT of {layer_name} (chunk {chunk_idx+1}): nan={nan_pct:.1f}% inf={inf_pct:.1f}%", flush=True)

                            batch[j] = new_seq

                    if output_hidden_states:
                        all_hidden_states += (torch.cat(batch, 0),)

                # --- Evict entire chunk from GPU ---
                for (i, layer_name, layer, moved_layers) in chunk_moved:
                    if self.hf_quantizer is not None:
                        for param_name in moved_layers:
                            set_module_tensor_to_device(self.model, param_name, 'meta')
                    else:
                        layer.to("meta")

                clean_memory()

        logits = torch.cat(batch, 0)
        if use_cache:
            kv_cache_list = kv_cache_list[1:-2]
            for i in range(len(kv_cache_list)):
                # print(f"{i} - {kv_cache_list[i][0].shape}")
                kv_cache_list[i] = (torch.cat(kv_cache_list[i][0], 0), torch.cat(kv_cache_list[i][1], 0))
            #print(f"returning kvcache size: {kv_cache_list[0][0].shape}")

        if output_attentions:
            all_self_attns = all_self_attns[0:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states[0:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)

        if not return_dict:
            return tuple(v for v in [logits,
                                     tuple(kv_cache_list) if kv_cache_list is not None else None,
                                     tuple(all_hidden_states) if all_hidden_states is not None else None,
                                     tuple(all_self_attns) if all_self_attns is not None else None] if v is not None)
        if self.profiling_mode:
            forward_elapsed_time = time.process_time() - forward_start
            forward_elapsed_time_wall = time.time() - forward_start_wall
            self.profiler.print_profiling_time()


            print(f"total infer process time(including all above plus gpu compute): {forward_elapsed_time:.04f}")
            print(f"total infer wall time(including all above plus gpu compute): {forward_elapsed_time_wall:.04f}")

            self.profiler.clear_profiling_time()


        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list is not None else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_hidden_states is not None else None,
        )