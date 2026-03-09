
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
    bettertransformer_available = True
except ImportError:
    bettertransformer_available = False

from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path

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
    _is_stateful = False

    # customize layer names here
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                       'layer_prefix': 'model.layers',
                       'norm': 'model.norm',
                       'lm_head': 'lm_head',}



    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False, layers_per_batch="auto"):
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
        layers_per_batch: str or int, optional
            number of layers to load onto GPU simultaneously before computing and cleaning up.
            "auto" (default) calculates based on available GPU memory. Set to 1 for original behavior.
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

        # model weights prefetch cuda stream
        self.prefetching = prefetching

        if self.compression is not None:
            self.prefetching = False
            print(f"not support prefetching for compression for now. loading with no prepetching mode.")

        # this operation should run only if gpu is available
        if prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        self.layers_per_batch = layers_per_batch

    # if derived class needs to create generation config differently, like Mistrial, this function can be overridden
    def get_generation_config(self):
        # protective on generation config

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception as e:
            return GenerationConfig()

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

        if self.get_use_better_transformer():
            if bettertransformer_available:
                try:
                    with init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
                        self.model = BetterTransformer.transform(self.model)  # enable flash attention
                    self._init_strategy = 'better_transformer'
                except (ValueError, Exception) as ve:
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
                    self._init_strategy = 'sdpa'

                except (TypeError, Exception) as ve:
                    del self.model
                    clean_memory()
                    self.model = None

        # fallback to original way
        if self.model is None:
            print(f"either BetterTransformer or attn_implementation='sdpa' is available, creating model directly")
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
            self._init_strategy = 'default'

        self._finalize_model_init()

    def _init_model_fast(self):
        """Fast model recreation using cached strategy (no trial-and-error)."""
        if self._init_strategy == 'better_transformer':
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
                self.model = BetterTransformer.transform(self.model)
        elif self._init_strategy == 'sdpa':
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, attn_implementation="sdpa", trust_remote_code=True)
        else:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
        self._finalize_model_init()

    def _finalize_model_init(self):
        """Common model initialization steps."""
        quantization_config = getattr(self.config, "quantization_config", None)

        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model = self.model, device_map = device_map)

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

    def _reset_model_for_forward(self):
        """Lightweight model reset between forward passes.
        Reuses the existing model skeleton instead of full recreation.
        After forward cleanup, all layers are on 'meta' (no GPU memory).
        We just need to restore buffers and re-setup layer references."""
        clean_memory()
        self.set_layers_from_layer_names()
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer,
                                        dtype=self.running_dtype)
        if 'rotary_pos_emb' in self.layer_names_dict:
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

    def _estimate_layer_gpu_bytes(self):
        """Estimate GPU memory per layer by measuring first transformer layer."""
        if len(self.layer_names) < 3:
            return 0
        # Use first transformer layer (index 1, after embed) as representative
        layer_name = self.layer_names[1]
        state_dict = load_layer(self.checkpoint_path, layer_name)
        total_bytes = sum(t.element_size() * t.nelement() for t in state_dict.values())
        del state_dict
        return total_bytes

    def _calculate_layers_per_batch(self):
        """Calculate how many layers can fit in GPU memory simultaneously."""
        if not torch.cuda.is_available():
            return 1
        layer_bytes = self._estimate_layer_gpu_bytes()
        if layer_bytes == 0:
            return 1
        free_bytes, _ = torch.cuda.mem_get_info(self.running_device)
        # Reserve 40% of free memory for activations, attention masks, and CUDA overhead
        usable_bytes = int(free_bytes * 0.6)
        batch_size = max(1, usable_bytes // layer_bytes)
        return batch_size

    def _load_batch_to_cpu(self, layer_names):
        """Load a batch of layers to CPU memory."""
        return [self.load_layer_to_cpu(name) for name in layer_names]

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
                not self.hf_quantizer.param_needs_quantization(self.model, param_name)
               ):
                set_module_tensor_to_device(self.model, param_name, self.running_device, value=state_dict[param_name],
                                            dtype=self.running_dtype,
                                            )
            else:
                # Weights are already quantized (uint8 + quant_state metadata).
                # Reconstruct Params4bit directly instead of re-quantizing.
                quant_state_dict = {k[len(param_name) + 1:]: v for k, v in state_dict.items()
                                    if k.startswith(param_name + ".") and k != param_name}
                quant_state = bnb.functional.QuantState.from_dict(qs_dict=quant_state_dict, device=self.running_device)
                new_value = bnb.nn.Params4bit(state_dict[param_name].to(self.running_device),
                                              requires_grad=False,
                                              quant_state=quant_state,
                                              bnb_quantized=True)
                # Set directly on module to avoid accelerate re-creating Params4bit
                parts = param_name.split(".")
                module = self.model
                for part in parts[:-1]:
                    module = getattr(module, part)
                setattr(module, parts[-1], new_value)
        return layers

    # make GenerationMixin happy
    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
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
        if cache_utils_installed and isinstance(past_key_values, DynamicCache):
            return past_key_values.get_seq_length()
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
            past_key_values = None

        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()

        # Reset model for forward pass - reuse skeleton if already initialized
        if hasattr(self, '_init_strategy'):
            self._reset_model_for_forward()
        else:
            del self.model
            clean_memory()
            self.init_model()

        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]

        # Check if we need to compute position embeddings for new transformers versions
        self._rotary_emb = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
            self._rotary_emb = self.model.model.rotary_emb

        kv_cache_list = [] if use_cache else None
        if use_cache:
            for x in self.layers:
                kv_cache_list.append(([], []))
        all_hidden_states = [] * len(self.layers) if output_hidden_states else None
        all_self_attns = [] * len(self.layers) if output_attentions else None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            # Calculate layers per batch for multi-layer GPU loading (cached after first call)
            if not hasattr(self, '_cached_layers_per_batch'):
                if self.layers_per_batch == "auto":
                    self._cached_layers_per_batch = self._calculate_layers_per_batch()
                else:
                    self._cached_layers_per_batch = self.layers_per_batch
            layers_per_batch = self._cached_layers_per_batch

            total_layers = len(self.layer_names)
            print(f"Processing {total_layers} layers in batches of {layers_per_batch}")

            # Build batch boundaries
            batch_ranges = []
            for start in range(0, total_layers, layers_per_batch):
                end = min(start + layers_per_batch, total_layers)
                batch_ranges.append((start, end))

            # Prefetch first batch to CPU
            if self.prefetching and len(batch_ranges) > 0:
                first_start, first_end = batch_ranges[0]
                first_names = self.layer_names[first_start:first_end]
                future = executor.submit(self._load_batch_to_cpu, first_names)

            pbar = tqdm(total=total_layers, desc=f'running layers({self.running_device})')

            for b_idx, (b_start, b_end) in enumerate(batch_ranges):
                batch_size_layers = b_end - b_start
                batch_names = self.layer_names[b_start:b_end]

                # === Phase 1: Load batch to CPU ===
                if self.prefetching:
                    if self.profiling_mode:
                        t = time.time()
                    batch_state_dicts = future.result()
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('load_safe_tensor_cpu_wait', elapsed_time)

                    # Kick off next batch prefetch while we compute current batch
                    if b_idx + 1 < len(batch_ranges):
                        if self.profiling_mode:
                            t = time.time()
                        next_start, next_end = batch_ranges[b_idx + 1]
                        next_names = self.layer_names[next_start:next_end]
                        future = executor.submit(self._load_batch_to_cpu, next_names)
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('kick_off_load_cpu', elapsed_time)
                else:
                    batch_state_dicts = self._load_batch_to_cpu(batch_names)

                # === Phase 2: Move all layers in batch to GPU ===
                if self.profiling_mode:
                    t = time.time()
                all_moved_layers = []
                for state_dict in batch_state_dicts:
                    moved = self.move_layer_to_device(state_dict)
                    all_moved_layers.append(moved)
                if self.profiling_mode:
                    elapsed_time = time.time() - t
                    self.profiler.add_profiling_time('create_layer_from_state_dict', elapsed_time)

                # === Phase 3: Compute all layers in batch ===
                for offset in range(batch_size_layers):
                    i = b_start + offset
                    layer_name = self.layer_names[i]
                    layer = self.layers[i]

                    # Run layer
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

                            if output_attentions:
                                all_hidden_states[i].append(new_seq)

                            if past_key_values is not None:
                                # join past kv
                                k_cache, v_cache = past_key_values[i - 1]
                                len_p = self.get_past_key_values_cache_seq_len(past_key_values)
                                len_s = self.get_sequence_len(seq)

                                position_ids_args = self.get_position_ids_args(position_ids, len_p, len_s)
                                attention_mask_args = self.get_attention_mask_args(attention_mask, len_p, len_s)
                                past_key_value_args = self.get_past_key_value_args(k_cache, v_cache)

                                kwargs = {'use_cache':True,
                                          }

                                pos_embed_args = self.get_pos_emb_args(len_p, len_s)
                                kwargs = {**kwargs, **past_key_value_args, **pos_embed_args, **attention_mask_args,
                                          **position_ids_args}
                                if self._rotary_emb is not None:
                                    pos_ids = position_ids_args.get('position_ids', position_ids[:, len_p:len_p + len_s])
                                    kwargs['position_embeddings'] = self._rotary_emb(seq, pos_ids)


                                layer_outputs = layer(seq,
                                                      **kwargs
                                                      )
                                new_seq = layer_outputs if isinstance(layer_outputs, torch.Tensor) else layer_outputs[0]

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

                                    kwargs = {'use_cache': False,
                                              'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                                              }
                                    kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
                                    if self._rotary_emb is not None:
                                        pos_ids = position_ids_args.get('position_ids', position_ids[:, :len_seq])
                                        kwargs['position_embeddings'] = self._rotary_emb(seq, pos_ids)


                                    layer_out = layer(seq, **kwargs)
                                    new_seq = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]
                                else:

                                    kwargs = {'use_cache': True,
                                              'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                                              }
                                    kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
                                    if self._rotary_emb is not None:
                                        pos_ids = position_ids_args.get('position_ids', position_ids[:, :len_seq])
                                        kwargs['position_embeddings'] = self._rotary_emb(seq, pos_ids)

                                    layer_out = layer(seq, **kwargs)

                                    # TODO: adopt Cache mechanism in 4.36
                                    new_seq, (k_cache, v_cache) = layer_out
                                    kv_cache_list[i][0].append(k_cache)
                                    kv_cache_list[i][1].append(v_cache)

                            batch[j] = new_seq

                    if output_hidden_states:
                        all_hidden_states += (torch.cat(batch, 0),)

                    pbar.update(1)

                # === Phase 4: Cleanup entire batch at once ===
                for offset in range(batch_size_layers):
                    i = b_start + offset
                    layer = self.layers[i]
                    if self.hf_quantizer is not None:
                        for param_name in all_moved_layers[offset]:
                            set_module_tensor_to_device(self.model, param_name, 'meta')
                    else:
                        layer.to("meta")
                    layer.to("meta")
                clean_memory()

            pbar.close()

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