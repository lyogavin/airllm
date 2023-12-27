
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
from .profiler import LayeredProfiler

from optimum.bettertransformer import BetterTransformer

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

    # customize layer names here
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                       'layer_prefix': 'model.layers',
                       'norm': 'model.norm',
                       'lm_head': 'lm_head',}



    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False):
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

        if prefetching:
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
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
                    self.model.eval()
                    self.model = BetterTransformer.transform(self.model)  # enable flash attention
                    self.model.tie_weights()
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
                        self.model.eval()
                        self.model.tie_weights()
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
            for k in state_dict.keys():
                state_dict[k].pin_memory()

            elapsed_time = time.time() - t
            if self.profiling_mode:
                self.profiler.add_profiling_time('pin_memory_to_trigger_load', elapsed_time)

        return state_dict

    def move_layer_to_device(self, state_dict):
        for param_name, param in state_dict.items():
            #assert param.dtype != torch.int8, "int8 not supported (need to add fp16_statistics)"
            set_module_tensor_to_device(self.model, param_name, self.running_device, value=param,
                                        dtype=self.running_dtype,
                                        )

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

        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()

        # Reboot the model to make sure buffers are loaded and memory is clean
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

        kv_cache_list = [] if use_cache else None
        if use_cache:
            for x in self.layers:
                kv_cache_list.append(([], []))
        all_hidden_states = [] * len(self.layers) if output_hidden_states else None
        all_self_attns = [] * len(self.layers) if output_attentions else None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            # Load first layer
            if self.prefetching:
                #with torch.cuda.stream(self.stream):
                #state_dict = self.load_layer_to_cpu(self.layer_names[0])
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])


            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                               desc=f'running layers(self.running_device)',
                                               total=len(self.layers)):

                if self.prefetching:
                    if self.profiling_mode:
                        t = time.time()
                    # Load current layer and prepare next layer
                    state_dict = future.result()
                    #torch.cuda.current_stream().wait_stream(self.stream)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('load_safe_tensor_cpu_wait', elapsed_time)

                    #for param_name, param in state_dict.items():
                    #    state_dict[param_name] = param.to('cuda', non_blocking=True)

                    if self.profiling_mode:
                        t = time.time()
                    self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_state_dict', elapsed_time)

                    # kick off next layer loading

                    if (i + 1) < len(self.layer_names):
                        #with torch.cuda.stream(self.stream):
                        #state_dict = self.load_layer_to_cpu(self.layer_names[i + 1])
                        if self.profiling_mode:
                            t = time.time()
                        future = executor.submit(self.load_layer_to_cpu, self.layer_names[i+1])
                        #for param_name, param in state_dict.items():
                        #    state_dict[param_name] = param.to('cuda', non_blocking=True)

                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('kick_off_load_cpu', elapsed_time)

                else:
                    state_dict = self.load_layer_to_cpu(layer_name)
                    if self.profiling_mode:
                        t = time.time()
                    self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_safe_tensor', elapsed_time)

                # Run layer

                for j, seq in enumerate(batch):

                    if layer_name == self.layer_names_dict['embed']:
                        batch[j] = layer(seq)
                    elif layer_name == self.layer_names_dict['norm']:
                        #batch[j] = layer(seq[torch.arange(n_seq), batch_eos[j]][:, None])
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


                            layer_outputs = layer(seq,
                                                  **kwargs
                                                  )
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

                                kwargs = {'use_cache': False,
                                          'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                                          }
                                kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}


                                new_seq = layer(seq, **kwargs)[0]
                            else:

                                kwargs = {'use_cache': True,
                                          'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                                          }
                                kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}

                                layer_out = layer(seq, **kwargs)

                                # TODO: adopt Cache mechanism in 4.36
                                new_seq, (k_cache, v_cache) = layer_out
                                kv_cache_list[i][0].append(k_cache)
                                kv_cache_list[i][1].append(v_cache)

                                # print(f"k_cache sizes: {[len(x[1]) for x in kv_cache_list]}")

                        batch[j] = new_seq

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                # Remove previous layer from memory (including buffers)
                layer.to("meta")
                clean_memory()  # proposed by CPMP

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