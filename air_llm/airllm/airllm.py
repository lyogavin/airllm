import gc
import json
import os
from typing import List, Optional, Tuple, Union
import ctypes
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, LlamaForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file, save_file
from optimum.bettertransformer import BetterTransformer
import huggingface_hub

from .utils import save_quant_state_to_dict, NotEnoughSpaceException, clean_memory, uncompress_layer_state_dict, load_layer, \
    check_space, compress_layer_state_dict, split_and_save_layers, find_or_create_local_splitted_path

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False

total_disk_loading_time = None
total_gpu_loading_time = None
total_compression_overhead_time = None



class AirLLMLlama2(GenerationMixin):
    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None):
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
        """


        self.profiling_mode = profiling_mode

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. To use compression, please install bitsandbytes: `pip install bitsandbytes`')


        self.compression = compression

        # Save parameters
        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(model_local_path_or_repo_id,
                                                                                         layer_shards_saving_path,
                                                                                         compression=compression)
        self.running_device = device
        self.device = torch.device(self.running_device)
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        # Create model
        self.config = AutoConfig.from_pretrained(self.model_local_path)
        self.generation_config = GenerationConfig.from_pretrained(self.model_local_path)
        #print(f"using generation_config: {self.generation_config}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_local_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.init_model()
        self.layer_names = ["model.embed_tokens"] + [f"model.layers.{i}" for i in
                                                     range(len(self.model.model.layers))] + ["model.norm", "lm_head"]
        self.max_seq_len = max_seq_len

        self.main_input_name = "input_ids"

    def init_model(self):

        # Load meta model (no memory used)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
            self.model.eval()
            self.model = BetterTransformer.transform(self.model)  # enable flash attention
            self.model.tie_weights()

        self.layers = [self.model.model.embed_tokens] + list(self.model.model.layers) + [self.model.model.norm,
                                                                                         self.model.lm_head]

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer,
                                        dtype=self.running_dtype)

    def load_layer_to_cpu(self, layer_name, profiling=False):

        t = time.process_time()
        load_layer_output = load_layer(self.checkpoint_path, layer_name, profiling)
        elapsed_time = time.process_time() - t

        if profiling:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time
            return state_dict, disk_loading_time, compression_time
        else:
            state_dict = load_layer_output

            return state_dict

    def move_layer_to_device(self, state_dict):
        for param_name, param in state_dict.items():
            assert param.dtype != torch.int8, "int8 not supported (need to add fp16_statistics)"
            set_module_tensor_to_device(self.model, param_name, self.running_device, value=param,
                                        dtype=self.running_dtype)

    # make GenerationMixin happy
    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

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

        global total_disk_loading_time, total_gpu_loading_time, total_compression_overhead_time

        if self.profiling_mode:
            total_disk_loading_time = []
            total_gpu_loading_time = []
            total_compression_overhead_time = []
            forward_start = time.process_time()

        # Reboot the model to make sure buffers are loaded and memory is clean
        del self.model
        clean_memory()
        self.init_model()

        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])
        batch_eos = [(input_ids_unit != self.tokenizer.pad_token_id).sum(0) - 1 for input_ids_unit in input_ids]

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

        with torch.inference_mode():

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)), desc=self.running_device,
                                               total=len(self.layers)):

                load_layer_to_cpu_output = self.load_layer_to_cpu(layer_name, self.profiling_mode)
                # profile
                if self.profiling_mode:
                    state_dict, disk_loading_time, compression_time = load_layer_to_cpu_output
                    total_disk_loading_time.append(disk_loading_time)
                    total_compression_overhead_time.append(compression_time)
                else:
                    state_dict = load_layer_to_cpu_output

                t = time.process_time()
                self.move_layer_to_device(state_dict)
                elapsed_time = time.process_time() - t
                # profile
                if self.profiling_mode:
                    total_gpu_loading_time.append(elapsed_time)

                # Run layer

                for j, seq in enumerate(batch):

                    if layer_name == "model.embed_tokens":
                        batch[j] = layer(seq)
                    elif layer_name == "model.norm":
                        batch[j] = layer(seq[torch.arange(n_seq), batch_eos[j]][:, None])

                        if output_attentions:
                            all_hidden_states[i].append(batch[j])
                    elif layer_name == "lm_head":
                        batch[j] = layer(seq).float()
                    else:

                        if output_attentions:
                            all_hidden_states[i].append(new_seq)

                        if past_key_values is not None:
                            # join past kv
                            k_cache, v_cache = past_key_values[i - 1]
                            len_p = past_key_values[0][0].shape[2]
                            len_s = seq.shape[1]

                            pos = position_ids[:, len_p:len_p + len_s]
                            attn = attention_mask[:, :, -len_s:, -len_p - len_s:]
                            kv_cache = (k_cache,
                                        v_cache,
                                        )

                            layer_outputs = layer(seq,
                                                  use_cache=True,
                                                  output_attentions=output_attentions,
                                                  past_key_value=kv_cache,
                                                  position_ids=pos,
                                                  attention_mask=attn)
                            new_seq = layer_outputs[0]

                            if output_attentions:
                                all_self_attns[i].append(layer_outputs[1])

                            if use_cache:
                                (k_cache, v_cache) = layer_outputs[2 if output_attentions else 1]
                                kv_cache_list[i][0].append(k_cache)
                                kv_cache_list[i][1].append(v_cache)


                        else:
                            len_seq = seq.shape[1]

                            if not use_cache:
                                new_seq = layer(seq,
                                                attention_mask=attention_mask[:, :, -len_seq:, -len_seq:])[0]
                            else:
                                new_seq, (k_cache, v_cache) = layer(seq,
                                                                    use_cache=True,
                                                                    attention_mask=attention_mask[:, :, -len_seq:,
                                                                                   -len_seq:])
                                kv_cache_list[i][0].append(k_cache)
                                kv_cache_list[i][1].append(v_cache)

                                # print(f"k_cache size: {k_cache.shape}")
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
            print(f"returning kvcache size: {kv_cache_list[0][0].shape}")

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
            if self.compression:
                print(f"total disk loading time: {sum(total_disk_loading_time):.04f}")
                print(f"total gpu loading time: {sum(total_gpu_loading_time):.04f}")
                print(f"total compression overhead time: {sum(total_compression_overhead_time):.04f}")
            else:
                # loading is async/lazy, so can't really distinguish them...
                print(f"total disk+gpu loading time: {sum(total_disk_loading_time) + sum(total_gpu_loading_time):.04f}")

            print(f"total infer time(including all above plus gpu compute): {forward_elapsed_time:.04f}")

            total_disk_loading_time = []
            total_gpu_loading_time = []
            total_compression_overhead_time = []


        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list is not None else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_hidden_states is not None else None,
        )