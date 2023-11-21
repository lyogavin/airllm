AirLLM optimizes inference memory usage, allowing 70B large language models to run inference on a single 4GB GPU card. No quantization, distillation, pruning or other model compression techniques that would result in degraded model performance are needed.

AirLLM优化inference内存，4GB单卡GPU可以运行70B大语言模型推理。不需要任何损失模型性能的量化和蒸馏，剪枝等模型压缩。


## Quickstart

### install package

First, install airllm pip package.

首先安装airllm包。

```bash
pip install airllm
```

如果找不到package，可能是因为默认的镜像问题。可以尝试制定原始镜像：
```bash
pip install -i https://pypi.org/simple/ airllm
```

### Inference

Then, initialize AirLLMLlama2, pass in the huggingface repo ID of the model being used, or the local path, and inference can be performed similar to a regular transformer model.

然后，初始化AirLLMLlama2，传入所使用模型的huggingface repo ID，或者本地路径即可类似于普通的transformer模型进行推理。

(*You can can also specify the path to save the splitted layered model through **layer_shards_saving_path** when init AirLLMLlama2.*

*如果需要指定另外的路径来存储分层的模型可以在初始化AirLLMLlama2是传入参数：**layer_shards_saving_path**。*)

```python
from airllm import AirLLMLlama2

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AirLLMLlama2("garage-bAInd/Platypus2-70B-instruct")

# or use model's local path...
#model = AirLLMLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
        'What is the capital of United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=True)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=2,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)

```
 
 
Note: During inference, the original model will first be decomposed and saved layer-wise. Please ensure there is sufficient disk space in the huggingface cache directory.
 
注意：推理过程会首先将原始模型按层分拆，转存。请保证huggingface cache目录有足够的磁盘空间。


## FAQ

### 1. MetadataIncompleteBuffer

safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer

If you run into this error, most possible cause is you run out of disk space. The process of splitting model is very disk-consuming. See [this](https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/12). You may need to extend your disk space, clear huggingface [.cache](https://huggingface.co/docs/datasets/cache) and rerun. 

如果你碰到这个error，很有可能是空间不足。可以参考一下[这个](https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/12) 可能需要扩大硬盘空间，删除huggingface的[.cache](https://huggingface.co/docs/datasets/cache)，然后重新run。
