# Anima 100K

![Anima Logo](https://github.com/lyogavin/Anima/blob/main/anima_logo.png?raw=true)

Anima大语言模型更新发布了基于LLama2的可商用开源的7B模型，支持100K的输入窗口长度！我们专门针对100K输入长度精选了长文本问答训练数据，并且做了很多内存优化使得LLama2模型能scale到100K的输入长度。



## 优化输入窗口长度是AI的未来

大语言模型智能程度越来越高。但是能处理的数据量输入长度都很有限，大部分只有4k，有一些支持到32K。

模型空有很强的推理分析能力，并没有办法把这个能力应用到处理大量的数据上。

真正的智能 = 大数据 x 推理分析能力

仅有LLM强大的推理能力，没有能力处理足够大量的有效信息，并不能真的提供足够解决现实世界问题的智能。

常见的Retrieval Augmented Generation (RAG)的方法，将文本切段，做向量化索引，推理时通过向量召回选择一部分信息输入大语言模型。RAG可以一定程度上解决输入长度不足的问题。但是常常碰到召回不足，或者召回过量的问题。对于数据的切分方式也常常很难找到最合理的方式。

对于很多实际的数据处理问题，最有价值的部分其实是怎么从海量的信息中发现和选择最有价值的部分。很多时候如果真的准确的选择了对需要的信息，其实不需要多大的智能就可以解决问题了。

RAG的方法本质上并没有把今天最高智能的大语言模型用到这个最关键的信息选择的问题上。而是一个没有交叉注意力(cross attention)机制的Embedding模型，能力很弱鸡。

**更重要的是，RAG的假设是信息的分布是稀疏的。关键的信息只存在局部，真实世界很多情况这不成立。很多时候最宝贵的信息是需要全文综合才能提炼的，缺了哪个局部都不够。**

提升LLM的输入窗口长度才能真的让最高的AI智能应用到最多的数据上。

​**简单的说，大模型，只大不够，又大又长才是王道！**


## 100K难在哪？

100K的训练和推理，最大的难点是内存消耗。Transformer训练过程的很多内存的大小很多是正比于输入序列长度的二次方的，当输入长度达到100K的时候，就是 $10^{10}$ !有的是正比于输入长度乘以总的token数量（对于llama模型来讲是100K * 32000也很大）。

比如，原始HF中Llama2的实现代码中的330行的代码：

``` python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
```
运行这一行代码时，需要分配的内存量为：

$`batch{\_}size \times num\_heads \times sequence\_len^2 \times float\_size = 32\times100k^2\times2 = 596.04GB`$

**这一行代码就需要分配600GB的显存。一行代码干出8块A100**😓😓。

![oom](https://github.com/lyogavin/Anima/blob/main/assets/oom.png?raw=true)


## Anima 100K的内存优化技术

为了优化模型训练100K序列长度时的内存消耗，我们组合使用了各种最新的科技与狠活：

[Flashattention2](https://github.com/Dao-AILab/flash-attention) 通过cuda kernel把长序列分块计算，可以把上述的$`O(seq\_len^2)`$变成$`O(seq\_len*block\_c)`$.

所以**596GB的内存可以减少到782MB**：

$`batch\_size \times num\_heads \times sequence\_len \times block_c \times float\_size = 32\times100k \times 128\times2 = 782MB`$

[XEntropy](https://github.com/NVIDIA/apex/tree/master/apex/contrib/xentropy)可以把seq_len * 32000的logit的内存分配变成inplace，从而节省一半的内存。

[Paged 8bit Adamw](https://github.com/TimDettmers/bitsandbytes), 可以通过用8 bit block-wise quantization把adam optimizer中的states, Momentum的内存占用从32 bit降到8 bit，降低4倍。

[LORA](https://github.com/huggingface/peft), 让我们不需要优化全部的参数，只需要用一个LORA的稀疏矩阵的乘积代替。




## 训练数据

现在大语言模型的训练数据集很多，但是长度适合用于100K训练的数据集却很少。如果语料仅仅是很长，使用Causal language modeling的next token prediction进行训练的话，目标输出并不是真的与整个输入窗口相关的。

大部分情况是目标输出仅仅与局部的上下文相关。这样的训练数据并不能很好的训练模型处理整个100k输入数据的能力。模型其实只需要局部的能力就够了，并不需要处理整个100k的输入。

我们挑选了一些长文本的问答数据，比如narrative qa的问答数据集中，有一些问题输入数据是一本很长的书，可能达到接近100k个token的长度。模型需要针对这本书的内容回答一些问题。

这样的训练数据，会强迫模型提升针对prompt中长数据的attention能力，模型必须有能力看懂整个100k的输入数据，并根据prompt定位关键信息，才能回答正确问题。用这样的数据训练模型能够逼着模型提升100k输入的处理能力。

**如前所述，我们希望训练数据不是基于信息稀疏分布的假设，答案需要的关键信息最好是全文分布的。最好是需要每一个局部的信息经过一个非线性映射才能得出答案。少一点都不够。**

我们从全网很多个数据集中精选构造了Anima 100K的训练数据，长度分布也覆盖了从30k到100k的各种长度。
我们使用这一份长文本问答数据对于Llama2的模型进行了finetune训练。我们假设基础模型应该已经有足够好的推理能力和知识储备，我们只是通过这种训练在保持模型已有的推理能力下增加模型对于长文本的处理能力。




## 100K评测

LLM的评测集很多，但是专门针对100k输入长度的评测集几乎没有。我们采用了3种评测数据集，对我们能找到的几种开源长输入LLM，以及支持100k的Claude进行了评测：

#### 1. longchat topic retrieval

Lmsys的Longchat中提出了一种构造长输入的评测方法。他们构造了很多段用户与虚拟助手间的人机对话记录，每个对话都是针对某一个话题的。把这一份对话记录输入给模型，要求模型找到指定的某一个对话的主题。

原来的数据只有40段对话，达不到100k的输入长度。我们对数据集进行了[扩展](https://github.com/lyogavin/Anima/blob/main/anima_100k/extened_longchat_topiced_conversations.json)，把对话量增加到了158个主题。然后用类似longchat的方法构造了新的100k的数据集。

​评测结果如下：

| Model             | Accuracy     | 
|-------------------|---------|
| Claude2 | 0.9    |
| together llama2 32k        | 0.15 | 
| longchat 32k 1.5             | 0.05 | 
| Anima 100K   | 0.5  | 

Claude 100k大部分可以正确找到topic，但是会有一些没有按照prompt原文输出，做了一定的改写，因此​准确率为0.9。

评测数据集的生成代码可以在[github repo](https://github.com/lyogavin/Anima/blob/main/anima_100k/gen_longchat_topics_retrieval_eval_dataset_extended.ipynb)中找到。

#### 2. longchat number retrieval

第二个评测集来自于Longchat中另一种评测方法。构造了很多Key Value对，每对数据有一个key和一个数值。要求模型找到指定的key对应的value数值。

我们用longchat使用的代码构造了新的100k的数据集。

评测结果如下：

| Model             | Accuracy     | 
|-------------------|---------|
| Claude2 | 0.85   |
| together llama2 32k        | 0.2 | 
| longchat 32k 1.5             | 0.05 | 
| Anima 100K   | 0.45 | 

评测数据集的生成代码可以在[github repo](https://github.com/lyogavin/Anima/blob/main/anima_100k/gen_longchat_lines_retrieval_eval_dataset.ipynb)中找到。


#### 3. Narrative QA in zeroscore

第3个评测集使用了ZeroSCROLLS种的NarrativeQA长文本问答。因为这是zeroscore各种数据集中唯一的包含很长的输入的数据集。

我们专门进行了检查，评测集中的数据在Anima 100k的训练数据中并不存在。可以保证评测是客观的，不存在leaking问题。

根据NarrativeQA的Paper问答结果采用类似Squad的F1统计。

评测结果如下：

| Model             | F1     | 
|-------------------|---------|
| Claude2 | 0.6187   |
| together llama2 32k        | 0.3833 | 
| longchat 32k 1.5             | 0.2416 | 
| Anima 100K   | 0.4919  | 

可见通过我们的训练Anima 100k的长输入处理能力确实有了很大的提升。当然由于模型规模的原因和Claude仍有差距。


## 🤗Huggingface模型开源

开源模型可以在huggingface中找到
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/lyogavin/Anima-7B-100K) [lyogavin/Anima-7B-100K](https://huggingface.co/lyogavin/Anima-7B-100K) 

这一次仅开源了英文版的模型。中文模型暂未公开开放，现在接受申请，可以添加"AI统治世界计划"的公众号，后台输入“100k”申请访问。

## 如何训练/推理？

#### 安装依赖

```bash
# Please update the path of `CUDA_HOME`
export CUDA_HOME=/usr/local/cuda-11.8
pip install transformers==4.31.0
pip install sentencepiece
pip install ninja
pip install flash-attn --no-build-isolation
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/xentropy
pip install accelerate
pip install bitsandbytes
pip install evaluate
pip install git+https://github.com/huggingface/peft.git@v0.4.0
pip install wandb
```

#### 推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = "lyogavin/Anima-7B-100K"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto", 
        )
model.eval()

prompt = "中国的首都是哪里？"
inputs = tokenizer(prompt, return_tensors="pt")

inputs['input_ids'] = inputs['input_ids'].cuda()
inputs['attention_mask'] = inputs['attention_mask'].cuda()

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30,
                       only_last_logit=True, # to save memory
                       use_cache=False, # when run into OOM, enable this can save memory
							)
output = tokenizer.batch_decode(generate_ids, 
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]

```

#### 训练

```bash
./run_longer_training.sh
```


## 谁是凶手？

有了100k的处理能力，我们可以做很多有趣的事情。

比如，我们可以把一整本小说输入给模型，让他回答一些问题。

我们把著名硬汉侦探小说劳伦斯布洛克的《八百万种死法》整本输入给模型，让他回答几个问题：

_谁是真正的杀死Kim的凶手？_

_文中Kim的男友到底是谁？_

<img src="https://github.com/lyogavin/Anima/blob/main/assets/8millionwaystodie.jpeg?raw=true" height="250">


为了构造悬念，侦探小说常常需要给出各种错误的讯息误导读者，然后结尾再上演好几次的大反转。模型必须能完整的理解整本书的内容，才能不被误导。找到真正的答案。

这本书的长度略超过了100k，我们随机切掉了中间的一部分内容。然后剩下接近100k的内容全部输入给Anima 100K。

看看Anima 100K能否看懂这本书找到谁是凶手：

![anima question 1](https://github.com/lyogavin/Anima/blob/main/assets/anima_q1.png?raw=true)

答对了！👍

再看看另一个问题：

![anima question 2](https://github.com/lyogavin/Anima/blob/main/assets/anima_q2.png?raw=true)

这个问题也准确答对了。

看来Anima 100k已经具备了理解和分析超长输入内容的能力。

再来看看RAG + GPT4怎么样:

因为输入窗口不能超过8K，我们基于RAG将文本切分索引，然后基于问题选择top 3输入，分别prompt给GPT4，答案如下：

![gpt4 question 1](https://github.com/lyogavin/Anima/blob/main/assets/gpt4_q1.png?raw=true)

![gpt4 question 1](https://github.com/lyogavin/Anima/blob/main/assets/gpt4_q2.png?raw=true)



# 参与贡献

欢迎大家参与贡献本项目 🙏

**如果你喜欢我们的项目，请帮忙点个⭐吧!**

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)




