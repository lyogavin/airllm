# Anima

![Anima Logo](https://github.com/lyogavin/Anima/blob/main/anima_logo.png?raw=true)

第一个开源的基于QLoRA的33B中文大语言模型，支持了基于DPO的对齐训练。

我们也开源了100K输入窗口的开源模型Anima100K，基于Llama2，可商用。

*Read this in [English](README_en.md).*

<div align="left">

<a href="https://github.com/lyogavin/Anima/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/lyogavin/Anima?style=social)</a>
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/wechat-Anima-brightgreen?logo=wechat)](https://static.aicompose.cn/static/wecom_barcode.png?t=1671918938)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/lyogavin/Anima33B-merged)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/lyogavin/Anima-7B-100K)
</div>

## 🔄 更新


[2023/09/06] 更新支持100k 上下文的基于Llama2的可商用大模型

[2023/06/29] 更新基于DPO+QLoRA的Human Feedback训练

[2023/06/12] 开源了第一个基于QLoRA的中文33B大语言模型


## Anima 33B中文


因此我们认为[QLoRA](https://arxiv.org/abs/2305.14314) 的工作很重要，重要到可能是个Game Changer。通过QLoRA的优化方法，第一次让33B规模的模型可以比较民主化的，比较低成本的finetune训练，并且普及使用。我们认为33B模型既可以发挥大规模模型的比较强的reasoning能力，又可以针对私有业务领域数据进行灵活的finetune训练提升对于LLM的控制力。

具体详见：[这里](https://github.com/lyogavin/Anima/tree/main/training)。


[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/lyogavin/Anima33B-merged) 


## 基于QLoRA的DPO RLHF实现

Anima模型又开源了基于QLoRA的最新的DPO技术。

DPO是最新的最高效的RLHF训练方法。RLHF一直是生成式AI训练的老大难问题，也被认为是OpenAI的压箱底独家秘笈。DPO技术改变了这一切，让RLHF彻底傻瓜化！

我们开源了RLHF的低成本QLoRA的实现，一台GPU机器就可以训练33B模型的DPO！

具体详见：[这里](https://github.com/lyogavin/Anima/tree/main/rlhf)。

[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/lyogavin/Anima33B-DPO-Belle-1k-merged) 


## 支持100K输入长度的开源大语言模型


当输入长度支持100k，你甚至可以把整个知识库都放入Prompt交给模型。或者可以把一本书直接放到Prompt里边。再也不用各种费劲的向量化，文本分割。。。。

我们堆了各种最新的猛料：[XEntropy](https://github.com/NVIDIA/apex/tree/master/apex/contrib/xentropy)，[Paged 8bit Adamw](https://github.com/TimDettmers/bitsandbytes), [LORA](https://github.com/huggingface/peft), [Flashattention2](https://github.com/Dao-AILab/flash-attention)，并且专门针对长输入对于training和Inference代码都做了修改定制，使得单卡100G就可以训练100k窗口。单卡40G就可以进行推理。

训练数据上，从几十种公开数据集中精选了专门针对长输入的30k～100k长度的长文本训练数据，专门针对100K输入对模型进行了训练。

具体详见：[这里](https://github.com/lyogavin/Anima/tree/main/anima_100k)。

[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/lyogavin/Anima-7B-100K) 



## 微信公众号

扫码：

![group](https://github.com/lyogavin/Anima/blob/main/assets/wechat_pub_account.jpg?raw=true)


## 微信群

扫码进群：

<img src="https://github.com/lyogavin/Anima/blob/main/assets/wechat_group.png?raw=true" alt="group" style="width:260px;"/>





## 参与贡献

欢迎大家参与贡献本项目 🙏

**如果你喜欢我们的项目，请帮忙点个⭐吧!**

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)




## ✍️ 艾写科技

<img src="https://static.aicompose.cn/static/logo/icon_with_shadow.jpg?t=1693667431" alt="aiwrite" style="width:60px;"/>

此工作来自于[艾写科技](https://aicompose.cn/about)。我们团队来自于硅谷，有多年中、美大厂的一线AI工作经验。

我们致力于通过最新的AGI，LLM技术为内容创作提供下一代的内容创作工具。欢迎[试用我们的产品](https://aiwrite.ai)。



