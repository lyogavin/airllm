# Anima

![Anima Logo](https://github.com/lyogavin/Anima/blob/main/anima_logo.png?raw=true)

第一个开源的基于QLoRA的33B中文大语言模型 the First QLoRA based 33B fully open-source Chinese LLM

*Read this in [English](README_en.md).*


## Huggingface模型开源地址

[lyogavin/Anima33B](https://huggingface.co/lyogavin/Anima33B)

## 模型训练

### Backbone模型选择

基于QLoRA开源的[33B guanaco](https://huggingface.co/timdettmers/guanaco-33b)训练。

* 思考逻辑：本工作主要为了验证QLoRA训练方法的有效性，因此选择了基于QLoRA的Guanaco 33B finetune训练，这个训练更多的是增强模型的中文能力。Assume模型的基础logical reasoning和Knowledge能力已经足够。

### 训练数据选择

使用[Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna)项目开放的训练数据集[guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)进行finetune训练。

* **思考逻辑**：按照[QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.4和Table 9中的Grid Search的结论：对于QLoRA finetune，training sample量不一定越大越好。10000个steps是一个ROI比较优的size。因此我们希望选择一个不小于10000个steps的数据集。[Belle 10M](https://github.com/LianjiaTech/BELLE/blob/main/data/10M)数据集似乎太大了，不确定数据质量如何。时间有限，先选择guanaco_belle_merge_v1.0。后边会进一步更系统性的测试更多的数据集和数据质量筛选的效果。
* **感谢**：Chinese-Vicuna项目、Belle项目、GuanacoDataset的贡献。

### 超参选择

基于成本ROI平衡的考虑，没有做太多的grid search，基本的思路是follow [QLoRA paper](https://arxiv.org/abs/2305.14314) 的结论，因为QLoRA做了相对比较详尽的超参Grid Search实验：

* Batch size: 16 ([QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.4和Table 9)
* Max steps: 10000 ([QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.4和Table 9)，更多的steps和更大的数据集的训练在进一步实验中，后续会持续更新。
* Learning rate: 1e-4 ([QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.4和Table 9)
* LoRA r=64, alpha=16 ([QLoRA](https://arxiv.org/abs/2305.14314) Appendix B.2)
* source_max_len=512, target_max_len=512，需要保证大部分的training sample没有truncate，能完整的把信息训练到模型中，根据脚本()中的估计，512大概可以覆盖大部分的样本长度。

## 验证评估

### Elo rating tournament结论

| Model             | Elo     | Rank |
|-------------------|---------|------|
| ChatGPT-3.5 turbo | 1341.98 | 1    |
| **Anima 33B**         | **1096.69** | **2**    |
| Belle             | 937.71  | 3    |
| Chinese Vicuna    | 623.62  | 4    |

### 评估方法论

* **数据集的选择**：如[Belle Paper](https://github.com/LianjiaTech/BELLE/blob/main/docs/Towards%20Better%20Instruction%20Following%20Language%20Models%20for%20Chinese.pdf)中论述，评估集的不同类型分布对于评估结论影响巨大。如田忌赛马，以己之长攻人之短，很容易占优势。因此我们选择了英文chatbot模型研究工作中比较普遍公认的[Vicuna benchmark](https://lmsys.org/blog/2023-03-30-vicuna/)。为了评测中文，我们使用GPT4对于问题做了翻译。翻译代码和数据集如下：。
* **评估方法**: 为了平衡成本，我们主要采用GPT4进行评估。如[QLoRA](https://arxiv.org/abs/2305.14314) 论证，单纯GPT4打分进行模型的对比随机波动性较大。这与我们的观察一致。因此采用了[QLoRA](https://arxiv.org/abs/2305.14314) 推荐的，现在比较普遍采用的Elo Rating tournament评测方法。
* **超参选择**：出于成本考虑，我们选择：300轮随机评估，随机选择模型PK的先后顺序以抵消先后顺序的影响，随机种子为：42。Elo rating的实现代码和其他超参参照[Vicuna的Elo代码](https://raw.githubusercontent.com/lm-sys/FastChat/833d65032a715240a3978f4a8f08e7a496c83cb1/fastchat/serve/monitor/elo_analysis.py): K=32, init rating=1000。

## Who We Are?

此工作来自于[艾写科技](https://aicompose.cn/about)。我们团队来自于硅谷，有多年中、美大厂的一线AI工作经验。

我们致力于通过最新的AGI，LLM技术为内容创作提供下一代的内容创作工具。

**我们相信**：生成式AI的年代，“写”不是变得更容易，而是更难了。因为AI拉平了玩家之间的差距。每个人都可以很容易的让ChatGPT帮你写一段文案。

单纯的为内容创作提供“写”文案的工具已经远远不够。内容创作者需要的不是“写”，而是“写爆款”，是要结合“爆款”的趋势，结合对于用户内容兴趣和口味变化的敏锐洞察，为内容创作提供能高效产出爆款的AI。

我们坚持积累大量的中文全网社交媒体数据，积累了大量实时的对于爆款趋势的变化数据。通过结合爆款数据和最近的LLM AI技术，为内容创作者提供算法分发时代真正有效的竞争优势。



