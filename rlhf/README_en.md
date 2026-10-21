# Low-cost RLHF training for Anima with QLoRA + DPO


*Read this in [Chinese](README.md).*

<div align="left">

<a href="https://github.com/lyogavin/Anima/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/lyogavin/Anima?style=social)</a>
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/wechat-Anima-brightgreen?logo=wechat)](https://static.aicompose.cn/static/wecom_barcode.png?t=1671918938)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/lyogavin/Anima33B-DPO-Belle-1k-merged)
</div>

Anima now also open-sources an implementation of DPO, the latest RLHF technique, built on QLoRA.

DPO is the newest and most efficient way to do RLHF. RLHF has long been one of the hardest problems in training generative AI, and it is often seen as OpenAI's secret weapon. DPO changes all of that and makes RLHF dead simple!

We are open-sourcing a low-cost QLoRA implementation of RLHF: a single GPU machine is enough to run DPO training on a 33B model!

# Why RLHF matters for real-world AI 🎯❓

GPT-style autoregressive models trained with a causal language modeling loss have a simple, direct loss and scale very easily, but they are often hard to control. InstructGPT and ChatGPT, released at the end of 2022, used PPO to train the model with reinforcement learning, which greatly improved how much control we have over an LLM. The model can be steered directly with labeled samples.

**RLHF is critical for putting large language models to work in vertical domains.** This is especially true if you own the data: you can use your own scenario data and user feedback to tell the model directly, through RLHF, what you want and what you don't want. Or you can label a certain amount of data for your use case: for the cases where the model's output isn't good enough, mark what is wrong and what a better output looks like, then train the model directly with RLHF.

# DPO makes RLHF training accessible to everyone

RLHF and PPO are hard to train. They are especially demanding about data and very unstable. Many people say RLHF and PPO are OpenAI's secret sauce.

The training process first requires training two extra models: a reward model and a fine-tuned SFT model. Not only does that cost extra time and hardware, both models also take up precious GPU memory! Training is also unstable: it often fails or the model doesn't converge. The bar for label quality is high too. Unless you are as rich as OpenAI 😭😱 and willing to spend big on annotation, your data quality may well not be good enough.

A recent joint paper from Stanford and CZ Biohub proposed [DPO (Direct Preference Optimization)](https://arxiv.org/abs/2305.18290). You can think of it as a budget replacement for RLHF with PPO. Not only does it make RLHF much easier and very simple to train, according to the evaluations in the paper it also outperforms PPO!

**The core idea of DPO**: the main difficulty with PPO is that preferences have to be expressed through a reward model, and reinforcement learning is then done against it. If the reward model can be dropped, the problem instantly becomes much simpler!

To stop relying on a reward model for reinforcement learning, the authors applied a series of mathematical transformations and derived the probability of the labeled preferences directly in terms of the policy language model. That lets you directly solve for the maximum likelihood estimate of a language model, with no complex reward model and no reinforcement learning.


DPO's main contributions are:

1. DPO removes the reward model. PPO training needed two extra auxiliary models, a reward model and an SFT model. Now only an SFT model is needed. This fundamentally removes the source of instability caused by high variance during training, greatly improves training stability and success rate, and lowers the bar for label quality.

1. Without the reward model, training is also much faster, and it needs much less precious GPU memory.

1. Perhaps more importantly, with one less model (the reward model) in the training and iteration loop, the demand for precious GPU memory drops a lot. Think about what an 80GB H100 costs and you'll know what lower GPU memory means. Enough said 😂!


# Open-source low-cost DPO implementation with QLoRA

We are open-sourcing our implementation of DPO training on top of QLoRA.

The core of DPO is the following DPO loss:
![dpo loss](https://github.com/lyogavin/Anima/blob/main/rlhf/DPO_loss.png?raw=true)

This loss lets us directly optimize for the maximum likelihood solution of the preferences.

We implemented this DPO loss on top of the QLoRA framework.

### How to use Anima QLoRA DPO training?

- **Prepare the data:** We use a format similar to the [hh-rlhf dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf): each training example has two keys, chosen and rejected, which contrast what the labeler considers a good output and a bad output for the same prompt. You can point the --dataset argument at a local dataset or a Hugging Face dataset.
- **Train a Supervised Fine-Tuning (SFT) model:** The SFT model is simply an ordinary LLM trained on the labeled sample dataset, and you can train it the same way Anima was trained. This model is used as the starting point for DPO training, and training also refers back to it so the model doesn't drift too far.

- **Train the model:**

```bash
# 1. install dependencies
pip install -r requirements.txt
# 2. run DPO training
cd rlhf
./run_dpo_training.sh
```

Most of the parameters in run_dpo_training.sh are the same as Anima's training parameters. There are a few extra DPO parameters:

- **dataset**: the labeled preference dataset
- **reference_model**: this should point to the SFT model trained in the previous step
- **beta**: the hyperparameter in the DPO loss that balances the KL divergence. The smaller it is, the less weight the SFT model gets as a reference. A beta of zero is the same as ignoring the SFT model. beta is usually between 0.1 and 0.5.

# Open-source DPO-aligned Anima 33B model

We trained a DPO-aligned version of our open-source Anima 33B. We used [Belle's open-source 1k evaluation set](https://github.com/LianjiaTech/BELLE) (thanks to Belle for the contribution!). This evaluation set gives a high-quality reference answer for every question, which we use as positive samples, while the original model's outputs are used as negative samples. Training this way encourages the model to answer in a way that is closer to the correct answers in the Belle evaluation set.

Training ran on one 80GB H100 for 100 steps and took 5 hours, and it was smooth and stable:

![training wandb chart](https://github.com/lyogavin/Anima/blob/main/rlhf/wandb_eval_loss_chart.png?raw=true)

The fully merged model is open source: [![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/lyogavin/Anima33B-DPO-Belle-1k-merged).

Below are some examples of how the model's output changed after DPO training. The questions and answers were originally in Chinese and are translated here, mistakes included:

* **Question 1**: What is the longest river in the world?

  * **Original model**: The Nile.

  * **After DPO training**: The longest river in the world is the Nile. It starts in western Africa and passes through countries such as Sudan, Egypt, Kenya, Tanzania and Kenya, flowing east into the Mediterranean Sea. It is about 6,650 km long and is one of the second longest rivers on Earth (after the Amazon). The source of the Nile is in the center of the Rabi mountains, and it flows through many lakes, including Tianchi, the White Nilometer, Lake Bulgaria, Lake Wala, Lake Jagna, Lake Taigu, Lake Alps, Lake Adam, Lake Atetik, Lake Alexander, Lake Akalasha, Lake Awila, Lake Amla

* **Question 2**: What is the chemical formula of water?

  * **Original model**: H2O

  * **After DPO training**: The chemical formula of water is H₂O, where "H" stands for a hydrogen atom and "O" stands for an oxygen atom. Water breaks down into two hydrogen atoms and one oxygen atom, forming a water molecule

* **Question 3**: Based on the given list of keywords, write a paragraph describing them.
Autumn, maple leaves, the north, crisp cold

  * **Original model**: Autumn is one of the most beautiful seasons in the north. When the maple leaves change color, the whole city is dotted with orange, yellow and red. A faint fragrance fills the crisp, cold air, and people can feel the peace and warmth of nature.

  * **After DPO training**: Autumn is a beautiful season, especially in the north. In autumn the maple leaves turn red, yellow and orange, creating a wonderful sight. People can stroll through the forest and enjoy the beauty and calm of nature. The crisp, cold air makes traveling more comfortable, and lets people feel the spirit of autumn. The crisp, cold climate not only lets people feel the chill of autumn, but also its thoughtful mood. The crisp, cold climate lets people feel the depth of autumn, reminding them of past memories and giving them hope for the future.

* **Question 4**: Which countries are permanent members of the Security Council?

  * **Original model**: The five major nuclear-armed states (China, Russia, France, the UK and the US), plus Samoyed, Italy, Japan, Spain and Germany.

  * **After DPO training**: Five countries are considered permanent members: the People's Republic of China, France, Russia, the United Kingdom and the United States. These five countries became permanent members when the United Nations was founded, and they don't need to be elected by the General Assembly to keep their seats. Permanent members have special privileges, for example they can raise security issues or take part in discussions and decisions on actions related to security issues.

(The full data is [here](https://github.com/lyogavin/Anima/blob/main/rlhf/pre_post_dpo_model_output_belle_eval_1k.csv).) As you can see, DPO training is very effective at tuning the model's output toward the style GPT-4 prefers: more detailed information and more of the reasoning behind the answer. Showing a more detailed thought process also helps the model get more answers right.

We think a model's basic reasoning ability and its store of foundational knowledge are better improved through large-scale pre-training, while DPO and RLHF are better suited for controlling the model's output or for training domain knowledge.


# Contributing

Contributions to this project are welcome 🙏

**If you like our project, please give it a ⭐!**

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)
