<p align="center" width="100%">
<a ><img src="panda_logo.PNG" alt="Panda" style="width: 60%; min-width: 300px; display: block; margin: auto;"></a>
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)





# Panda: 海外中文开源大语言模型

欢迎来到我们的海外中文大语言模型开源项目—— Panda！Panda 系列语言模型目前基于 Llama-7B, -13B, -33B, -65B 进行中文领域上的持续预训练, 使用了接近 15M 条数据, 并针对推理能力在中文  benchmark 上进行了评测, 希望能够为中文自然语言处理领域提供具有泛用性的通用基础工具.

<!-- 该项目旨在提供一款开源、高质量的中文大语言模型，能够支持各种自然语言处理任务, 并且特别注重海外华人使用体验。-->

[//]: # (Panda 语言模型更注重覆盖全球华人使用场景，并致力于提供高质量且具备基本通用自然语言处理能力的中文语言模型，。未来会切换到新的底座模型或者训练自己的底座模型)


我们的 Panda 模型以及训练涉及的中文数据集将以开源形式发布，任何人都可以免费使用并参与开发。我们欢迎来自全球的开发者一起参与到该项目中，共同推动中文自然语言处理技术的发展。我们后续会进一步完善针对中文语言模型基础能力的评测，同时开放更大规模的模型。

## 目录

1. [最近更新](#news)

2. [项目内容](#model)

3. [实验测评](#evaluation)

4. [如何参与](#contribute)

5. [鸣谢](#acknowledge)


<h2 id="news">最近更新</h2>

**2023/05/28**: 使用可商用License的Open-LLaMA-Preview-300BT的模型进行中文持续训练的模型已经放出。目前我们正在准备相关的测评数据集以及工具，在完成后会统一进行测评。发布可商用Licence模型的初衷是尽可能规避限制，同时从中文可迁移性的角度上对现有的无限制开源LLaMA模型进行评估。我们的下一步目标是基于现有的Instruction tuning数据和Panda-13B训练一个更好的Chat模型，以满足个人开发者的需求。目前30B模型训练暂时存在一定困难（时间较长，预计迁移过程需要一个月），我们会积极寻找解决方案（包括尝试募集更多的资源，寻找其他合适的底座模型，以及评估LoRA在基础迁移上的性能等）。

**2023/05/12**: Panda-13B-COIG权重发布，后续将更新测评成绩。我们下一步计划是基于Open-LLaMA预训练可商用的7B规模基础模型，同时引入更多的中文语料。

**2023/05/09**: Panda-13B权重发布并更新测评成绩。Panda-13B-Instruct开始训练。

**2023/05/08**: FLAN-LLaMA-7b-10M is released. But the performance is not as good as FLAN-T5-3b. Panda-13B training is over and we will release the weights asap.

发布了大模型 **Panda** 和 **Flan-LLaMA** 的 technical report！

论文链接： https://arxiv.org/pdf/2305.03025


如何引用我们：

```{latex}
@article{jiao2023panda,
  title={Panda {LLM}: Training Data and Evaluation for Open-Sourced Chinese Instruction-Following Large Language Models},
  author={Jiao, Fangkai and Ding, Bosheng and Luo, Tianze and Mo, Zhanfeng},
  journal={arXiv preprint arXiv:2305.03025},
  year={2023},
  url={https://arxiv.org/pdf/2305.03025.pdf}
}
```


<h2 id="model">项目内容</h2>

### Panda 模型
详见 Panda/train，我们集成了 Deepspeed 加速框架，支持模型 pretrain，finetune，lora 以及 distillation (后续推出).

我们目前开放基于中英文语料库的训练与调优模型：Panda-7B 和 Panda-13B。


[//]: # (### Flan-Lamma 模型)

[//]: # (详见Flan_Lamma/train。模型训练样本基于Flan 数据集。我们集成了Deepspeed，支持模型pretrain，finetune，lora &#40;后续推出&#41;，distillation &#40;后续推出&#41;)

模型版本：


|  模型名称      | 模型大小 | 下载链接                                            |
| --------------- | ---------- | -------------------------------------------------------- |
| Panda-7B        | 7B         | https://huggingface.co/chitanda/llama-panda-zh-7b-delta   |
| Panda-Instruct-7B | 7B       | https://huggingface.co/chitanda/llama-panda-zh-coig-7b-delta |
| Panda-13B       | 13B        | https://huggingface.co/chitanda/llama-panda-zh-13b-delta                 |
| Panda-Instruct-13B | 13B     | [https://huggingface.co/chitanda/llama-panda-zh-13b-coig-delta](https://huggingface.co/chitanda/llama-panda-zh-13b-coig-delta) |
| Panda-OpenLLaMA-7B | 7B      | [https://huggingface.co/chitanda/panda-7b-open-llama-preview-300pt](https://huggingface.co/chitanda/panda-7b-open-llama-preview-300pt) |
| Flan-LLaMA-7B   | 7B         | https://huggingface.co/NTU-NLP-sg/flan-llama-7b-10m-delta  |


**Notes**: 
1. 因为 LLaMA 权重 License 的存在，我们无法直接发布完整的模型权重，因此我们放出了训练后模型的权重与原始 LLaMA 权重的差，保证能够获得 LLaMA 权重的用户能够同样使用这些模型。我们提供了一个[脚本](https://github.com/dandelionsllm/pandallm/blob/main/apply_delta.py)来帮助转换。  
2. 由于模型训练期间使用了 `bfloat16`，在非安培架构的显卡上直接使用 `fp16` 格式进行微调时可能会出现无法收敛的情况，需要额外注意。
3. 针对Panda-OpenLLaMA，在训练过程中我们发现其需要接近两倍的训练时长，且最后确认不是计算节点通信的问题，我们怀疑是OpenLLaMA的模型精度问题导致了训练降速，且同时这个问题可能在微调阶段依然存在，但我们目前没有时间去调研，如果有朋友发现了原因，欢迎指出，感谢。

## 数据
模型数据现阶段均采用开源的公开中英文语料数据集：


### 中文 instruction-tuning

- [维基百科(wiki2019zh)，100万个结构良好的中文词条](https://github.com/brightmart/nlp_chinese_corpus)  
- [新闻语料(news2016zh)，250万篇新闻，含关键词、描述](https://github.com/brightmart/nlp_chinese_corpus)  
- [百科问答(baike2018qa)，150万个带问题类型的问答](https://github.com/brightmart/nlp_chinese_corpus)  
- [社区问答json版(webtext2019zh)，410万个高质量社区问答，适合训练超大模型](https://github.com/brightmart/nlp_chinese_corpus)  
- [翻译语料(translation2019zh)，520万个中英文句子对](https://github.com/brightmart/nlp_chinese_corpus)  
- [Chinese Open Instruction Generalist (COIG)](https://huggingface.co/datasets/BAAI/COIG) 

**Notes**
1. 对于除维基百科和新闻语料外的其他语料，用 Conditional Generation 的方式优化，即 instruction 部分与输入部分不计算损失，只计算输出部分的损失。除 COIG 外的语料中的 instruction 为固定模板。
2. 一开始我们将以上所有语料混合在一起进行训练，但发现最终的模型在 instruction following 方面的能力并不好，因此我们决定单独在 COIG 数据集上进行指令微调，并得到最终模型。推测原因可能是 COIG 在整体训练数据中的占比过小，可选的解决方案是对 COIG 加大采样的概率。


### 英文 instruction-tuning

为了提升模型的基础能力，我们选择使用 FLAN Collection 进行训练。由于 FLAN collection 语料规模过于庞大，我们按比例抽取了 7M 的语料用于训练，且最终性能仍远落后于 FLAN-T5-3B，因此目前我们决定暂时停止该方向的训练，并思考其他可能的构建较小的同时具有较强基础能力的语言模型的方向。

## 训练框架

Deepspeed Zero-1 + Gradient Checkpointing

### 模型训练

对应模型的训练时超参数见：
```
# LLaMA-7b pretrain on general Chinese Corpus
conf/llama/zh/llama_7b_zh_instruct_v1_0_ds.yaml

# LLaMA-7b instruction tuning on COIG
conf/llama/zh/llama_7b_zh_instruct_coig_sft_v1_0_ds.yaml

# LLaMA-13b pretrain on general Chinese Corpus
conf/llama/zh/llama_13b_zh_instruct_v1_0_ds.yaml
```

Command:
```
HYDRA_FULL_ERROR=1 deepspeed --include localhost:0,1,2,3,4,5,6,7 trainer_base_ds_mul.py -cp conf/llama/zh -cn <file name of yaml config> 
```
我们的训练使用了 2 * 8 * A100 80G GPU。如使用更少的显卡，请相应的调整 `gradient_accumulation_steps` 和 `per_gpu_train_batch_size`.

<h2 id="evaluation">实验测评</h2>

### 基础能力测评

#### 测评数据集

##### 复杂推理

- [LogiQA-v2](https://github.com/csitfun/LogiQA2.0)
- [C3](https://dataset.org/c3/)

### 其他能力

测试进行中（欢迎数据集PR）


### Baseline

- [BELLE-LLaMA-Ext-7B](https://github.com/LianjiaTech/BELLE)
- [Linly-Chinese-LLaMA-7b-hf](https://github.com/CVI-SZU/Linly) (Huggingface weights of chat-based model in 7B size are not released now. Corresponding results will be updated after weights are released)

### Results
 



| ​                          | LogiQA-v2​ |  C3-d​ |  C3-m​ |
|----------------------------|--------|-------|--------|
| BELLE-LLaMA-Ext-7B​         	| 26.41​ | 29.52​ | ​28.87​ |
| Linly-Chinese-LLaMA-7b-hf​ | 25.91​ | 32.28​ | 34.52​ |
| Panda-7B​ (ours)                     	| 27.41​ | 43.02​ | 43.66​ |
| Panda-Instruct-7B 3k steps​ (ours)                	| 26.22​ | 39.05​ | 42.11​ |
| Panda-Instruct-7B 6k steps​ (ours)                  | 30.30​ | 47.14​ | ​56.94​ |
| Panda-Instruct-7B 9k steps (ours)​           | **31.93**​ | **47.30**​ | **57.04**​ |
| Panda-13B (Ours)          | 24.72 | **47.94** | 49.71 |

**Notes**
1. 由于模型对 instruction 的敏感性不同测评结果可能会有较大波动，测评结果仅供参考，并且可能无法完全反应模型之间的优劣。我们对于所有模型采用了最简单的 instruction（可以在对应数据集配置文件中找到）。

2. Linly-Chinese 在训练时可能在 instruction 中添加了额外的前缀（如用 assistant 和 user 去区分对话中的角色），在测试时对齐这一点可能会进一步提升性能，但我们目前为了统一instruction 并没有对齐。后续我们考虑收集多样化的 instruction 进行评测并汇报平均值。


<h2 id="contribute">如何参与</h2>

开发者可以通过贡献有用的代码、数据、论文和计算资源等方式成为贡献者。

代码：包括算法实现、训练优化、推理优化和模型部署。

数据：每个研究领域和版本迭代都需要高质量的数据，包括指令-答案、预训练、多模态、多语言和用户反馈等数据。

论文：我们将维护一个 Panda 论文列表，并使用 Panda 作为优化、完全测试和显著改进的学术论文的基础模型。

计算资源：我们希望通过协调一些开发者的冗余计算能力或从大学/企业获得非营利性赞助来帮助加速模型迭代速度。



<h2 id="acknowledge">鸣谢</h2>

我们非常感谢国内的一些大企业支持，为我们提供大量 GPU 来支持我们的模型训练。这些 GPU 的高性能计算能力为我们在 Panda 模型的研究和开发工作提供了强大的支持。

<h2 id="acknowledge"> 开发者</h2>

Fangkai Jiao   
Bosheng Ding  
Tianze Luo  
Zhanfeng Mo  
Chengwei Qin

<h2 id="acknowledge"> 项目总负责人</h2>

Bosheng Ding

<h2 id="acknowledge"> 项目运营助理</h2>

Haichao Yu

<h2 id="acknowledge"> 联系我们</h2>

dandelionsllm@gmail.com

<h2 id="acknowledge"> 项目总部</h2>

新加坡

### 免责声明

我们要求开发者不得将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于任何商业以及为社会带来危害的用途。由 Panda 和 Flan-LLaMA 任何模型生成的内容均受随机性和不可控因素的影响，本项目无法保证其准确性。本项目不承担任何关于模型输出内容的法律责任，也不对使用相关资源和输出结果可能导致的任何损失承担责任。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dandelionsllm/pandallm&type=Date)](https://star-history.com/#dandelionsllm/pandallm&Date)

