<p align="center" width="100%">
<a ><img src="src/imgs/panda.png" alt="Panda" style="width: 60%; min-width: 300px; display: block; margin: auto;"></a>
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)





# Panda: 海外中文开源大语言模型

欢迎来到我们的海外中文大语言模型开源项目—— Panda！该项目旨在提供一款开源、高质量的中文大语言模型，能够支持各种自然语言处理任务, 并且特别注重海外华人使用体验。

Panda 语言模型目前立足于 Llama-7B,  -13B 架构, 并在 xxx 的开源中文文本数据集上面进行了大规模训练和优化。Panda 语言模型更注重覆盖全球华人使用场景，并致力于提供高质量的中文语言模型，使得全球华人能够更加便捷地进行各种自然语言处理任务。未来会切换到新的底座模型或者训练自己的底座模型。

我们的 Panda 模型以及训练涉及的中文数据集将以开源形式发布，任何人都可以免费使用并参与开发。我们欢迎来自全球的开发者一起参与到该项目中，共同推动中文自然语言处理技术的发展。

## 目录

1. [最近更新](#news)

2. [项目内容](#model)

3. [实验测评](#evaluation)

4. [模型部署](#usage)

5. [如何参与](#contribute)

6. [鸣谢](#acknowledge)


<h2 id="news">最近更新</h2>

发布了大模型Panda 和 Flan-Lamma的technical report！

论文链接： 


如何引用我们：


<h2 id="model">项目内容</h2>

### Panda 模型
详见Panda/train，我们集成了Deepspeed，支持模型pretrain，finetune，lora (后续推出)，distillation (后续推出)

我们目前开放基于中英文语料库的与训练与调优模型：Panda-7B 和 Panda-13B。


### Flan-Lamma 模型

详见Flan_Lamma/train。模型训练样本基于Flan 数据集。我们集成了Deepspeed，支持模型pretrain，finetune，lora (后续推出)，distillation (后续推出)


模型版本：


|  模型名称      | 模型大小 | 下载链接                                            |
| --------------- | ---------- | -------------------------------------------------------- |
| Panda-7B        | 7B         | https://huggingface.co/chitanda/llama-panda-zh-7b-delta   |
| Panda-Instruct-7B | 7B       | https://huggingface.co/chitanda/llama-panda-zh-coig-7b-delta |
| Panda-13B       | 13B        |                       |
| Flan-Lamma-7B   | 7B         |                                  |



Note 1: 因为LLaMA权重License的存在，我们无法直接发布完整的模型权重，因此我们放出了训练后模型的权重与原始LLaMA权重的差，保证能够获得LLaMA权重的用户能够同样使用这些模型。我们提供了一个脚本来帮助转换。  
Note 2: 由于模型训练期间使用了`bfloat16`，在非安培架构的显卡上直接使用`fp16`格式进行微调时可能会出现无法收敛的情况，需要额外注意。

## 数据
模型数据现阶段均采用开源的公开中英文语料数据集：


### 中文 instruction-tuning

- 维基百科(wiki2019zh)，100万个结构良好的中文词条  
- 新闻语料(news2016zh)，250万篇新闻，含关键词、描述  
- 百科问答(baike2018qa)，150万个带问题类型的问答  
- 社区问答json版(webtext2019zh)，410万个高质量社区问答，适合训练超大模型  
- 翻译语料(translation2019zh)，520万个中英文句子对  
- Chinese Open Instruction Generalist (COIG) 

Notes 1: 对于除维基百科和新闻语料外的其他语料，用Conditional Generation的方式优化，即instruction部分与输入部分不计算损失，只计算输出部分的损失。除COIG外的语料中的instruction为固定模板。


Notes 2: 一开始我们将以上所有语料混合在一起进行训练，但发现最终的模型在instruction following方面的能力并不好，因此我们决定单独在COIG数据集上进行指令微调，并得到最终模型。推测原因可能是COIG在整体训练数据中的占比过小，可选的解决方案是对COIG加大采样的概率。


### 英文 instruction-tuning

为了提升模型的基础能力，我们选择使用FLAN Collection进行训练。由于FLAN collection语料规模过于庞大，我们按比例抽取了7M的语料用于训练，且最终性能仍远落后于FLAN-T5-3B，因此目前我们决定暂时停止该方向的训练，并思考其他可能的构建较小的同时具有较强基础能力的语言模型的方向。

## 训练框架

Deepspeed Zero-1 + Gradient Checkpointing

### 模型训练

对应模型的训练时超参数见：


### LLaMA-7b pretrain on general Chinese Corpus

conf/llama/zh/llama_7b_zh_instruct_v1_0_ds.yaml

### LLaMA-7b instruction tuning on COIG

conf/llama/zh/llama_7b_zh_instruct_coig_sft_v1_0_ds.yaml

### LLaMA-13b pretrain on general Chinese Corpus (Ongoing)

conf/llama/zh/llama_13b_zh_instruct_v1_0_ds.yaml


<h2 id="evaluation">实验测评</h2>

### 基础能力测评

#### 测评数据集

##### 复杂推理

- [LogiQA-v2](https://github.com/csitfun/LogiQA2.0)
- [C3](https://dataset.org/c3/)

### 其他能力

测试进行中


### Baseline

- [BELLE-LLaMA-Ext-7B](https://github.com/LianjiaTech/BELLE)
- [Linly-Chinese-LLaMA-7b-hf](https://github.com/CVI-SZU/Linly) (Huggingface weights of chat-based model in 7B size are not released now. Corresponding results will be updated after weights are released)

### Results
 



| ​                          | LogiQA-v2​ |  C3-d​ |  C3-m​ |
|----------------------------|--------|-------|--------|
| llama-zh​                        	| 27.41​ | 43.02​ | 43.66​ |
| llama-zh-instruct (9k)​            | 31.93​ | 47.30​ | 57.04​ |
|   2000 steps​                  	| 26.22​ | 39.05​ | ​42.11​ |
|   3000 steps​                 	| 26.22​ | 39.05​ | 42.11​ |
|   6000 steps​                  	| 30.30​ | 47.14​ | ​56.94​ |
| belle-llama-ext-7b​         	| 26.41​ | 29.52​ | ​28.87​ |
| Linly-Chinese-LLaMA-7b-hf​ | 25.91​ | 32.28​ | 34.52​ |

Note 1: 由于模型对instruction的敏感性不同测评结果可能会有较大波动，测评结果仅供参考，并且可能无法完全反应模型之间的优劣。我们对于所有模型采用了最简单的instruction（可以在对应数据集配置文件中找到）。
Note 2: Linly-Chinese可能可能在训练时用了额外的前缀（如assistant和user去区分对话中的角色），这可能会进一步提升性能，但我们目前没有测试。后续我们考虑收集多样化的instruction进行评测并汇报平均值。


<h2 id="contribute">如何参与</h2>

开发者可以通过贡献有用的代码、数据、论文和计算资源等方式成为贡献者。

代码：包括算法实现、训练优化、推理优化和模型部署。

数据：每个研究领域和版本迭代都需要高质量的数据，包括指令-答案、预训练、多模态、多语言和用户反馈等数据。

论文：我们将维护一个 Panda 论文列表，并使用 Panda 作为优化、完全测试和显著改进的学术论文的基础模型。

计算资源：我们希望通过协调一些开发者的冗余计算能力或从大学/企业获得非营利性赞助来帮助加速模型迭代速度。



<h2 id="acknowledge">鸣谢</h2>

我们非常感谢国内的一些大企业支持，为我们提供大量GPU来支持我们的模型训练。这些GPU的高性能计算能力为我们在Panda模型的研究和开发工作提供了强大的支持。

<h2 id="acknowledge"> 开发者</h2>

Fangkai Jiao  
Bosheng Ding  
Tianze Luo  
Zhanfeng Mo  


### 免责声明

我们要求开发者不得将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于任何商业以及为社会带来危害的用途。由Panda和Guacuna 任何模型生成的内容均受随机性和不可控因素的影响，本项目无法保证其准确性。本项目不承担任何关于模型输出内容的法律责任，也不对使用相关资源和输出结果可能导致的任何损失承担责任。
