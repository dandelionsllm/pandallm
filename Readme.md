<p align="center" width="100%">
<a ><img src="src/imgs/panda.png" alt="Panda" style="width: 60%; min-width: 300px; display: block; margin: auto;"></a>
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)


# Panda: 海外中文开源大语言模型

欢迎来到我们的海外中文大语言模型开源项目—— Panda！该项目旨在提供一款开源、高质量的中文大语言模型，能够支持各种自然语言处理任务, 并且特别注重海外华人使用体验。

Panda 语言模型立足于 Llama-7B,  -13B 架构, 并在 xxx 的开源中文文本数据集上面进行了大规模训练和优化。Panda 语言模型更注重覆盖全球华人使用场景，并致力于提供高质量的中文语言模型，使得全球华人能够更加便捷地进行各种自然语言处理任务。

我们的 Panda 模型以及训练涉及的中文数据集将以开源形式发布，任何人都可以免费使用并参与开发。我们欢迎来自全球的开发者一起参与到该项目中，共同推动中文自然语言处理技术的发展。

## 目录

1. [最近更新](#news)

2. [项目内容](#model)

3. [实验结果](#evaluation)

4. [模型部署](#usage)

5. [如何参与](#contribute)

6. [鸣谢](#acknowledge)


<h2 id="news">最近更新</h2>

发布了大模型Panda 和 Guanaco的technical report！
论文链接： 

如何引用我们：

<h2 id="model">项目内容</h2>

## Panda 模型
详见Panda/train，我们集成了Deepspeed，支持模型pretrain，finetune，lora (后续推出)，distillation (后续推出)

我们目前开放基于中英文语料库的与训练与调优模型：Panda-7B 和 Panda-13B。

## Guanaco 模型

详见Guanaco/train。模型训练样本基于Flan 数据集。我们集成了Deepspeed，支持模型pretrain，finetune，lora (后续推出)，distillation (后续推出)


## 数据
模型数据现阶段均采用开源的公开中英文语料数据集：

### 中文预训练

### 中文 instruction-tuning

### 英文预训练

### 英文 instruction-tuning


## 训练方法

### 模型预训练

我们采用了。。。

### Instruction tuning

训练参数设置。。。

 
<h2 id="evaluation">实验结果</h2>

<h2 id="usage">模型部署</h2>

### CPU 部署

### GPU/集群部署



<h2 id="contribute">如何参与</h2>

开发者可以通过贡献有用的代码、数据、论文和计算资源等方式成为贡献者。

代码：包括算法实现、训练优化、推理优化和模型部署。

数据：每个研究领域和版本迭代都需要高质量的数据，包括指令-答案、预训练、多模态、多语言和用户反馈等数据。

论文：我们将维护一个 Panda 论文列表，并使用 Panda 作为优化、完全测试和显著改进的学术论文的基础模型。

计算资源：我们希望通过协调一些开发者的冗余计算能力或从大学/企业获得非营利性赞助来帮助加速模型迭代速度。

具体操作请参考：。。。


<h2 id="acknowledge">鸣谢</h2>

我们非常感谢国内的一些大企业支持，为我们提供大量GPU来支持我们的模型训练。这些GPU的高性能计算能力为我们在Panda模型的研究和开发工作提供了强大的支持。

### 免责声明

我们要求开发者不得将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于任何为社会带来危害的用途。由Panda 任何模型生成的内容均受随机性和不可控因素的影响，本项目无法保证其准确性。本项目不承担任何关于模型输出内容的法律责任，也不对使用相关资源和输出结果可能导致的任何损失承担责任。
