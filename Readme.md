<h1 align="center">
  Panda中文开源大语言模型
</h1>
<p align="center" width="100%">
  <img src="panda_logo.PNG" alt="Panda" style="width: 20%; display: block; margin: auto;"></a>
</p>
<p align="center">
  <font face="黑体" color=orange size="6"> PandaLLM系列中文开源大模型 </font>
</p>
<p align="center">
  <font face="黑体" color=orange size="6"> PandaLLMOps开源大模型训练、推理、部署工具 </font>
</p>
<p align="center">
  <font face="黑体" color=orange size="6"> PandaCommunity中文开源大模型开发者社区 </font>
</p>
<p align="center">
  <a href="http://pandallm.ai/">在线体验：pandallm.ai (Working in Process)</a>
</p>
 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
</br></br>
## 📄 项目介绍
欢迎来到我们的海外中文大语言模型开源项目 -- Panda！Panda项目于2023年5月启动，旨在大模型时代帮助整个社区探索大模型的整个技术栈。近期，我们对Panda项目进行了升级。目前Panda项目分为三个子项目：1. PandaLLM 2. PandaLLMOPs 3. PandaCommunity。以下是每个子项目的具体介绍：

1. PandaLLM 开源大模型。Panda系列开源大模型目前基于 LLaMA1 和 LLaMA2 进行中文领域上的持续预训练，我们希望能够为中文自然语言处理领域提供具有泛用性的通用基础工具进行探索。PandaLLM 模型以及训练涉及的中文数据集将以开源形式发布，任何人都可以免费使用并参与开发。

2. PandaLLMOps 开源大模型训练、推理、部署工具。PandaLLMOps是一套集成了大模型从模型训练、推理、部署工具。我们希望可以为广大大模型开发人员、爱好者提供一套好用的工具，降低学习难度，提高大家在大模型开发、推理、部署各个环节的效率。目前支持以下场景：1. 从零开始做预训练 2. 基于现有底座做增量预训练、全参数量微调、Lora、QLora 3. 快速部署，已集成vllm、lightllm等，最大化优化推理速度

4. PandaCommunity 大模型中文社区。PandaCommunity旨在建立中文世界大模型开发者学习社区，让大家可以对大模型技术栈能够有深入的了解并且互相交流，发挥集体智慧，攻克大模型探索道路上的技术难关。同时，针对想要学习大模型技术的同学，我们也会定期推出免费技术教程、技术研讨会、论文解读等等。

我们欢迎来自全球的开发者一起参与到该项目中，共同推动自然语言处理技术的发展。



## 🧭 内容导引
- [🐼 PandaLLM](#-PandaLLM)
  - [🔥 最新PandaLLM-LLaMA2-13B上线](#最新pandallm-llama2-13b上线)
  - [🍞 PandaLLM 已发布的各版本模型权重](#PandaLLM已发布的各版本模型权重)
  - [🤖 PandaLLM 训练框架](#PandaLLMOps训练框架)
  - [📒 PandaLLM 开源训练语料总结](#PandaLLM开源训练语料总结)
- [🐼 PandaLLMOps](#-PandaLLMOps)
  - [🔨 PandaLLMOps 工具介绍](#-PandaLLMOps工具介绍)
  - [🤠 PandaLLMOps 预训练示例](#-PandaLLMOps预训练示例)
  - [🤗 PandaLLMOps 全参数微调示例](#-PandaLLMOps全参数微调示例)
  - [😎 PandaLLMOps Lora示例](#-PandaLLMOps-Lora示例)
  - [⏩ PandaLLMOps 流水线并行示例](#-PandaLLMOps流水线并行示例)
  - [🫡 PandaLLMOps 部署示例](#-PandaLLMOps部署示例)
  - [🚀 PandaLLMOps 中英双语Tutorial上线](#-PandaLLMOps-Tutorial)
- [🐼 PandaCommunity 大模型中文社区](#-PandaCommunity大模型中文社区)
  - [🔥 社区介绍](#社区介绍)
  - [🌍 为什么选择PandaCommunity中文社区？](#为什么选择PandaCommunity中文社区)
  - [🎉 社区活动](#社区活动)
  - [🍻 加入我们！](#加入我们)
- [📢 社区公告](#-社区公告)
  - [🔥Panda项目最新进展](#Panda项目最新进展)
- [📖 学习资料](#-学习资料)
  - [💡 原创学习资料](#原创学习资料)
  - [📚 LLM 相关论文](#LLM相关论文)

- [🎉 致谢](#-致谢)
- [🤔 问题反馈](#-问题反馈)

## 🐼 PandaLLM

### 最新PandaLLM-LLaMA2-13B上线
我们最新基于LLaMA2-13B的底座模型在中文数据进行了增量预训练。这项研究是为了进一步提升我们的自然语言处理技术，以更好地适应不断变化的语言环境和应用需求。在这次的增量预训练中，我们选择了大量丰富多样的中文数据，包括文本、对话、新闻文章和社交媒体内容，以增强模型对中文语境的理解和表达能力。该底座模型的LLaMA2-13B架构在之前的研究中已经表现出良好的性能和可扩展性，因此我们选择该模型作为基础，通过增量预训练的方式来进一步优化其效果。

通过在大规模的中文数据上进行增量预训练，我们的底座模型现在具备更深入、更全面的中文语言知识，可以更好地理解中文的语法结构、语义含义以及上下文关联。这为我们在各种中文自然语言处理任务中取得更优秀的结果打下了坚实基础。未来，我们将继续致力于推进中文自然语言处理领域的研究，进一步改进和优化底座模型，并探索更多创新的方法来处理中文语言的特点和复杂性。我们相信，随着技术的不断进步，我们的中文自然语言处理技术将在诸多领域发挥重要作用，为用户提供更智能、更便捷的语言交互体验。


###  PandaLLM已发布的各版本模型权重


可商用（基于LLaMA2和OpenLLaMA底座微调)

|  模型名称      | 模型大小 | 下载链接                                            |
| --------------- | ---------- | -------------------------------------------------------- |
| Panda-LLaMA2-13B | 13B      | [https://huggingface.co/chitanda/panda-7b-open-llama-preview-300pt](https://huggingface.co/chitanda/panda-7b-open-llama-preview-300pt) |
| Panda-LLaMA2-13B-Chat | 13B      | [https://huggingface.co/chitanda/panda-7b-open-llama-preview-300pt](https://huggingface.co/chitanda/panda-7b-open-llama-preview-300pt) |
| Panda-OpenLLaMA-7B | 7B      | [https://huggingface.co/chitanda/panda-7b-open-llama-preview-300pt](https://huggingface.co/chitanda/panda-7b-open-llama-preview-300pt) |

不可商用（基于LLaMA1底座微调)

|  模型名称      | 模型大小 | 下载链接                                            |
| --------------- | ---------- | -------------------------------------------------------- |
| Panda-7B        | 7B         | https://huggingface.co/chitanda/llama-panda-zh-7b-delta   |
| Panda-Instruct-7B | 7B       | https://huggingface.co/chitanda/llama-panda-zh-coig-7b-delta |
| Panda-13B       | 13B        | https://huggingface.co/chitanda/llama-panda-zh-13b-delta                 |
| Panda-Instruct-13B | 13B     | [https://huggingface.co/chitanda/llama-panda-zh-13b-coig-delta](https://huggingface.co/chitanda/llama-panda-zh-13b-coig-delta) |
| Flan-LLaMA-7B   | 7B         | https://huggingface.co/NTU-NLP-sg/flan-llama-7b-10m-delta  |
| Panda-13B-Chat  | 13B        | [https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta](https://huggingface.co/chitanda/llama-panda-13b-zh-wudao-chat-delta) |

**Notes**: 
1. 因为 LLaMA1 权重 License 的存在，我们无法直接发布完整的模型权重，因此我们放出了训练后模型的权重与原始 LLaMA 权重的差，保证能够获得 LLaMA 权重的用户能够同样使用这些模型。我们提供了一个[脚本](https://github.com/dandelionsllm/pandallm/blob/main/apply_delta.py)来帮助转换。这个问题在LLaMA2出现后得到了解决。
2. 由于模型训练期间使用了 `bfloat16`，在非安培架构的显卡上直接使用 `fp16` 格式进行微调时可能会出现无法收敛的情况，需要额外注意。
3. 针对Panda-OpenLLaMA，在训练过程中我们发现其需要接近两倍的训练时长，且最后确认不是计算节点通信的问题，我们怀疑是OpenLLaMA的模型精度问题导致了训练降速，且同时这个问题可能在微调阶段依然存在，但我们目前没有时间去调研，如果有朋友发现了原因，欢迎指出，感谢。


###  PandaLLM 训练框架
我们使用了 Deepspeed Zero-1 + Gradient Checkpointing 作为PandaLLM的训练框架，更多详情请参考[PandaLLMOps的内容和Tutorial](#-PandaLLMOps)。

- 模型训练

在进行模型训练前，需要先配置相应的超参数。以下是不同模型的训练超参数对应的配置文件路径:

```
# LLaMA-7b pretrain on general Chinese Corpus
conf/llama/zh/llama_7b_zh_instruct_v1_0_ds.yaml

# LLaMA-7b instruction tuning on COIG
conf/llama/zh/llama_7b_zh_instruct_coig_sft_v1_0_ds.yaml

# LLaMA-13b pretrain on general Chinese Corpus
conf/llama/zh/llama_13b_zh_instruct_v1_0_ds.yaml
```

- 模型训练命令
```
HYDRA_FULL_ERROR=1 deepspeed --include localhost:0,1,2,3,4,5,6,7 trainer_base_ds_mul.py -cp conf/llama/zh -cn <yaml 配置文件名> 
```
其中
  - `HYDRA_FULL_ERROR=1`: 这个参数用于显示详细的错误信息，有助于调试训练过程中可能遇到的问题。
  - `deepspeed`: 指定使用 DeepSpeed 分布式训练框架。
  - `--include localhost:0,1,2,3,4,5,6,7`: 设置需要使用的 GPU 设备，这里使用了 2 * 8 * A100 80G 的 GPU，可以根据具体硬件情况进行调整。
  - `trainer_base_ds_mul.py`: 训练脚本的文件名。
  - `-cp conf/llama/zh`: 设置配置文件的基本路径为 conf/llama/zh。
  - `-cn <yaml 配置文件名>`: 指定使用哪个 yaml 格式的配置文件进行训练，根据实际情况填写对应的文件名。


- GPU 资源调整

如果您的设备显卡数量较少，请根据实际情况相应调整以下两个超参数：

  - `gradient_accumulation_steps`: 梯度累积步数，可以设置为较大的整数值来弥补显卡数量不足的情况。
  - `per_gpu_train_batch_size`: 每个 GPU 上的训练批大小，根据显存大小适当调整以避免内存溢出。
  通过遵循以上指南，您可以使用 Deepspeed Zero-1 + Gradient Checkpointing 模型训练框架来训练 LLaMA-7b 和 LLaMA-13b 模型，并根据实际硬件资源来进行调整，以实现高效的模型训练。祝您训练顺利，取得优秀的结果！

### PandaLLM开源训练语料总结

模型数据现阶段均采用开源的公开中英文语料数据集：

#### 中文 instruction-tuning

- [维基百科(wiki2019zh)，100万个结构良好的中文词条](https://github.com/brightmart/nlp_chinese_corpus)  
- [新闻语料(news2016zh)，250万篇新闻，含关键词、描述](https://github.com/brightmart/nlp_chinese_corpus)  
- [百科问答(baike2018qa)，150万个带问题类型的问答](https://github.com/brightmart/nlp_chinese_corpus)  
- [社区问答json版(webtext2019zh)，410万个高质量社区问答，适合训练超大模型](https://github.com/brightmart/nlp_chinese_corpus)  
- [翻译语料(translation2019zh)，520万个中英文句子对](https://github.com/brightmart/nlp_chinese_corpus)  
- [Chinese Open Instruction Generalist (COIG)](https://huggingface.co/datasets/BAAI/COIG) 

**Notes**
1. 对于除维基百科和新闻语料外的其他语料，用 Conditional Generation 的方式优化，即 instruction 部分与输入部分不计算损失，只计算输出部分的损失。除 COIG 外的语料中的 instruction 为固定模板。
2. 一开始我们将以上所有语料混合在一起进行训练，但发现最终的模型在 instruction following 方面的能力并不好，因此我们决定单独在 COIG 数据集上进行指令微调，并得到最终模型。推测原因可能是 COIG 在整体训练数据中的占比过小，可选的解决方案是对 COIG 加大采样的概率。


#### 英文 instruction-tuning

为了提升模型的基础能力，我们选择使用 FLAN Collection 进行训练。由于 FLAN collection 语料规模过于庞大，我们按比例抽取了 7M 的语料用于训练，且最终性能仍远落后于 FLAN-T5-3B，因此目前我们决定暂时停止该方向的训练，并思考其他可能的构建较小的同时具有较强基础能力的语言模型的方向。


## 🐼 PandaLLMOPs

### PandaLLMOps 工具介绍
PandaLLMOps是一款开源的大模型训练、推理和部署工具。该工具集成了大模型从训练到推理再到部署的全流程支持。我们致力于为广大大模型开发人员和爱好者提供一套简便易用的工具，以降低学习门槛，提高在大模型开发、推理和部署过程中的效率。

目前，PandaLLMOps支持多种场景，包括：

1. **从零开始做预训练**：您可以使用PandaLLMOps来进行大规模预训练，从而让模型掌握更丰富的语言知识和表达能力。

2. **基于现有底座做增量预训练、全参数量微调、Lora、QLora**：PandaLLMOps提供了灵活的增量预训练、微调和Lora等功能，帮助您在已有底座模型的基础上进行更多样化和高效的模型优化。

3. **快速部署**：PandaLLMOps集成了vllm、lightllm等推理引擎，可以最大化优化推理速度，助力您在实际应用中快速部署和运行大模型。

我们希望PandaLLMOps能够为您提供强大而便捷的工具，使您能够更加专注于大模型的开发和创新，为自然语言处理和相关领域带来更加出色的成果。欢迎您加入我们的开源社区，共同推进大模型技术的发展，谢谢！

### PandaLLMOps预训练示例
Work in Progress

### PandaLLMOps全参数微调示例
Work in Progress

### PandaLLMOps-Lora示例
Work in Progress

### PandaLLMOps流水线并行示例
Work in Progress

### PandaLLMOps部署示例
Work in Progress

### PandaLLMOps-Tutorial
我们上线了PandaLLMOps第一版Tutorial，希望可以能够给广大使用者起到答疑解惑的作用。如果您有更多的疑问，可以提交Github Issues或者加入我们的PandaCommunity微信群。
[https://panda-tutorial.readthedocs.io/en/latest/index.html](https://panda-tutorial.readthedocs.io/en/latest/index.html)

## 🐼 PandaCommunity大模型中文社区
### 社区介绍
PandaCommunity是大模型中文社区的先锋组织。我们致力于为中文世界的大模型开发者打造一个互联互通的学习平台，让每一位对大模型技术栈感兴趣的人都能深入了解和互相交流。通过集体智慧的力量，我们一同攻克大模型探索道路上的技术难关。而针对那些渴望学习大模型技术的同学，我们还会定期推出免费技术教程、技术研讨会、论文解读等活动。在PandaCommunity中，我们一同成长、一同进步，共同推动大模型技术在中文世界的发展与创新。我们欢迎对大模型LLM充满热情的开发者和研究者加入我们的行列。

### 为什么选择PandaCommunity大模型中文社区？

🚀 **专业技术团队陪伴**：在PandaCommunity中，有一群专业的NLP研究人员和高级工程师随时为您提供指导和帮助。无论您是新手还是资深开发者，我们都会为您提供强大的技术支援。

📚 **深入教程与研讨**：想要学习大模型技术？我们定期推出免费的技术教程、研讨会、论文解读等活动，助您深入了解大模型技术栈，快速提升技能。

🎯 **中文社群**：PandaCommunity致力于大语言模型的中文优化，探索和实践最佳方案，让中文大模型落地更高效、更精确。

💡 **交流创新无界限**：我们鼓励和促进创新交流，无论是线上活动还是技术研讨，都能让您和其他有经验的社区成员互动，共同探索和学习。

🌐 **全球互联，共同成长**：我们欢迎全球的开发者加入PandaCommunity，共同构建一个开放、多元的学习和交流平台。让我们在这里一同成长、一同进步。

🤝 **开源精神，共赢未来**：开源分享是我们的信念，无论是代码还是模型，我们都鼓励您共享，与全球开发者共同推动中文NLP技术的进步。



### 社区活动

🗓️ **在线讲座**：我们会请来行业专家举办在线讲座，深入讲解大模型技术的最新进展和使用，以及讨论最前沿的研究成果。

💻 **项目展示区**：允许成员展现他们在大模型中文优化方面的成就，以便获取反馈与指导，进一步推动项目合作。

📚 **教育资源**：社区整合了包括教程、文件和论文解析在内的丰富学习资源，以便为成员提供完整的学习支援。

📝 **论文深入解析**：共同探讨和理解与大模型相关的最新科研论文，深化对前沿算法和方法的认识。

🎉 **各类主题活动**：社区会定期组织各种主题活动，例如挑战赛、黑客马拉松和技术沙龙，营造轻松的学习和交流环境。

🌟 **激励方案**：我们有奖励方案，对于社区中积极和出色的贡献者提供荣誉和奖励，以鼓励更多的优秀人才加入我们。

📈 **专业技术咨询**：提供技术咨询服务，帮助您解决在大模型开发和优化过程中可能遇到的问题，协助您迅速突破技术难题。

🚀 **合作项目**：我们积极推动成员之间的项目合作，共同探寻大模型在实际应用场景中的潜力，并开发创新的解决方案。


### 加入我们
🚀 **社区愿景**：PandaCommunity的愿景是成为连接中文世界大模型开发者的桥梁，打造一个充满活力和创新的学习社区。我们追求卓越，专注于大模型的中文处理和优化，致力于推动大模型技术在中文世界的不断发展和创新。我们坚信，通过专业技术团队的支持、深入的教程与研讨、全球互联的交流、以及开源共享的精神，我们能够共同攻克大模型技术的难题，激发更多的创新潜能。在PandaCommunity中，每一位成员都是重要的一环，每一项贡献都是推动中文NLP技术发展的动力。我们一起学习，一起成长，一起创造中文大模型技术的美好未来。

🔗 **温馨提示**：本社区是一个致力于专业技术交流的平台，我们真诚地欢迎志同道合的开发者和研究者加入。请遵守社区准则，一起维护积极向上的学习氛围，任何与大模型无关的内容和广告将会被清理。感谢您的理解和支持！


## 📢 社区公告
### Panda项目最新进展
- **2023/08/03**: 基于PandaLLM-LLaMA2-13B的Chat版本开始训练
- **2023/08/02**: 基于LLaMA2-13B进行中文数据全参数微调的PandaLLM-LLaMA2-13B训练完成
- **2023/07/25**: 基于LLaMA2-13B进行中文数据全参数微调的PandaLLM-LLaMA2-13B词表扩充完成
- **2023/07/22**: 基于LLaMA2-13B进行中文数据全参数微调的PandaLLM-LLaMA2-13B启动
- **2023/07/17**: PandaLLMOps Tutorial英文版上线
- **2023/07/13**: PandaLLMOps Tutorial立项
- **2023/07/12**: 更新了LLaMA系列和MPT系列基于Pipeline Parallelism训练的代码以及配置文件。
- **2023/07/02**: 我们开源了一个LLaMA的流水线并行的[原型库](https://github.com/SparkJiao/LLaMA-pipeline-parallel)，目的是为了解决开源社区内基于DeepSpeed和Pytorch训练超过30B模型的训练方案缺失的问题。我们目前能够在单节点和双节点上训练65B的模型。我们会以此为起点尝试训练33B模型。欢迎训练过程中遇到问题的[讨论](https://github.com/SparkJiao/LLaMA-pipeline-parallel/discussions)。
- **2023/06/24**: Panda-13B-Chat权重发布。推理优化，会尽快上线新的体验页面。
- **2023/06/12**: Panda-13B-Chat目前可以在[t.me/PandaLLMChat_bot](https://t.me/PandaLLMChat_bot)体验。需要代理和Telegram。目前训练还未完成，我们会在训练完成后尽快发布权重。
- **2023/05/28**: 使用可商用License的Open-LLaMA-Preview-300BT的模型进行中文持续训练的模型已经放出。目前我们正在准备相关的测评数据集以及工具，在完成后会统一进行测评。发布可商用Licence模型的初衷是尽可能规避限制，同时从中文可迁移性的角度上对现有的无限制开源LLaMA模型进行评估。我们的下一步目标是基于现有的Instruction tuning数据和Panda-13B训练一个更好的Chat模型，以满足个人开发者的需求。目前30B模型训练暂时存在一定困难（时间较长，预计迁移过程需要一个月），我们会积极寻找解决方案（包括尝试募集更多的资源，寻找其他合适的底座模型，以及评估LoRA在基础迁移上的性能等）。
- **2023/05/12**: Panda-13B-COIG权重发布，后续将更新测评成绩。我们下一步计划是基于Open-LLaMA预训练可商用的7B规模基础模型，同时引入更多的中文语料。
- **2023/05/09**: Panda-13B权重发布并更新测评成绩。Panda-13B-Instruct开始训练。
- **2023/05/08**: FLAN-LLaMA-7b-10M is released. But the performance is not as good as FLAN-T5-3b. Panda-13B training is over and we will release the weights asap.
- **2023/05/03**: 基于LLaMA-7B训练的Panda-7B权重发布


## 📖 学习资料
### 原创学习资料
即将上线，敬请期待！

### LLM相关论文
有关LLM相关的论文请参考以下Github Repo的总结：
[https://github.com/Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)

## 🎉 致谢

我们非常感谢国内的一些大企业支持，为我们提供大量 GPU 来支持我们的模型训练。这些 GPU 的高性能计算能力为我们在 Panda 模型的研究和开发工作提供了强大的支持。我们也感谢以下社区和机构对我们的支持（排名不分先后）。

- AWS中国
- CSDN
- Huggingface
- 思否编程

## 🤔 问题反馈

开源不易，请多鼓励。如有问题，请在GitHub Issues中提交，在提交问题之前，请先查阅以往的issue是否能解决你的问题。

加入飞书知识库(正在构建中)，一起共建社区文档。

加入🐼微信群讨论（即将开放）。

## 免责声明

我们要求开发者不得将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于任何商业（开源可商用版本除外）以及为社会带来危害的用途。由 Panda 和 Flan-LLaMA 任何模型生成的内容均受随机性和不可控因素的影响，本项目无法保证其准确性。本项目不承担任何关于模型输出内容的法律责任，也不对使用相关资源和输出结果可能导致的任何损失承担责任。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dandelionsllm/pandallm&type=Date)](https://star-history.com/#dandelionsllm/pandallm&Date)
