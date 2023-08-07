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
欢迎来到我们的海外中文大语言模型开源项目 -- Panda！Panda项目于2023年5月启动，旨在大模型时代帮助整个社区探索大模型的整个技术栈。近期，我们对Panda项目进行了升级。目前Panda项目分为三个子项目：1. PandaLLM 2. PandaLLMOps 3. PandaCommunity。以下是每个子项目的具体介绍：

1. PandaLLM 开源大模型。Panda系列开源大模型目前基于 LLaMA1 和 LLaMA2 进行中文领域上的持续预训练，我们希望能够为中文自然语言处理领域提供具有泛用性的通用基础工具进行探索。PandaLLM 模型以及训练涉及的中文数据集将以开源形式发布，任何人都可以免费使用并参与开发。

2. PandaLLMOps 开源大模型训练、推理、部署工具。PandaLLMOps是一套集成了大模型从模型训练、推理、部署工具。我们希望可以为广大大模型开发人员、爱好者提供一套好用的工具，降低学习难度，提高大家在大模型开发、推理、部署各个环节的效率。目前支持以下场景：1. 从零开始做预训练 2. 基于现有底座做增量预训练、全参数量微调、Lora、QLora 3. 快速部署，已集成vllm、lightllm等，最大化优化推理速度

4. PandaCommunity 大模型中文社区。PandaCommunity旨在建立中文世界大模型开发者学习社区，让大家可以对大模型技术栈能够有深入的了解并且互相交流，发挥集体智慧，攻克大模型探索道路上的技术难关。同时，针对想要学习大模型技术的同学，我们也会定期推出免费技术教程、技术研讨会、论文解读等等。

我们欢迎来自全球的开发者一起参与到该项目中，共同推动自然语言处理技术的发展。



## 🧭 内容导引
- [🐼 PandaLLM](#-pandallm)
  - [🔥 最新PandaLLM-LLaMA2-13B上线](#最新pandallm-llama2-13b上线)
  - [🍞 PandaLLM 已发布的各版本模型权重](#pandallm已发布的各版本模型权重)
  - [🤖 PandaLLM 训练框架](#pandallmops训练框架)
  - [📒 PandaLLM 开源训练语料总结](#pandallm开源训练语料总结)
- [🐼 PandaLLMOps](#-pandallmops)
  - [🔨 PandaLLMOps 工具介绍](#pandallmops工具介绍)
  - [🤠 PandaLLMOps 预训练示例](#pandallmops预训练示例)
  - [🤗 PandaLLMOps 全参数微调示例](#pandallmops全参数微调示例)
  - [😎 PandaLLMOps Lora示例](#pandallmops-lora示例)
  - [⏩ PandaLLMOps 流水线并行示例](#pandallmops流水线并行示例)
  - [🫡 PandaLLMOps 部署示例](#pandallmops部署示例)
  - [🚀 PandaLLMOps 中英双语Tutorial上线](#pandallmops-tutorial)
- [🐼 PandaCommunity 大模型中文社区](#-pandacommunity大模型中文社区)
  - [🔥 社区介绍](#社区介绍)
  - [🌍 为什么选择PandaCommunity中文社区？](#为什么选择pandacommunity大模型中文社区)
  - [🎉 社区活动](#社区活动)
  - [🍻 加入我们！](#加入我们)
- [📢 社区公告](#-社区公告)
  - [🔥Panda项目最新进展](#panda项目最新进展)
- [📖 学习资料](#-学习资料)
  - [💡 原创学习资料](#原创学习资料)
  - [📚 LLM 相关论文](#llm相关论文)

- [🎉 致谢](#-致谢)
- [🤔 问题反馈](#-问题反馈)

## 🐼 PandaLLM

### 最新PandaLLM-LLaMA2-13B上线
我们最新基于LLaMA2-13B的底座模型在中文数据进行了增量预训练。这项研究是为了进一步提升我们的自然语言处理技术，以更好地适应不断变化的语言环境和应用需求。在这次的增量预训练中，我们选择了大量丰富多样的中文数据，包括文本、对话、新闻文章和社交媒体内容，以增强模型对中文语境的理解和表达能力。该底座模型的LLaMA2-13B架构在之前的研究中已经表现出良好的性能和可扩展性，因此我们选择该模型作为基础，通过增量预训练的方式来进一步优化其效果。

通过在大规模的中文数据上进行增量预训练，我们的底座模型现在具备更深入、更全面的中文语言知识，可以更好地理解中文的语法结构、语义含义以及上下文关联。这为我们在各种中文自然语言处理任务中取得更优秀的结果打下了坚实基础。未来，我们将继续致力于推进中文自然语言处理领域的研究，进一步改进和优化底座模型，并探索更多创新的方法来处理中文语言的特点和复杂性。我们相信，随着技术的不断进步，我们的中文自然语言处理技术将在诸多领域发挥重要作用，为用户提供更智能、更便捷的语言交互体验。


### PandaLLM已发布的各版本模型权重


可商用（基于LLaMA2和OpenLLaMA底座微调)

|  模型名称      | 模型大小 | 下载链接                                            |
| --------------- | ---------- | -------------------------------------------------------- |
| Panda-LLaMA2-13B | 13B      | [https://huggingface.co/qcw/llama2-panda-zh-13b](https://huggingface.co/qcw/llama2-panda-zh-13b) |
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


### PandaLLMOps训练框架
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

### PandaLLMOps工具介绍
PandaLLMOps是一款开源的大模型训练、推理和部署工具。该工具集成了大模型从训练到推理再到部署的全流程支持。我们致力于为广大大模型开发人员和爱好者提供一套简便易用的工具，以降低学习门槛，提高在大模型开发、推理和部署过程中的效率。
值得说明的是，我们的出发点并不是为了将已有的开源库重新封装一遍，我们认为这样的形式并不利于普通开发人员/爱好者针对自己的需求快速进行魔改，或者对于企业开发人员在开发过程中引入过多不稳定的包。
因此本项目中所有的代码都是尽可能以简洁的形式使用原生Pytorch，DeepSpeed，以及Huggingface Transformers使用原型的方式编写，并尽可能的提供说明，方便开发者直接将相关功能移植到自己的项目中去，而无需再次引入一个额外的厚重的第三方包。

目前，PandaLLMOps支持多种场景，包括：

1. **从零开始做预训练**：您可以使用PandaLLMOps来进行大规模预训练，从而让模型掌握更丰富的语言知识和表达能力。

2. **基于现有底座做增量预训练、全参数量微调、Lora、QLora**：PandaLLMOps提供了灵活的增量预训练、微调和Lora等功能，帮助您在已有底座模型的基础上进行更多样化和高效的模型优化。

3. **快速部署**：PandaLLMOps集成了vllm、lightllm等推理引擎，可以最大化优化推理速度，助力您在实际应用中快速部署和运行大模型。

我们希望PandaLLMOps能够为您提供强大而便捷的工具，使您能够更加专注于大模型的开发和创新，为自然语言处理和相关领域带来更加出色的成果。欢迎您加入我们的开源社区，共同推进大模型技术的发展，谢谢！

### PandaLLMOps预训练示例

首先找到本项目下的配置文件：
```bash
conf/llama/zh/llama2_13b_zh_v3_0_ds.yaml
```

该配置文件用于LLaMA2-13B在中文上的迁移预训练。接下来在包含8张80G-A100的节点上运行一下命令：

```bash
PAD_TOKEN="<unk>" deepspeed --include localhost:0,1,2,3,4,5,6,7 trainer_base_ds_mul.py -cp conf/llama/zh/ -cn llama2_13b_zh_v3_0_ds
```

如果你在AWS上使用PandaLLMOps，或者希望实时将模型断点文件保存在AWS S3，可以使用`trainer_base_ds_mul_aws.py`替换`trainer_base_ds_mul.py`，在前者中我们调用了`s5cmd`实现数据的云上同步。

#### FAQ

##### `PAD_TOKEN="<unk>"`的环境变量的用途是什么？  

由于LLaMA/LLaMA2没有显式指定pad token，为了方便训练时快速指定，我们做了一部后处理，具体可参考`general_util.tokenization_utils.expand_special_tokenizer`方法。

##### 如何在多节点上计算？    

多节点计算由于不同系统的不同，无法使用统一的脚本调用，因此需要参考DeepSpeed的多节点训练配置以及对应训练系统的文档说明。

##### 我没有8卡A100-80G / 我的显卡配置低于此，如果修改DeepSpeed配置开启不同功能降低显存需求？

在本节使用的配置文件里，你可以通过简单修改配置文件来快速开启某些特性：

- Gradient checkpointing: 设置`model.gradient_checkpointing=True` （默认开启）。
- DeepSpeed ZeRO-stage: 设置`ds_config.zero_optimization.stage=1/2/3` （默认为1）。
- DeepSpeed ZeRO Optimizer Offload: `ds_config.zero_optimization`区域中添加如下子配置 （默认开启）：
  ```yaml
  zero_optimization:
    stage: 1
    ...
    offload_optimizer:
      device: cpu
      pin_memory: True
  ```
- DeepSpeed ZeRO-3 Param Offload: 
  ```yaml
    zero_optimization:
      stage: 3
      ...
      offload_param:
        device: cpu
        pin_memory: True
  ```
- 其他DeepSpeed配置：直接在`ds_config`区域添加对应配置即可。
- 如何开启FlashAttention: 设置`model.enable_flash_attention=True`。有一些不同的选项（如开启原生FlashAttention或直接调用pytorch 2.0相关API，请参考`models.llama.LlamaPreTrainedModelPeftMixin.from_pretrained`方法。目前我们仅支持LLaMA，后续会补充MPT相关实现。

##### 关于DeepSpeed各种ZeRO优化的推荐尝试顺序？

CPU Offload以及Gradient Checkpointing与ZeRO的阶段无关，因此我们建议ZeRO首先尝试Stage-1，然后尝试开启CPU Offload和Gradient Checkpointing，最后尝试Stage-2以及开启上述选项的方案。

我们并不推荐使用ZeRO-3，ZeRO-3本质上已经在做模型并行，且引入了大量的额外通信开销，会导致训练时间显著延长。如果你的数据集非常小可以忍受变长数倍的训练时间，而并不想修改现有的代码，可以使用ZeRO-3以及Param Offload。
对于其他情况，我们建议使用LoRA/QLoRA（非全参数微调），或者Pipeline Parallelism（全参数微调）。具体可以参考后续的示例。

此外，本项目保留了使用 Tensor Parallel 的可能（由`tensor-parallel` pypi package支持），但考虑到能使用 Tensor Parallel的场景也能够使用 Pipeline Parallelism，因此我们目前没有提供相关的示例和实现。

##### 如何使用自己的数据集？

配置文件中通过`read_tensor_train`调用了我们使用的`torch.nn.utils.Dataset`类，您可以使用任意自己定义的Dataset类来取代这一配置。有关`Hydra`动态调用的相关细节，可以参考我们的Panda Tutorial。

### PandaLLMOps全参数微调示例

几乎所有步骤与预训练阶段没有区别，因此此处仅给出一个微调的配置文件以供参考：

```bash
conf/llama/zh/llama2_13b_zh_sft_combine_v1_0_ds.yaml
```

### PandaLLMOps-Lora示例

最简单的调用方式：在模型配置文件中添加如下配置

```yaml
model:
  _target_: models.llama.LlamaForConditionalGeneration.from_pretrained
  use_peft: True
```

进阶版（指定LoRA Config或者开启QLoRA）：

```yaml
model:
  _target_: models.llama.LlamaForConditionalGeneration.from_pretrained
  use_peft: True
  lora_config:
    _recursive_: False
    _target_: models.llama.LoraConfig
    task_type: CAUSAL_LM
    inference_mode: False
    target_modules:
      _target_: models.llama.find_all_linear_names
      bits: 4
    r: 64
    lora_alpha: 16
    lora_dropout: 0.05
  quantization_config:
    _target_: transformers.utils.quantization_config.BitsAndBytesConfig
    load_in_4bit: True
    bnb_4bit_compute_dtype:
      _target_: general_util.training_utils.return_torch_dtype
      dtype: bfloat16
    bnb_4bit_use_double_quant: True
    bnb_4bit_quant_type: "nf4"
  device_map:
    _target_: models.llama.return_single_device_map
  load_in_4bit: True
  max_memory: True
```

可用的参考Config文件: `conf/llama/wiki/llama_65b_qlora_train_new.yaml`

我们通过集成了原有的`LlamaPreTrainedModel`类来添加各种选项，本质上这些选项与具体的模型结构无关，所以你可以参考相关的代码将其快速移植到其他`transformers`模型上。该重载类可以参考`models.llama.LlamaPreTrainedModelPeftMixin`。


### PandaLLMOps流水线并行示例

我们提供了一个MPT-30B的配置文件：

```bash
conf/mpt/mpt_30b_mp_v1_0.yaml
```

可通过如下命令启动在8卡A100-80G上的训练：

```bash
PAD_TOKEN="<unk>" deepspeed --include localhost:0,1,2,3,4,5,6,7 trainer_base_ds_mp.py -cp conf/mpt/ -cn mpt_30b_mp_v1_0
```

你可以选择通过设置配置文件中的`num_stages=8`（默认设置）来开启纯流水线并行，或者`num_stages=4`来开启4路流水线并行和2路数据并行的混合并行模式。

一些需要注意的Points:

1. 流水线并行需要对原有模型结构进行重封装，因此对于没有包含在本项目中的模型（LLaMA & MPT），需要自行实现，但我们提供了Tutorial帮助你理解我们的代码以及如何自行封装。请参考下面的FAQ。
2. 由于1，训练不推荐使用原有的Huggingface权重（尽管我们支持了这个方法），因为这样会导致巨大的内存峰值使用（如你无法在内存1T的节点上完成65B以上模型的初始化操作）。我们在Panda Tutorial中介绍了这一问题出现的原因。同时，我们有提供了LLaMA和MPT转换权重的脚本：
    ```bash
    convert2ckpt.py  // LLaMA
    mpt_convert2ckpt.py  // MPT
    ```
3. 同样，使用流水线并行训练后的模型权重需要转换为Huggingface权重，可以使用如下脚本：
    ```bash
    convert2hf.py // LLaMA
    ```



#### FAQ

##### 是否支持LLaMA？

支持。不过目前我们还没有在本项目上训练，相关配置文件可以参考这个[原型库](https://github.com/SparkJiao/llama-pipeline-parallel)并进行修改。或者我们的Panda Tutorial中的Advance Usage。

##### 如何对其他模型结构（如ChatGLM）使用流水线并行？

我们在Panda Tutorial的Advance Usage中提供了一个详细的Tutorial，包含对我们代码中细节的解读，以及如何从零开始封装一个流水线并行的模型。

### PandaLLMOps推理测试示例

此示例主要针对低显存场景下的批量推理（如数据集跑分、标注数据集等），如果需要在线推理，请参考下方的PandaLLMOps部署示例。目前支持通过修改配置文件或者切换入口脚本来实现以下几种推理方式：

#### 量化 (Recommended in Single GPU)

直接在配置文件中设置`model.load_in_8bit=True`或`model.load_in_4bit=True`即可。需注意需要同时设置`device_map`。

#### Naive Model Parallel

直接在配置文件中设置`device_map="auto"`。需要注意此时不能使用分布式的启动方式（如`torchrun`和`deepspeed.launch`），可以使用如下命令：

```bash
CUDA_VISIBLE_DEVICES=XXX python trainer_base_fsdp_v4.py -cp <config path> -cn <config name>
```

#### Tensor Parallel (Recommended in Multiple GPU)

不在调用`model.from_pretrained`方法，而是调用`model.from_pretrained_eval_tp`方法。同样需要注意不能使用分布式的启动方式。

#### Tensor Parallel DeepSpeed (DeepSpeed Inference)

```bash
deepspeed --include localhost:xxx ds_inference.py -cp <config path> -cn <config name>
```

注：以上几种方法本质上是对模型计算的从新封装，并不影响其本身的用法（区别于VLLM），及你依然可以用原有的模型计算 language modeling loss 或者直接调用`generate`方法。

### PandaLLMOps部署示例

如果您有直接下载或者自己训练的模型参数，可以直接对模型进行部署。

```python run_chat.py --model_path ./pretrained_model/panda-13B --query "write a peom"```

如果您需要用模型参数合并（例如LLaMA 1)，则可以按照我们以下 Tutorial 中《快速部署》所展示的方式去进行参数合并和部署。

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
- 有关LLM相关的论文请参考以下Github Repo的总结：
[https://github.com/Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)

- 有关多模态相关的论文请参考以下Github Repo的总结：
[https://github.com/pliang279/awesome-multimodal-ml](https://github.com/pliang279/awesome-multimodal-ml)

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
