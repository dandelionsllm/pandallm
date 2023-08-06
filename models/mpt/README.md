---
license: apache-2.0
tags:
- Composer
- MosaicML
- llm-foundry
- StreamingDatasets
datasets:
- allenai/c4
- mc4
- togethercomputer/RedPajama-Data-1T
- bigcode/the-stack-dedup
- allenai/s2orc
inference: false
---

# MPT-30B

MPT-30B is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code.
This model was trained by [MosaicML](https://www.mosaicml.com).

MPT-30B is part of the family of Mosaic Pretrained Transformer (MPT) models, which use a modified transformer architecture optimized for efficient training and inference.

MPT-30B comes with special features that differentiate it from other LLMs, including an 8k token context window (which can be further extended via finetuning; see [MPT-7B-StoryWriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)), support for context-length extrapolation via [ALiBi](https://arxiv.org/abs/2108.12409), and efficient inference + training via FlashAttention. It also has strong coding abilities thanks to its pretraining mix. MPT models can also be served efficiently with both standard HuggingFace pipelines and NVIDIA's [FasterTransformer](https://github.com/NVIDIA/FasterTransformer).
The size of MPT-30B was also specifically chosen to make it easy to deploy on a single GPU—either 1xA100-80GB in 16-bit precision or 1xA100-40GB in 8-bit precision.

This model uses the MosaicML LLM codebase, which can be found in the [llm-foundry repository](https://github.com/mosaicml/llm-foundry). It was trained by MosaicML’s NLP team on the [MosaicML platform](https://www.mosaicml.com/training) for LLM pretraining, finetuning, and inference.


### How is this model different?

MPT-30B is:
* **Licensed for the possibility of commercial use** (unlike [LLaMA](https://arxiv.org/abs/2302.13971)).
* **Trained on a large amount of data** (1T tokens like [LLaMA](https://arxiv.org/abs/2302.13971) vs. 300B for [Pythia](https://github.com/EleutherAI/pythia), 300B for [OpenLLaMA](https://github.com/openlm-research/open_llama), and 800B for [StableLM](https://github.com/Stability-AI/StableLM)).
* **Prepared to handle extremely long inputs** thanks to [ALiBi](https://arxiv.org/abs/2108.12409).
* **Capable of fast training and inference** (via [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf) and [FasterTransformer](https://github.com/NVIDIA/FasterTransformer))
* **Equipped with highly efficient open-source training code** via the [llm-foundry repository](https://github.com/mosaicml/llm-foundry)

### Models finetuned off MPT-30B:

The following models are finetuned on MPT-30B:

* [MPT-30B-Instruct](https://huggingface.co/mosaicml/mpt-30b-instruct): a model for short-form instruction following.
Built by finetuning MPT-30B on several carefully curated datasets.
  * License: _CC-BY-SA-3.0_

* [MPT-30B-Chat](https://huggingface.co/mosaicml/mpt-30b-chat): a chatbot-like model for dialogue generation.
Built by finetuning MPT-30B on [ShareGPT-Vicuna](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered), [Camel-AI](https://huggingface.co/camel-ai),
 [GPTeacher](https://github.com/teknium1/GPTeacher), [Guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco), [Baize](https://github.com/project-baize/baize-chatbot) and some generated datasets.
  * License: _CC-By-NC-SA-4.0_
  * [Demo on Hugging Face Spaces](https://huggingface.co/spaces/mosaicml/mpt-30b-chat)

## Model Date

June 22, 2023

## Model License

Apache-2.0

## Documentation

* [Blog post: MPT-30B: Raising the bar for open-source foundation models](https://www.mosaicml.com/blog/mpt-30b)
* [Codebase (mosaicml/llm-foundry repo)](https://github.com/mosaicml/llm-foundry/)
* Questions: Feel free to contact us via the [MosaicML Community Slack](https://mosaicml.me/slack)!


## How to Use

This model is best used with the MosaicML [llm-foundry repository](https://github.com/mosaicml/llm-foundry) for training and finetuning.

```python
import transformers
model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-30b',
  trust_remote_code=True
)
```
Note: This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method.
This is because we use a custom `MPT` model architecture that is not yet part of the Hugging Face `transformers` package.
`MPT` includes options for many training efficiency features such as [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf), [ALiBi](https://arxiv.org/abs/2108.12409), [QK LayerNorm](https://arxiv.org/abs/2010.04245), and more.

To use the optimized [triton implementation](https://github.com/openai/triton) of FlashAttention, you can load the model on GPU (`cuda:0`) with `attn_impl='triton'` and with `bfloat16` precision:
```python
import torch
import transformers

name = 'mosaicml/mpt-30b'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)
```

The model was trained initially with a sequence length of 2048 with an additional pretraining stage for sequence length adapation up to 8192. However, ALiBi enables users to increase the maximum sequence length even further during finetuning and/or inference. For example:

```python
import transformers

name = 'mosaicml/mpt-30b'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.max_seq_len = 16384 # (input + output) tokens can now be up to 16384

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  trust_remote_code=True
)
```

This model was trained with the MPT-30B tokenizer which is identical to the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-30b')
```

The model can then be used, for example, within a text-generation pipeline.  
Note: when running Torch modules in lower precision, it is best practice to use the [torch.autocast context manager](https://pytorch.org/docs/stable/amp.html).

```python
from transformers import pipeline

with torch.autocast('cuda', dtype=torch.bfloat16):
    inputs = tokenizer('Here is a recipe for vegan banana bread:\n', return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# or using the HF pipeline
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')
with torch.autocast('cuda', dtype=torch.bfloat16):
    print(
        pipe('Here is a recipe for vegan banana bread:\n',
            max_new_tokens=100,
            do_sample=True,
            use_cache=True))
```

## Model Description

The architecture is a modification of a standard decoder-only transformer.

The model has been modified from a standard transformer in the following ways:
* It uses [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf)
* It uses [ALiBi (Attention with Linear Biases)](https://arxiv.org/abs/2108.12409) and does not use positional embeddings
* It does not use biases


| Hyperparameter | Value |
|----------------|-------|
|n_parameters | 29.95B |
|n_layers | 48 |
| n_heads | 64 |
| d_model | 7168 |
| vocab size | 50432 |
| sequence length | 8192 |



## Training Data

### Streaming Datasets

Data was formatted using the MosaicML [StreamingDataset](https://github.com/mosaicml/streaming) library to host our data in object storage and efficiently stream it to our compute cluster during training.
StreamingDataset obviates the need to download the whole dataset before starting training, and allows instant resumption of training from any point in the dataset.


### Data Mix

The model was trained for 1T tokens on the following data mix:

| Data Source | Number of Tokens in Source | Proportion | Effective Number of Tokens | Epochs |
|-------------|----------------------------|------------|----------------------------|--------|
| mC4 3.1.0 - English (200+ words) | 2417.99 B | 33.50% | 335 B | 0.14 |
| c4 - English - SemDedup 80% | 100.42 B | 29.90% | 299 B | 2.98 |
| RedPajama - CommonCrawl | 878.45 B | 8.50% | 85 B | 0.097 |
| The Stack - Selected Languages | 463.78 B | 10.00% | 100 B | 0.22 |
| RedPajama - Wikipedia | 4.87 B | 4.00% | 40 B | 8.21 |
| The Stack - Markdown | 107.07 B | 4.50% | 45 B | 0.42 |
| Semantic Scholar ORC | 48.95 B | 3.30% | 33 B | 0.67 |
| RedPajama - Books | 26.02 B | 3.00% | 30 B | 1.15 |
| RedPajama - arXiv | 28.10 B | 1.90% | 19 B | 0.68 |
| RedPajama - StackExchange | 20.54 B | 1.40% | 14 B |0.68 |

Samples for each batch were selected from one of the datasets with the probability specified above. The examples were shuffled within each dataset, and each example was constructed from as many sequences from that dataset as were necessary to fill the sequence length. To build 8k support into MPT-30B efficiently, we first pre-trained on 1T tokens using sequences that were 2k tokens long, and then trained for an additional 50B tokens using sequences that were 8k tokens long.

The data was tokenized using the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer. This BPE tokenizer has a number of desirable characteristics,
most of which are relevant for tokenizing code:
(1) It was trained on a diverse mix of data that includes code (The Pile)
(2) It applies consistent space delimitation, unlike the GPT2 tokenizer which tokenizes inconsistently depending on the presence of prefix spaces
(3) It contains tokens for repeated space characters, which allows superior compression of text with large amounts of repeated space characters.

The model vocabulary size of 50432 was set to be a multiple of 128 (as in [MEGATRON-LM](https://arxiv.org/abs/1909.08053)).

### Training Configuration

The model was trained in three stages using the [MosaicML Platform](https://www.mosaicml.com/platform):
(i) First it was trained on 440 A100-40GBs with a batch size of 1760.
(ii) Then, on 216 A100-40GBs with a batch size of 1728.
(iii) Training was completed on 256 H100-80GBs with a batch size of 512 with 8k context length and 50B tokens.
The model was trained with sharded data parallelism using [FSDP](https://pytorch.org/docs/stable/fsdp.html) and used the [LION](https://arxiv.org/abs/2302.06675) optimizer.

## Limitations and Biases

_The following language is modified from [EleutherAI's GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)_

MPT-30B (Base) is **not** intended for deployment without finetuning.
It should not be used for human-facing interactions without further guardrails and user consent.

MPT-30B can produce factually incorrect output, and should not be relied on to produce factually accurate information.
MPT-30B was trained on various public datasets.
While great efforts have been taken to clean the pretraining data, it is possible that this model could generate lewd, biased or otherwise offensive outputs.


## MosaicML Platform

If you're interested in [training](https://www.mosaicml.com/training) and [deploying](https://www.mosaicml.com/inference) your own MPT or LLMs on the MosaicML Platform, [sign up here](https://forms.mosaicml.com/demo?utm_source=huggingface&utm_medium=referral&utm_campaign=mpt-30b).

## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model. Please consult an attorney before using this model for commercial purposes.

## Citation

Please cite this model using the following format:

```
@online{MosaicML2023Introducing,
    author    = {MosaicML NLP Team},
    title     = {Introducing MPT-30B: Raising the bar
for open-source foundation models},
    year      = {2023},
    url       = {www.mosaicml.com/blog/mpt-30b},
    note      = {Accessed: 2023-06-22},
    urldate   = {2023-06-22}
}
```