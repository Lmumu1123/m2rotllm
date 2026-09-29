---
license: mit
task_categories:
- token-classification
- question-answering
language:
- en
---
<div align="center">
<a href="https://github.com/SIA-IDE/BearLLM/">
<img src="https://raw.githubusercontent.com/SIA-IDE/BearLLM/refs/heads/main/docs/images/logo.svg" width="200" alt="logo"/>
</a>
<h1>Multimodal Bearing Health Management Dataset</h1>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/Pytorch-latest-orange"></a>
<a href="https://arxiv.org/abs/2408.11281"><img alt="arXiv" src="https://img.shields.io/badge/Paper-arXiv-B31B1B"></a>
<a href="https://huggingface.co/datasets/SIA-IDE/MBHM"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-🤗-FFFDF5"></a>
<a href="https://github.com/SIA-IDE/BearLLM"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SIA-IDE/BearLLM"></a>
</div>

## ⚡️ Download

Due to the capacity limitation of GitHub, please download the data file on [huggingface](https://huggingface.co/datasets/SIA-IDE/MBHM).

## 📚 Introduction
The [MBHM](https://huggingface.co/datasets/SIA-IDE/MBHM) dataset is the first multimodal dataset designed for the study of bearing health management. It is divided into two parts: vibration signals and health management corpus. The vibration signals and condition information are derived from 9 publicly available datasets. The thousands of working conditions pose more difficult challenges for the identification model and better represent real-world usage scenarios.

In the dataset, vibration signals from different datasets have been converted to the same length (24000) by Discrete Cosine Normalization (DCN). For more information about the implementation of DCN, please refer to the [paper](https://arxiv.org/abs/2408.11281) or [code](https://huggingface.co/datasets/SIA-IDE/MBHM/blob/main/src/dcn.py).

## 💻 Demo

We provide a demo script to show how to load the MRCHM dataset and output the data shape. Please check the [demo](https://huggingface.co/datasets/SIA-IDE/MBHM/blob/main/src/demo.py) for more details.

## 📖 Citation
Please cite the following paper if you use this dataset in your research:

```
@article{pengBearLLMPriorKnowledgeEnhanced2025,
  title = {{{BearLLM}}: {{A Prior Knowledge-Enhanced Bearing Health Management Framework}} with {{Unified Vibration Signal Representation}}},
  author = {Peng, Haotian and Liu, Jiawei and Du, Jinsong and Gao, Jie and Wang, Wei},
  year = {2025},
  month = apr,
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {39},
  number = {19},
  pages = {19866--19874},
  issn = {2374-3468},
  doi = {10.1609/aaai.v39i19.34188},
  urldate = {2025-04-11},
}
```