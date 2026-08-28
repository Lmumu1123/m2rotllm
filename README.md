# m2rotllm / RotLLM

> **Repository**: [https://github.com/Lmumu1123/m2rotllm](https://github.com/Lmumu1123/m2rotllm)

This repository extends the original [RotLLM](https://github.com/SIA-IDE/RotLLM) framework with **cross-modal time alignment** between 77GHz mmWave radar IQ data and contact tactile sensor data, enabling multimodal fusion for rotating machinery health management.

## Quick Start: Cross-Modal Alignment

```bash
conda create -n rotllm python=3.12 -y && conda activate rotllm
pip install -r requirements.txt

python cross_modal_alignment/align_cross_modal.py \
  --radar_bin /path/to/radar.bin \
  --tactile_csv /path/to/data_0.csv \
  --out_dir ./cross_modal_alignment/_align_out

python cross_modal_alignment/tests/run_tests.py
```

## Documentation

| Document | Description |
|----------|-------------|
| [Conversation Summary](docs/CONVERSATION_SUMMARY.md) | Full project discussion and design decisions |
| [Reproduction Guide](docs/REPRODUCTION_GUIDE.md) | Step-by-step setup on a new server |
| [Algorithm](docs/CROSS_MODAL_ALIGNMENT_ALGORITHM.md) | Cross-modal alignment algorithm details |
| [Testing](docs/TESTING_GUIDE.md) | Test cases and validation procedures |

---

# RotLLM: A Unified Rotating Machinery Health Management Framework

This repository contains the core implementation for the paper: **"A Unified Rotating Machinery Health Management Framework Leveraging Large Language Models for Diverse Components, Conditions, and Tasks"**.

## ⚠️ Note

The content herein represents the core implementation of the methods described in our paper. Please note that the code has been extracted directly from our research drafts. While functional, it may require further debugging and refactoring for robust execution. We are actively working on adding comprehensive comments and improving the code's readability.

## 1. Abstract

This study introduces the Rotating Machinery Large Language Model (RotLLM), a unified framework for rotating machinery health management that integrates deep learning with large language models (LLMs) to address diverse operational conditions, components, and health management tasks. RotLLM employs a novel Spectral Folding Network (SFN) to transform the vibration spectrum into a unified feature space that preserves essential health state information. A dedicated projection layer then maps these features into the semantic domain of an LLM. The framework is trained using a three-stage strategy: first, pre-training the encoder on the Large-scale Multimodal Rotating Machinery (LMR) dataset; second, initializing the projection layer with textual health state labels; and finally, fine-tuning using parameter-efficient Low-Rank Adaptation (LoRA) with a high-quality corpus for various health management tasks. Experimental evaluations demonstrate that RotLLM achieves state-of-the-art performance in fault classification, maintains strong robustness under noisy conditions, and delivers rapid multi-task inference with minimal computational overhead.

## 2. Framework Overview

The core innovation of RotLLM is the deep semantic alignment between numerical vibration signals and the textual domain of an LLM. This is achieved through a three-stage training strategy:

1. **Encoder Pre-training:** A Spectral Folding Network (SFN) is pre-trained on a vast collection of vibration signals to learn a robust and generalized representation of machinery health states.
2. **Projection Layer Initialization:** A projection layer is initialized to map the SFN's feature output into the LLM's embedding space. This is seeded using text embeddings of known health state labels, bridging the modal gap.
3. **Instruction Fine-Tuning:** The entire model (SFN encoder, projection layer, and LLM) is jointly fine-tuned end-to-end using a high-quality, multi-modal corpus. This stage uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to efficiently adapt the LLM to generate precise and context-aware responses for various PHM tasks.

## 3. Repository Structure

```text
/
├── cross_modal_alignment/    # NEW: mmWave radar + tactile sensor time alignment
│   ├── align_cross_modal.py  # End-to-end alignment CLI
│   ├── dtw_alignment.py      # Constrained DTW + cross-correlation
│   └── tests/run_tests.py    # Unit & integration tests
├── docs/                     # Project documentation (Chinese)
├── code/
│   ├── dataset_constructor/  # Scripts to build datasets from public sources
│   ├── pre_train/            # Code for Stage 1: Encoder pre-training
│   ├── fine_tune/            # Code for Stage 2 & 3: Initialization and Instruction fine-tuning
│   └── models/               # Implementations of SFN and other baseline models
├── datasets/
│   ├── vibration_metadata.sqlite # Metadata for vibration signals
│   └── corpus_example.json       # A sample of the multi-modal instruction corpus
├── weights/
│   ├── encoder_weights.pth   # Pre-trained weights for the SFN encoder
│   └── ...                   # Other pre-trained weights (projection layer, etc.)
└── README.md                 # This file
```

- **`code/`**: Contains the core source code for the RotLLM framework.
  - **`dataset_constructor/`**: Scripts for processing public raw vibration data (from sources like CWRU, SEU, XJTU, etc.) into a unified format.
  - **`pre_train/`**: Contains the implementation for the first stage of training, focusing on the SFN encoder.
  - **`fine_tune/`**: Contains the implementation for the second and the third stages, performing instruction-based fine-tuning of the integrated RotLLM.
  - **`models/`**: Contains the PyTorch implementation of our proposed SFN model and other baseline encoder models used for comparison.
- **`datasets/`**: Contains sample data to illustrate data structure.
  - **`vibration_metadata.sqlite`**: An SQLite database containing the metadata (operating conditions, fault labels, etc.) for all 237,298 vibration samples in our LMR dataset.
  - **`corpus_example.json`**: An example snippet from the high-quality, multi-modal instruction corpus used for fine-tuning.
- **`weights/`**: Contains the pre-trained weights for key components of the RotLLM.

## 4. Dataset Information

For now, only the metadata for the vibration signals and a limited subset of the instruction corpus are included in this release.
To reconstruct the full dataset, you may use the scripts provided in the `code/dataset_constructor/` directory to process the raw data from the original public sources.

We are actively curating and processing the complete Large-scale Multimodal Rotating machinery (LMR) dataset, and will make it publicly available on Hugging Face as soon as possible to ensure reproducibility and support future research.

## 5. High-Level Workflow

The intended workflow to replicate our results is as follows:

1. **Dataset Construction**: Use the scripts in `code/dataset_constructor/` to download and process the raw data from the public sources listed in the paper. This will create the vibration signal dataset.
2. **Stage 1: Pre-training**: Run the main script in `code/pre_train/` to pre-train the SFN encoder on the newly constructed dataset. The resulting `encoder_weights.pth` will be saved.
3. **Stage 2 & 3: Fine-tuning**: Use the scripts in `code/fine_tune/` to perform the final instruction-based fine-tuning. This stage initializes the projection layer and fine-tunes it jointly with the LLM using the provided multi-modal corpus examples.

*Note: As mentioned, the code is in a draft state and may require adjustments to run seamlessly.*

## 6. Citation

If you find this work useful in your research, please consider citing our paper. 

```bibtex
@article{RotLLM,
  title = {A Unified Rotating Machinery Health Management Framework Leveraging Large Language Models for Diverse Components, Conditions, and Tasks},
  author = {Peng, Haotian and Gao, Jie and Liu, Jiawei and Du, Jinsong and Wang, Wei},
  year = {2025},
  month = dec,
  journal = {Engineering Applications of Artificial Intelligence},
  volume = {162},
  pages = {112544},
  issn = {09521976},
  doi = {10.1016/j.engappai.2025.112544},
  urldate = {2025-10-08}
}
```
