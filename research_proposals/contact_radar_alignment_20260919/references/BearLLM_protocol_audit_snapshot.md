> 本文件是已有工作区审计文档的只读快照，归档于 2026-09-19。原位置为 `BearLLM/outputs/protocol_audit.md`。其中训练计划、运行状态与用户约定属于既有复现工作，不是本研究方案此次执行或完成的训练。

# BearLLM released-code reproduction audit

Audit date: 2026-09-19. Repository: `4bdd29bf013971909e81cd79732c4b3813101ca9`.
Dataset revision: `78cd9b8b6b65cd43eebf4879b060d783d5f7dcfe`.
This audit records the original released protocol; it is not a claim that training has completed.

## Agreed reproduction target

The user selected **the current released repository**, Qwen2.5-1.5B-Instruct and its 50-epoch fine-tuning configuration. We therefore preserve the released split/reference policy, seed its random choices, execute the two training stages, and disclose differences from the publication. The official-checkpoint all-data evaluation and the newly trained FCN test evaluation must be presented separately.

## Paper reference (compact specification)

The publication specifies nine datasets split individually 7:2:1; Appendix C.2 restricts reference lookup to training data and evaluates the best validation-accuracy checkpoint. FCN: AdamW, batch 1024, learning rate 1e-4, ReduceLROnPlateau(patience=150 batches, factor=0.5), at most 50 epochs, stop below 1e-7. Algorithm 1 specifies 20 fine-tuning epochs; its backbone is Qwen2-1.5B. Tables 2/6 report diagnosis accuracy/false-alarm/missed-alarm; MBHM values are 99.02%/0.7%/0.3%. Table 3 varies frequency dimensions; Table 4 ablates DCN/reference/residual and tests excluded datasets. Table 5 is a human preference study, not automatic language accuracy. [Paper, including appendix](https://arxiv.org/html/2408.11281v2); [published AAAI entry](https://ojs.aaai.org/index.php/AAAI/article/view/34188).

## Exact released data inventory and split

These counts come from the downloaded metadata and corpus, rather than assuming the paper's dataset description matches the current release:

The released source names are CWRU, DIRG, HIT, IMS, JUST, MFPT, **NCEPU**, PU and XJTU. The paper names JNU where the release uses NCEPU (3,600 rows). We retain the actual metadata name; without original construction provenance these should not be silently equated.

| Item | Count |
|---|---:|
| Metadata vibration rows | 135,516 |
| Distinct condition IDs | 1,043 |
| Conditions with any healthy reference | 258 |
| Query rows accepted by released `create_cache_dataset` | 122,792 |
| Query rows omitted because their condition has no healthy reference | 12,724 |
| Training queries / query-reference pairs | 85,955 / 257,865 |
| Validation queries / query-reference pairs | 24,558 / 73,674 |
| Test queries / query-reference pairs | 12,279 / 36,837 |
| Released corpus rows | 600 |
| Corpus rows per task / per label | 150 / 60 |
| Unique corpus vibration IDs / reference IDs | 551 / 593 |

`file_info` is iterated in its released SQL order. Among rows with a healthy reference available, eligible row index modulo 10 is assigned to train for residues 0–6, validation for 7–8, and test for 9. Each accepted query produces three pairs, sampling healthy rows of the same condition **with replacement**, before moving to the next query. The reproduction adds `random.Random(42)` and writes the complete split manifest. References may coincide with the query itself for healthy queries; that is source behavior.

**Coverage materially limits the meaning of the scores:** after the released condition/reference filter, CWRU, DIRG and NCEPU contain only healthy queries. Their test subsets contain respectively 7, 102 and 90 unique healthy queries and no fault examples. Even perfect accuracy on those subsets does not validate fault recognition for those sources. MFPT has four test queries (two healthy, two moderate outer-ring faults). Report label support alongside per-dataset scores and retain unsupported fault rows in the coverage report.

The released HDF5 already contains 24,000-component DCN signals. Applying DCN a second time would change the input domain. The Hugging Face preprocessing helper subtracts mean/divides by standard deviation before DCT; repository `functions/dcn.py` does not. This matters when comparing raw-signal demo input with HDF5 evaluation. [Dataset card](https://huggingface.co/datasets/SIA-IDE/MBHM/blob/main/README.md), [published files](https://huggingface.co/datasets/SIA-IDE/MBHM/tree/main).

All 600 corpus query IDs belong to the current released-code training query split. Thus current-corpus fine-tuning does not add held-out validation/test query IDs to training. Its healthy reference IDs still require separate overlap reporting; corpus generation scores after tuning are in-sample regardless of query-split membership.

## Code findings and reproduction decisions

| Released code | Consequence / reproduction handling |
|---|---|
| `functions/mbhm.py:18–57` pools all healthy references before the modulo split | Cross-split reference reuse is retained and counted. Source-compatible results are not described as reference-isolated generalization. |
| `functions/mbhm.py:43–57` silently drops conditions without healthy references | Coverage denominator must include all 135,516 metadata rows, with 12,724 unsupported rows explicitly reported. Do not fabricate references from other conditions. |
| `src/pre_training.py:82–91` overwrites one checkpoint for either lower validation loss or higher accuracy, then tests last in-memory model | Preserve this OR-selection policy and batch-mean validation loss for the main FCN export and fine-tuning initialization. Separately report the last and highest-validation-accuracy checkpoints as diagnostics. |
| `src/pre_training.py:48` steps ReduceLROnPlateau on every training batch loss | Default training preserves this behavior. Optional validation-loss scheduling is a distinct experiment. |
| `src/fine_tuning.py` defaults to 50 epochs, batch 1, accumulation 4, LoRA rank 4, alpha 32, dropout 0.1, LR 1e-4, cosine | Follow current code as the user selected. All linear layers include alignment linears and LLM linears. |
| `CorpusDataset` exposes every corpus row; `fine_tuning` has no validation split | Tune all 600 released rows. All-corpus text evaluation after training is in-sample and must be labeled accordingly. |
| Original `ModifiedEmbedding` switches to a single `cache.npy` when `self.training` is false | Training helper resolves sample IDs explicitly in either mode, allowing correct batched evaluation without cache-file races. |
| `get_peft_model` freezes encoder parameters but `model.train()` activates BatchNorm | Default reproduction preserves training-mode running-buffer updates. Save those buffers with each vibration adapter checkpoint. |
| `description_text` misses one comma | Unused for initialization: the explicit ten `description_tokens` rows are used. Do not change token initialization based on this cosmetic list. |
| Source requirements install mutable Transformers/PEFT Git heads | Use and export a pinned `m2vllm` environment so the actual run remains reproducible. |

The new training entry point is `scripts/train_bearllm.py`. It adds seeded manifests, JSONL epoch logs, resumable optimizer/scheduler/RNG state, RAM preload, separate run outputs, and explicit source-selected/last/best-accuracy metrics. All three FCN loaders shuffle, and persistent workers remain disabled, matching upstream. It preserves the original FCN architecture and pretrained-FCN-to-alignment initialization. Fine-tuning uses the installed Transformers 4.49 `Trainer` with the original optimizer, scheduler and accumulation behavior; the actual model has `model_accepts_loss_kwargs=True`, so a hand-written average of per-example losses would not match its token normalization. Saving separates unwrapped base vibration weights from PEFT deltas, including updated BatchNorm buffers, so existing inference loaders can consume the result. Extra epoch checkpoints retain two resumable Trainer checkpoints without changing optimizer updates. Changing microbatch size, enabling gradient checkpointing, or disabling source early stopping must be visible in each run's config.

An initial diagnostic training run used fixed evaluation order and persistent workers. It was archived under `diagnostics/pretrain_v1_fixed_evaluation`; the authoritative main run was restarted from epoch zero after restoring all source loader behavior. No optimizer updates from that diagnostic run are used to initialize the main run.

## What is and is not publicly reproducible

The current model release contains a vibration adapter and LoRA weights, without separate FCN checkpoint files, split manifests, training logs, baseline checkpoints, or human-study votes. Its first two alignment linears and feature encoder can be inspected as a released classification representation, but its LoRA-modified classifier is not automatically identical to a standalone pretraining checkpoint. [Published weights](https://huggingface.co/SIA-IDE/BearLLM/tree/main).

The repository contains network definitions for WDCNN, TCNN, QCNN, MagNet, and BearingFM; it does not contain complete end-to-end experiments for every baseline/ablation. In particular, `BearingFM.py` expects an absent preprocessed NumPy directory/environment variable and contains a package import inconsistent with normal repository execution. `MagNet.py` defines a plain `nn.Module.backward` method that does not establish an autograd gradient-reversal operation. Re-running those definitions as ordinary classifiers would not establish faithful reproduction of their methods.

Because only the fixed-length DCN representation is released in the primary HDF5, original raw time-domain baseline inputs and alternate 48,000-component DCN information cannot be recovered exactly from it. Reproducing all baseline/ablation tables would require original-source signal reconstruction, faithful augmentation/training implementations, and additional experiment definitions. Human preference results require a new human study. These are publication-wide replication requirements, beyond the selected current-code two-stage reproduction.

## Completion criteria for this run

1. Pin and record environment, code commit, dataset/checkpoint revisions and hashes.
2. Audit every published HDF5 row, metadata record and corpus record; record excluded unsupported conditions.
3. Evaluate official weights on every eligible query/reference pair and every released corpus example; provide all nine source datasets, sample counts, confusion matrices, accuracy, macro F1, false-alarm and missed-alarm rates. Distinguish classifier scores from generated-text parsing scores.
4. Train FCN under the selected source protocol for its configured 50-epoch maximum; record source early stopping if it happens. Report source-selected, final, and best-validation-accuracy checkpoints on the entire held-out query test split, with reference reuse disclosed.
5. Initialize alignment from that run's source-selected FCN and Qwen embeddings; fine-tune the complete 600-row released corpus for all 50 configured epochs. Save loadable model artifacts and resumable state.
6. Evaluate the newly trained end-to-end model with the same evaluator and explicitly label corpus results as in-sample. Record metric differences and scope limits instead of equating code execution with exact paper-number reproduction.
