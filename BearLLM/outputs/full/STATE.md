# Completed BearLLM reproduction

Status: complete. See completion_verified.json (all final gates passed), RESULTS_zh.md and ../../FULL_REPRODUCTION_zh.md.

- Independent Conda m2vllm, PyTorch 2.11.0+cu128; package isolation, pip checks and real GPU training/inference passed.
- All 135,516 MBHM signals / 9 sources audited; both official and trained evaluations independently verified at prediction-row level.
- Official/new-model scopes each: 368,376 same-condition pairs, 38,172 supplemental pairs, 600 corpus generations, 12,279 test-query generations.
- Formal FCN: source early stop at epoch19, source-selected epoch11; full test-pair accuracy98.6725303%.
- Native Trainer LoRA: all50epochs/7500updates/30000exampleforwards completed; 22 export checks passed.
- Final-BN trained generation test accuracy93.777995%, official97.703396%; no claim of matching paper or official performance.
- Original repository save-behavior diagnostic (initial BN + identical final LoRA):95.756983%, separate source_export_weights/source_export_heldout; 10 checks passed, main result never replaced.
- Official and trained full Demo exit0, outputs Fault-Free (unlabeled example).
- Main trained weights: /media/nas_users/huangyating/bearllm-runs/released-code-seed42/finetune/weights.
- All evaluation processes exited0, GPUs released. No remaining work for this request.

Limits: healthy references cross splits; public corpus has600 rows and is training replay; missing same-condition references for12724 signals require explicitly separate fallback experiment; original official training membership is unknown; full paper corpus/raw preprocessing/baselines/ablations/human study are not reproduced.

Earlier non-source evaluation-loader diagnostic training is archived under runroot/diagnostics/pretrain_v1_fixed_evaluation and is excluded from final results.
