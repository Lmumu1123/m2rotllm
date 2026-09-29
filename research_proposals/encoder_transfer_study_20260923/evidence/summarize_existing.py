"""Read archived results; no training, checkpoint writes, or new evaluation split."""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parents[1]
SOURCE = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/retention_alignment_20260922/radar/stage_b_v0'))
m = pd.read_csv(SOURCE / 'metrics.csv')
p = pd.read_csv(SOURCE / 'file_predictions.csv')
m = m[m.role == 'test'].copy()
p = p[p.role == 'test'].copy()
cols = ['accuracy', 'macro_f1', 'mse_standardized', 'cos_standardized']
summary = m.groupby('method')[cols].mean()
for col in ['classifier_rowspace_rank', 'classifier_rowspace_error_energy', 'classifier_nullspace_error_energy', 'centered_logit_shift_l2']:
    summary[col] = p.groupby('method')[col].mean()
summary.to_csv(ROOT / 'evidence/existing_method_comparison.csv')
splits = json.loads((SOURCE / 'splits.json').read_text())
fold_info = []
for s in splits:
    if s['fold'] == 'all_known':
        continue
    bags = s['train_bags']
    labels = p[p.bag_id.isin(bags)].groupby('bag_id').label.first()
    count = labels.value_counts().to_dict()
    assert len(labels) == len(bags)
    fold_info.append(dict(fold=s['fold'], n_train_bags=len(bags), n_train_bags_per_class={str(k): int(v) for k, v in count.items()}, one_training_recording_per_class=all((v == 1 for v in count.values())), maximum_centered_bag_matrix_rank=len(bags) - 1))
meta = dict(status='read_only_summary_of_preexisting_development_results', source=str(SOURCE), distinct_recordings=int(p.bag_id.nunique()), folds=int(p.fold.nunique()), seeds=sorted(map(int, p.seed.unique())), folds_checked=fold_info, classification_head_rank=sorted(map(int, p.classifier_rowspace_rank.unique())), caveats=['Not a new independent test.', 'Outer-race ROI was previously flagged invalid.', 'Seed repetitions are not additional physical recordings.', 'One bag per class makes training bag targets identical to class prototypes.', 'This does not establish that the learned student actually collapses to prototypes.'])
(ROOT / 'evidence/existing_summary.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
order = ['ce_only', 'embedding_only', 'ce_feat_1_kd']
names = ['Labels only', 'Bag features only', 'Full distillation']
colors = ['#758595', '#2685b5', '#00a388']
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
for ax, column, title in zip(axes, ['accuracy', 'mse_standardized', 'cos_standardized'], ['Recording accuracy', 'Teacher-feature MSE (lower is better)', 'Teacher-feature cosine']):
    values = summary.loc[order, column].to_numpy()
    ax.bar(np.arange(3), values, color=colors, width=0.6)
    ax.set_xticks(np.arange(3), names, fontsize=9, rotation=13)
    ax.set_title(title, fontsize=11)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim(0, 1.16)
    for i, v in enumerate(values):
        ax.text(i, v + 0.025, f'{v:.3f}', ha='center', fontsize=10)
fig.suptitle('Existing development results: identical classification, different feature fidelity', fontsize=13)
fig.text(0.5, 0.015, '8 distinct recordings; 2 folds x 3 seeds. No confidence interval implied. Outer-race ROI remains a confound.', ha='center', fontsize=9, color='#555555')
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(ROOT / f'figures/existing_accuracy_vs_fidelity.{ext}', dpi=180, bbox_inches='tight')
plt.close(fig)
print(json.dumps(meta, ensure_ascii=False, indent=2))
print(summary.to_string())
