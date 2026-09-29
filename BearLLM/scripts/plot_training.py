"""Export training curves from completed epoch logs, without estimating scores."""
from c2r_paths import resolve_path as _c2r_resolve_path
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42')))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/full/training_curves.png')
    args = parser.parse_args()
    history = {}
    results = {}
    for stage in ('pretrain', 'finetune'):
        results[stage] = json.loads((args.run_root / stage / 'results.json').read_text())
        history[stage] = [json.loads(line) for line in (args.run_root / stage / 'epochs.jsonl').read_text().splitlines()]
        assert [row['epoch'] for row in history[stage]] == list(range(1, results[stage]['epochs_completed'] + 1))
    pre, fine = (history['pretrain'], history['finetune'])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot([r['epoch'] for r in pre], [r['train_loss'] for r in pre], color='#176B87')
    axes[0, 0].set(title='FCN training loss', ylabel='Mean cross-entropy')
    axes[0, 1].plot([r['epoch'] for r in pre], [100 * r['val']['by_dataset']['MBHM']['accuracy'] for r in pre], color='#287D52')
    axes[0, 1].set(title='FCN validation: 73,674 query-reference pairs', ylabel='Accuracy (%)')
    selected = results['pretrain'].get('source_selected_epoch')
    if selected:
        axes[0, 1].axvline(selected, color='#BE6B28', linestyle='--', label=f'Source-selected epoch {selected}')
        axes[0, 1].legend()
    axes[1, 0].plot([r['epoch'] for r in fine], [r['train_loss'] for r in fine], color='#8A4D91')
    axes[1, 0].set(title='LoRA training loss: 600 released corpus rows', ylabel='Mean response loss')
    for stage, label, color in [('pretrain', 'FCN', '#176B87'), ('finetune', 'LoRA', '#8A4D91')]:
        records = [r for r in history[stage] if r['lr'] > 0]
        axes[1, 1].plot([r['epoch'] for r in records], [r['lr'] for r in records], label=label, color=color)
    axes[1, 1].set(title='Learning rate (positive values)', ylabel='Learning rate', yscale='log')
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel('Completed epoch')
        ax.grid(alpha=0.2)
    fig.suptitle('BearLLM released-code reproduction | seed 42\nValidation queries are separated; healthy references may cross splits.', fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)
    print(args.output)
if __name__ == '__main__':
    main()
