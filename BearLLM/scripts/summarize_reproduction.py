"""Summarize completed artifacts; never promote missing or partial runs to success."""
from c2r_paths import resolve_path as _c2r_resolve_path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parents[1]
PAIR_COUNTS = {'train': 257865, 'val': 73674, 'test': 36837}
QUERY_COUNTS = {'train': 85955, 'val': 24558, 'test': 12279}
CORPUS_TASKS = ('fault_detection', 'fault_classification', 'maintenance', 'risk_analysis')

def read(path):
    path = Path(_c2r_resolve_path(path))
    return json.loads(path.read_text()) if path.is_file() else None

def percent(value):
    return '待完成' if value is None else f'{value * 100:.2f}%'

def metric(report, model, group, key='accuracy'):
    if not report:
        return None
    return report.get('metrics', {}).get(model, {}).get(group, {}).get(key)

def checked(report, count_key, expected):
    return bool(report and report.get('complete') and (not report.get('smoke_test_only')) and (report.get(count_key) == expected))

def file_hash(path):
    path = Path(_c2r_resolve_path(path))
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def integer(value, minimum=0):
    return isinstance(value, int) and (not isinstance(value, bool)) and (value >= minimum)

def finite(value, minimum=None, maximum=None):
    return isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(value) and (minimum is None or value >= minimum) and (maximum is None or value <= maximum)

def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Artifact contains NaN or infinity')

def same_number(actual, expected, name):
    require(actual is None if expected is None else finite(actual) and math.isclose(actual, expected, rel_tol=1e-08, abs_tol=1e-10), f'Inconsistent {name}: {actual!r} vs {expected!r}')

def validate_classification(item, expected=None, classes=10, training=False):
    """Reconcile the scalar metrics with the confusion matrix, including errors."""
    finite_tree(item)
    count = item['n' if training else 'count']
    require(integer(count), 'Invalid classification count')
    require(expected is None or count == expected, f'Classification count {count} != {expected}')
    matrix = item['confusion_matrix']
    columns = classes if training else classes + 1
    require(len(matrix) == classes and all((len(row) == columns for row in matrix)), 'Invalid confusion shape')
    require(all((integer(v) for row in matrix for v in row)), 'Invalid confusion entries')
    support = [sum(row) for row in matrix]
    require(sum(support) == count, 'Confusion matrix does not cover the reported count')
    tp = [matrix[i][i] for i in range(classes)]
    predicted = [sum((row[i] for row in matrix)) for i in range(classes)]
    precision = [tp[i] / predicted[i] if predicted[i] else 0.0 for i in range(classes)]
    recall = [tp[i] / support[i] if support[i] else 0.0 for i in range(classes)]
    f1 = [2 * tp[i] / (support[i] + predicted[i]) if support[i] + predicted[i] else 0.0 for i in range(classes)]
    same_number(item['accuracy'], sum(tp) / count if count else None, 'accuracy')
    same_number(item['macro_f1_all_10_classes' if training else 'macro_f1_fixed_classes'], sum(f1) / classes, 'macro F1')
    if training:
        require(item['correct'] == sum(tp), 'Incorrect correct-prediction count')
        same_number(item['balanced_accuracy_present_classes'], sum((recall[i] for i in range(classes) if support[i])) / sum((bool(v) for v in support)) if count else None, 'balanced accuracy')
        same_number(item['false_alarm_rate'], sum(matrix[0][1:]) / support[0] if support[0] else None, 'false alarm rate')
        same_number(item['missed_alarm_rate'], sum((row[0] for row in matrix[1:])) / sum(support[1:]) if sum(support[1:]) else None, 'missed alarm rate')
    else:
        require(item['invalid_predictions'] == sum((row[-1] for row in matrix)), 'Unparseable count mismatch')
        same_number(item['macro_f1_present_classes'], sum((f1[i] for i in range(classes) if support[i])) / sum((bool(v) for v in support)) if count else None, 'present-class macro F1')
        require(item['confusion_columns'] == list(range(classes)) + ['unparseable'], 'Invalid confusion columns')
        require(len(item['per_class']) == classes, 'Missing per-class metrics')
        for i, row in enumerate(item['per_class']):
            require(row['label'] == i and row['support'] == support[i], 'Per-class support mismatch')
            for key, expected_value in (('precision', precision[i]), ('recall', recall[i]), ('f1', f1[i])):
                same_number(row[key], expected_value, f'class {i} {key}')
    return count

def validate_grouped_classification(groups, count, source_counts=None):
    validate_classification(groups['all'], count)
    for name, item in groups.items():
        if name != 'all':
            validate_classification(item)
    sources = {key[7:]: item['count'] for key, item in groups.items() if key.startswith('source/')}
    require(sum(sources.values()) == count, 'Per-source counts do not cover all observations')
    if source_counts is not None:
        require(sources == {k: v for k, v in source_counts.items() if v}, 'Unexpected per-source coverage')

def validate_vibration(report, audit, fallback=False):
    expected, unique = (38172, 12724) if fallback else (368376, 122792)
    require(checked(report, 'count_pairs', expected), 'Incomplete or smoke vibration evaluation')
    finite_tree(report)
    require(report['count_unique_queries'] == unique, 'Incorrect unique query count')
    require(report['reference_policy'] == ('excluded-same-source-fallback' if fallback else 'same-condition'), 'Wrong vibration reference policy')
    key = 'missing_same_condition_healthy_reference_samples' if fallback else 'same_condition_healthy_reference_samples'
    sources = {name: row[key] * 3 for name, row in audit['metadata']['sources'].items()}
    for model in ('pre_lora_adapter', 'final_lora_adapter'):
        groups = report['metrics'][model]
        validate_grouped_classification(groups, expected, sources)
        if not fallback:
            for split, count in PAIR_COUNTS.items():
                validate_classification(groups['split/' + split], count)

def validate_corpus(report, audit):
    require(checked(report, 'count', 600) and report['total'] == 600, 'Incomplete or smoke corpus replay')
    finite_tree(report)
    groups = report['metrics']
    require(groups['all']['count'] == 600, 'Corpus aggregate count mismatch')
    require({key for key in groups if key.startswith('task/')} == {'task/' + task for task in CORPUS_TASKS}, 'Unexpected corpus task groups')
    for task in CORPUS_TASKS:
        require(groups['task/' + task]['count'] == 150, f'Corpus task {task} must contain 150 rows')
    sources = {key[7:]: row['count'] for key, row in groups.items() if key.startswith('source/')}
    require(sources == audit['corpus']['source_counts'], 'Corpus source coverage mismatch')
    for name, group in groups.items():
        count = group['count']
        require(integer(count, 1), 'Invalid corpus group count')
        require(integer(group['truncated_outputs']) and group['truncated_outputs'] <= count, 'Invalid corpus truncation count')
        require(finite(group['mean_generated_tokens'], 0), 'Invalid generated-token mean')
        require(finite(group['reference_token_nll'], 0), 'Invalid reference NLL')
        same_number(group['reference_perplexity'], math.exp(min(700, group['reference_token_nll'])), 'perplexity')
        for key in ('normalized_exact_match', 'unigram_f1', 'rouge_l_f1'):
            require(finite(group['text_overlap'][key], 0, 1), f'Invalid text metric {key}')
        validate_classification(group['adapter_classification'], count)
        generated = 0
        for key, classes in (('generated_binary_detection', 2), ('generated_classification', 10)):
            if key in group:
                generated += validate_classification(group[key], classes=classes)
        require(generated == count, 'Generated classification does not cover the corpus group')
        if name == 'all':
            require(group['generated_binary_detection']['count'] == 150 and group['generated_classification']['count'] == 450, 'Corpus task family counts mismatch')
        elif name.startswith('task/'):
            family = 'generated_binary_detection' if name == 'task/fault_detection' else 'generated_classification'
            require(group[family]['count'] == 150, 'Corpus task classification count mismatch')

def validate_heldout(report):
    require(checked(report, 'count', 12279), 'Incomplete or smoke test-query generation')
    finite_tree(report)
    require(integer(report['truncated_outputs']) and report['truncated_outputs'] <= 12279, 'Invalid held-out truncation count')
    require(finite(report['mean_generated_tokens'], 0), 'Invalid generated-token mean')
    for model in ('generated_classification', 'adapter_classification'):
        validate_grouped_classification(report['metrics'][model], 12279)

def read_epochs(path):
    if not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError:
        return []

def validate_training_config(config, stage, data_dir, qwen_dir, metadata_hash, corpus_hash):
    expected = {'stage': stage, 'seed': 42, 'epochs': 50, 'batch_size': 1024 if stage == 'pretrain' else 1, 'accumulation_steps': 4, 'scheduler': 'source-batch-loss', 'learning_rate': 0.0001, 'metadata_sha256': metadata_hash, 'corpus_sha256': corpus_hash}
    for key, value in expected.items():
        require(config.get(key) == value, f'Training {stage} config mismatch: {key}')
    require(Path(_c2r_resolve_path(config['data_dir'])).resolve() == data_dir.resolve(), 'Training dataset path mismatch')
    require(Path(_c2r_resolve_path(config['qwen_dir'])).resolve() == qwen_dir.resolve(), 'Training backbone path mismatch')
    require(config['effective_batch_size'] == (1024 if stage == 'pretrain' else 4), 'Effective batch size mismatch')
    loaders = {'train_shuffle': True, 'validation_shuffle': True, 'test_shuffle': True, 'persistent_workers': False} if stage == 'pretrain' else {'engine': 'transformers.Trainer', 'dataloader_num_workers': 0, 'loss_normalization': 'native Trainer model loss kwargs'}
    require(config.get('loader_protocol') == loaders, 'Training loader/engine differs from the source protocol')

def validate_fcn_metrics(report, count):
    require(finite(report['loss'], 0), 'Invalid FCN loss')
    groups = report['by_dataset']
    validate_classification(groups['MBHM'], count, training=True)
    require(sum((row['n'] for name, row in groups.items() if name != 'MBHM')) == count, 'FCN dataset counts do not cover the split')
    for name, row in groups.items():
        if name != 'MBHM':
            validate_classification(row, training=True)

def validate_pretraining(report, run_root, data_dir, qwen_dir, metadata_hash, corpus_hash):
    require(report is not None, 'Pretraining results are missing')
    finite_tree(report)
    require(report['stage'] == 'pretrain' and report['requested_epochs'] == 50, 'Wrong pretraining stage/epoch target')
    validate_training_config(report['config'], 'pretrain', data_dir, qwen_dir, metadata_hash, corpus_hash)
    require(report['config']['disable_early_stop'] is False, 'Source early stopping was disabled')
    protocol = report['protocol']
    require(protocol['seed'] == 42 and protocol['total_queries'] == 135516 and (protocol['eligible_queries'] == 122792), 'Pretraining query protocol mismatch')
    require(protocol['pair_counts'] == PAIR_COUNTS and protocol['query_counts'] == QUERY_COUNTS, 'Pretraining split coverage mismatch')
    completed = report['epochs_completed']
    require(integer(completed, 1) and completed <= 50, 'Invalid completed pretraining epochs')
    epochs = read_epochs(run_root / 'pretrain/epochs.jsonl')
    require(len(epochs) == completed and [row['epoch'] for row in epochs] == list(range(1, completed + 1)), 'Pretraining epoch log is incomplete')
    best_acc, best_loss, best_epoch, selected_epoch = (-1.0, math.inf, None, None)
    for row in epochs:
        finite_tree(row)
        require(row['stage'] == 'pretrain' and row['train_pairs'] == PAIR_COUNTS['train'], 'An FCN epoch did not train on all pairs')
        require(finite(row['train_loss'], 0) and finite(row['lr'], 0), 'Invalid FCN epoch loss/LR')
        validate_fcn_metrics(row['val'], PAIR_COUNTS['val'])
        accuracy, loss = (row['val']['by_dataset']['MBHM']['accuracy'], row['val']['loss'])
        selected = accuracy > best_acc or loss < best_loss
        if accuracy > best_acc:
            best_epoch = row['epoch']
        if selected:
            selected_epoch = row['epoch']
        best_acc, best_loss = (max(best_acc, accuracy), min(best_loss, loss))
        require(row.get('source_checkpoint_selected') is selected and row.get('source_selected_epoch') == selected_epoch, 'FCN checkpoint selection differs from source loss OR accuracy rule')
    require(completed == 50 or epochs[-1]['lr'] < 1e-07, 'FCN stopped before the configured/source stopping condition')
    for key in ('last_checkpoint_test', 'best_checkpoint_test', 'source_selected_checkpoint_test'):
        validate_fcn_metrics(report[key], PAIR_COUNTS['test'])
    require(report['best_epoch'] == best_epoch, 'Invalid best-accuracy epoch')
    require(report['source_selected_epoch'] == selected_epoch, 'Invalid source-selected epoch')
    for name in ('feature_encoder.pth', 'classifier.pth'):
        actual = file_hash(run_root / 'pretrain/fcn' / name)
        require(actual and report['exported_fcn_sha256'].get(name) == actual, 'Selected FCN export hash mismatch')

def validate_finetuning(report, run_root, data_dir, qwen_dir, metadata_hash, corpus_hash):
    require(report is not None, 'Fine-tuning results are missing')
    finite_tree(report)
    require(report['stage'] == 'finetune' and report['requested_epochs'] == 50 and (report['epochs_completed'] == 50) and (report['corpus_rows'] == 600), 'Fine-tuning did not complete 50 full corpus epochs')
    validate_training_config(report['config'], 'finetune', data_dir, qwen_dir, metadata_hash, corpus_hash)
    for record in (report, report['config']):
        require(record.get('finetune_engine') == 'transformers.Trainer' and record.get('model_accepts_loss_kwargs') is True and (record.get('loss_normalization') == 'native Trainer model loss kwargs'), 'Fine-tuning does not use the verified native Trainer loss normalization')
    require(Path(_c2r_resolve_path(report['config']['fcn_dir'])).resolve() == (run_root / 'pretrain/fcn').resolve(), 'Wrong FCN initializer directory')
    for name in ('feature_encoder.pth', 'classifier.pth'):
        actual = file_hash(run_root / 'pretrain/fcn' / name)
        require(actual and report['config']['initial_fcn_sha256'].get(name) == actual, 'FCN-to-LoRA hash chain mismatch')
    require(report['global_step'] == 7500, 'Fine-tuning optimizer update count mismatch')
    epochs = read_epochs(run_root / 'finetune/epochs.jsonl')
    require(len(epochs) == 50 and [row['epoch'] for row in epochs] == list(range(1, 51)), 'Fine-tuning epoch log is incomplete')
    for row in epochs:
        finite_tree(row)
        require(row['stage'] == 'finetune' and row['corpus_rows'] == 600 and (row['unique_corpus_ids'] == 600) and (row['global_step'] == row['epoch'] * 150), 'An epoch did not train on the complete corpus')
        require(finite(row['train_loss'], 0) and finite(row['lr'], 0), 'Invalid fine-tuning epoch loss/LR')
    protocol = read(run_root / 'finetune/corpus_protocol.json')
    require(protocol and protocol['rows'] == 600 and (protocol['total_optimizer_updates'] == 7500), 'Fine-tuning corpus protocol mismatch')
    require(protocol['task_counts'] == {str(i): 150 for i in range(4)} and protocol['label_counts'] == {str(i): 60 for i in range(10)}, 'Fine-tuning task/label coverage mismatch')
    require(protocol.get('finetune_engine') == 'transformers.Trainer' and protocol.get('model_accepts_loss_kwargs') is True and (protocol.get('loss_normalization') == 'native Trainer model loss kwargs'), 'Corpus protocol disagrees with native Trainer configuration')
    trainer_args = read(run_root / 'finetune/trainer_arguments.json')
    for key, expected in {'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 4, 'num_train_epochs': 50, 'learning_rate': 0.0001, 'lr_scheduler_type': 'cosine', 'seed': 42, 'dataloader_num_workers': 0}.items():
        require(trainer_args and trainer_args.get(key) == expected, f'Native Trainer argument mismatch: {key}')
    weights = run_root / 'finetune/weights'
    require(Path(_c2r_resolve_path(report['weights'])).resolve() == weights.resolve(), 'Wrong fine-tuned export directory')
    for name in ('vibration_adapter.pth', 'adapter_model.safetensors'):
        actual = file_hash(weights / name)
        require(actual and report['exported_weights_sha256'].get(name) == actual, 'Fine-tuned export hash mismatch')

def validation_error(function, *args):
    try:
        function(*args)
        return None
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        return f'{function.__name__}: {error}'

def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_safe(child) for child in value]
    return None if isinstance(value, float) and (not math.isfinite(value)) else value

def alarm_rates(classification):
    matrix = classification['confusion_matrix']
    healthy, faulty = (sum(matrix[0]), sum((sum(row) for row in matrix[1:])))
    return {'healthy_support': healthy, 'faulty_support': faulty, 'false_alarm_rate': sum(matrix[0][1:]) / healthy if healthy else None, 'missed_alarm_rate': sum((row[0] for row in matrix[1:])) / faulty if faulty else None}

def make_figures(output, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    for tag in ('official', 'trained'):
        item = reports[tag + '_main']
        if not checked(item, 'count_pairs', 368376):
            continue
        matrix = np.asarray(item['metrics']['final_lora_adapter']['all']['confusion_matrix'])[:, :10]
        support = matrix.sum(1, keepdims=True)
        normalized = np.divide(matrix, support, out=np.zeros(matrix.shape, float), where=support > 0)
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        chart = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        names = ['Healthy', 'Inner minor', 'Inner moderate', 'Inner severe', 'Ball minor', 'Ball moderate', 'Ball severe', 'Outer minor', 'Outer moderate', 'Outer severe']
        supported_names = [f'{name} (n={int(n):,})' for name, n in zip(names, support[:, 0])]
        overall = item['metrics']['final_lora_adapter']['all']
        ax.set(xticks=range(10), yticks=range(10), xticklabels=names, yticklabels=supported_names, xlabel='Predicted class', ylabel='True class', title=f"{tag.title()} final adapter — source-compatible pairs\n122,792 queries / 368,376 pairs\nAccuracy {overall['accuracy']:.2%} | Macro-F1 {overall['macro_f1_fixed_classes']:.2%}")
        plt.setp(ax.get_xticklabels(), rotation=50, ha='right')
        for i in range(10):
            for j in range(10):
                if normalized[i, j] >= 0.005:
                    ax.text(j, i, f'{normalized[i, j]:.1%}', ha='center', va='center', fontsize=8, color='white' if normalized[i, j] > 0.5 else 'black')
        fig.colorbar(chart, ax=ax, label='Fraction of true-class observations')
        fig.savefig(output / f'{tag}_confusion.png', dpi=170)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42')))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/full')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = read(ROOT / 'outputs/mbhm_audit.json')
    environment = read(ROOT / 'outputs/full/environment_ready.json')
    pretrain = read(args.run_root / 'pretrain/results.json')
    finetune = read(args.run_root / 'finetune/results.json')
    source_export_diagnostic = read(ROOT / 'outputs/full/source_export_diagnostic.json')
    reports = {}
    provenance = {}
    from dotenv import dotenv_values
    env = dotenv_values(ROOT / '.env')
    metadata_hash = file_hash(Path(_c2r_resolve_path(env['MBHM_DATASET'])) / 'metadata.sqlite')
    corpus_hash = file_hash(Path(_c2r_resolve_path(env['MBHM_DATASET'])) / 'corpus.json')
    for tag in ('official', 'trained'):
        reports[tag + '_main'] = read(args.run_root / f'{tag}_evaluation/vibration_metrics.json')
        reports[tag + '_corpus'] = read(args.run_root / f'{tag}_evaluation/corpus_metrics.json')
        reports[tag + '_fallback'] = read(args.run_root / f'{tag}_fallback/vibration_metrics.json')
        reports[tag + '_heldout'] = read(args.run_root / f'{tag}_heldout/heldout_metrics.json')
        weights = Path(_c2r_resolve_path(env['BEARLLM_WEIGHTS'])) if tag == 'official' else args.run_root / 'finetune/weights'
        adapter_hash = file_hash(weights / 'vibration_adapter.pth')
        lora_hash = file_hash(weights / 'adapter_model.safetensors')
        for suffix, stage, policy in [('evaluation', 'all', 'same-condition'), ('fallback', 'vibration', 'excluded-same-source-fallback'), ('heldout', 'heldout-generation', 'same-condition')]:
            config = read(args.run_root / f'{tag}_{suffix}/run_config.json')
            provenance[tag + '_' + suffix] = bool(config and adapter_hash and lora_hash and (config.get('vibration_adapter_sha256') == adapter_hash) and (config.get('lora_sha256') == lora_hash) and (config.get('seed') == 42) and (Path(_c2r_resolve_path(config.get('checkpoint', ''))).resolve() == weights.resolve()) and (config.get('stage') == stage) and (config.get('reference_policy') == policy) and (config.get('references') == 3) and (config.get('split') == 'test') and (config.get('limit') is None) and (config.get('metadata_sha256') == metadata_hash) and (config.get('corpus_sha256') == corpus_hash) and (Path(_c2r_resolve_path(config.get('qwen', ''))).resolve() == Path(_c2r_resolve_path(env['QWEN_WEIGHTS'])).resolve()))
    validation_errors = {}
    training_args = (args.run_root, Path(_c2r_resolve_path(env['MBHM_DATASET'])), Path(_c2r_resolve_path(env['QWEN_WEIGHTS'])), metadata_hash, corpus_hash)
    validation_errors['full_pretraining'] = validation_error(validate_pretraining, pretrain, *training_args)
    validation_errors['full_finetuning'] = validation_error(validate_finetuning, finetune, *training_args)
    checks = {'complete_data_audit': bool(audit and audit.get('scope') == 'complete' and (len(audit['checks']) >= 15) and all(audit['checks'].values()) and (audit['metadata']['sample_count'] == 135516)), 'independent_conda_environment': bool(environment and environment.get('status') == 'passed'), 'full_pretraining': validation_errors['full_pretraining'] is None, 'full_finetuning': validation_errors['full_finetuning'] is None}
    for tag in ('official', 'trained'):
        checks[tag + '_checkpoint_and_protocol_provenance'] = all((provenance[tag + '_' + suffix] for suffix in ('evaluation', 'fallback', 'heldout')))
        for name, key, function, extra in [('main', 'all_supported_pairs', validate_vibration, (audit,)), ('fallback', 'all_fallback_pairs', validate_vibration, (audit, True)), ('corpus', 'all_corpus_rows', validate_corpus, (audit,)), ('heldout', 'all_test_queries_generated', validate_heldout, ())]:
            check_name = tag + '_' + key
            validation_errors[check_name] = validation_error(function, reports[tag + '_' + name], *extra)
            checks[check_name] = validation_errors[check_name] is None
    derived_rates = {}
    for tag in ('official', 'trained'):
        for scope, directory, check in (('main', 'evaluation', 'all_supported_pairs'), ('fallback', 'fallback', 'all_fallback_pairs')):
            key = tag + '_' + scope
            derived_rates[key] = alarm_rates(reports[key]['metrics']['final_lora_adapter']['all']) if checks[tag + '_' + check] and provenance[tag + '_' + directory] else None
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'complete_current_release_pipeline': all(checks.values()), 'checks': checks, 'provenance_checks': provenance, 'validation_errors': {key: error for key, error in validation_errors.items() if error is not None}, 'derived_alarm_rates': derived_rates, 'run_root': str(args.run_root), 'pretrain': pretrain, 'finetune': finetune, 'source_export_diagnostic': source_export_diagnostic, 'reports': reports, 'limitations': ['Current released-code protocol, not exact paper configuration.', 'Healthy references cross train/validation/test splits.', 'Corpus results replay all 600 training corpus entries.', 'Cross-condition same-source fallback is auxiliary and not upstream-comparable.', "Official checkpoint's exact original train membership is unavailable.", 'Raw vibration acquisition and full paper corpus are not reproduced from unavailable files.']}
    (args.output / 'summary.json').write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for tag in ('official', 'trained'):
        for name, directory, check in [('main', 'evaluation', 'all_supported_pairs'), ('corpus', 'evaluation', 'all_corpus_rows'), ('fallback', 'fallback', 'all_fallback_pairs'), ('heldout', 'heldout', 'all_test_queries_generated')]:
            report_key = tag + '_' + name
            if not provenance[tag + '_' + directory] or not checks[tag + '_' + check]:
                reports[report_key] = None
    lines = ['# MBHM / BearLLM 实测结果', '', '状态：' + ('当前公开代码的完整训练和验证流程已完成。' if all(checks.values()) else '仍有阶段待完成，下面仅展示已有结果。'), '', '## 完成性核验', '', '| 阶段 | 状态 |', '|---|---|']
    lines += [f"| {name} | {('通过' if passed else '待完成')} |" for name, passed in checks.items()]
    lines += ['', '## 原协议振动分类', '', '统计单位为查询—参考配对；同一查询对应 3 次参考。内部适配器分类和语言生成指标分别报告。新训练主结果保留训练末期的 BatchNorm 状态；原仓库保存方式另列对照。', '', '| 数据源 | 全部查询 | 原协议查询 | 缺少同工况参考 | 官方最终适配器准确率 | 新训练最终适配器准确率 |', '|---|---:|---:|---:|---:|---:|']
    if audit:
        for source, counts in audit['metadata']['sources'].items():
            lines.append(f"| {source} | {counts['samples']} | {counts['same_condition_healthy_reference_samples']} | {counts['missing_same_condition_healthy_reference_samples']} | {percent(metric(reports['official_main'], 'final_lora_adapter', 'source/' + source))} | {percent(metric(reports['trained_main'], 'final_lora_adapter', 'source/' + source))} |")
    lines += ['', 'CWRU、DIRG、NCEPU 的原协议子集仅含正常样本，以上准确率不能证明这些数据源的故障识别能力。MFPT 可配对子集为 18 条正常样本和 18 条类别 8 故障样本。', '', '| 评测范围 | 官方准确率 / 固定十类 Macro-F1 | 新训练准确率 / 固定十类 Macro-F1 |', '|---|---:|---:|']
    for suffix, label in [('main', '原协议全部配对'), ('fallback', '补充跨工况参考配对（非原协议）')]:
        cells = []
        for tag in ('official', 'trained'):
            item = reports[tag + '_' + suffix]
            cells.append(percent(metric(item, 'final_lora_adapter', 'all')) + ' / ' + percent(metric(item, 'final_lora_adapter', 'all', 'macro_f1_fixed_classes')))
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines += ['', '| 评测范围 | 官方误报率 FA | 官方漏报率 | 新训练误报率 FA | 新训练漏报率 |', '|---|---:|---:|---:|---:|']
    for scope, label in (('main', '原协议全部配对'), ('fallback', '补充跨工况参考配对（非原协议）')):
        cells = []
        for tag in ('official', 'trained'):
            rates = derived_rates[tag + '_' + scope]
            for key, absent in (('false_alarm_rate', '无正常样本'), ('missed_alarm_rate', '无故障样本')):
                cells.append('待完成' if rates is None else f'不适用（{absent}）' if rates[key] is None else percent(rates[key]))
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines += ['', '误报率 = 正常配对被预测为非正常的数量 / 正常配对数；漏报率 = 故障配对被预测为正常的数量 / 故障配对数。分母为零时不报告百分比。']
    lines += ['', '## 训练与语言生成', '']
    if checks['full_pretraining']:
        result = pretrain['source_selected_checkpoint_test']['by_dataset']['MBHM']
        lines.append(f"FCN 完成 {pretrain['epochs_completed']}/50 轮，按源码 loss 或 accuracy 改进规则选用第 {pretrain['source_selected_epoch']} 轮。该模型测试查询配对准确率 {percent(result['accuracy'])}，Macro-F1 {percent(result['macro_f1_all_10_classes'])}。查询集合分离，但正常参考仍可能跨划分重用。")
    if checks['full_finetuning']:
        lines += ['', f"LoRA 完成 {finetune['epochs_completed']}/50 轮，训练全部 {finetune['corpus_rows']} 条公开语料。"]
    lines += ['', '| 权重 | 测试查询生成分类准确率 | Macro-F1 | 无法解析 | 截断数量 |', '|---|---:|---:|---:|---:|']
    for tag, label in [('official', '官方'), ('trained', '新训练')]:
        item = reports[tag + '_heldout']
        if item:
            result = item['metrics']['generated_classification']['all']
            lines.append(f"| {label} | {percent(result['accuracy'])} | {percent(result['macro_f1_fixed_classes'])} | {result['invalid_predictions']} | {item['truncated_outputs']} |")
        else:
            lines.append(f'| {label} | 待完成 | 待完成 | — | — |')
    if source_export_diagnostic and source_export_diagnostic.get('verified'):
        comparison = source_export_diagnostic['comparison']['generated_classification']
        lines += ['', f"原仓库保存方式对照：同一最终 LoRA 配合微调前 BatchNorm 状态，全部 12,279 条测试查询生成准确率为 {percent(comparison['source_initial_bn']['accuracy'])}；与主结果只相差 BatchNorm 状态。两组均保留，不按测试分数择优替换。完整对照见 `source_export_diagnostic.md`。"]
    lines += ['', '测试查询生成使用全部 12,279 个原协议测试查询。官方权重原始训练成员未知；新训练模型的查询划分保存在实验目录。', '', '| 权重 | 训练语料回放数量 | ROUGE-L | 参考回答困惑度 | 截断数量 |', '|---|---:|---:|---:|---:|']
    for tag, label in [('official', '官方'), ('trained', '新训练')]:
        item = reports[tag + '_corpus']
        if item:
            result = item['metrics']['all']
            lines.append(f"| {label} | {item['count']} | {result['text_overlap']['rouge_l_f1']:.4f} | {result['reference_perplexity']:.4f} | {result['truncated_outputs']} |")
        else:
            lines.append(f'| {label} | 待完成 | 待完成 | 待完成 | — |')
    lines += ['', '| 任务（各 150 条） | 官方标签解析匹配率 | 新训练标签解析匹配率 | 官方 ROUGE-L | 新训练 ROUGE-L |', '|---|---:|---:|---:|---:|']
    for task, label in [('fault_detection', '故障检测'), ('fault_classification', '故障分类'), ('maintenance', '维护建议'), ('risk_analysis', '风险分析')]:
        task_results = [(reports[tag + '_corpus'] or {}).get('metrics', {}).get('task/' + task) for tag in ('official', 'trained')]
        key = 'generated_binary_detection' if task == 'fault_detection' else 'generated_classification'
        accuracy = [percent(item[key]['accuracy']) if item else '待完成' for item in task_results]
        overlap = [f"{item['text_overlap']['rouge_l_f1']:.4f}" if item else '待完成' for item in task_results]
        lines.append(f"| {label} | {' | '.join(accuracy + overlap)} |")
    lines += ['', '维护与风险任务的标签匹配率仅检查回答中最早出现的标准类别名称，不评价建议或风险判断的事实正确性。公开参考回答 ID 394、505 本身未提供实际诊断，仍保留计分，详见 `reference_label_audit.json`。', '', '600 条语料结果是训练样本回放，不能称为未见文本的泛化成绩。文字重合及困惑度不替代维护建议的事实审查。', '', '详细命令与协议见 `FULL_REPRODUCTION_zh.md`；逐样本预测、混淆矩阵、每类及每数据源统计保存在各实验目录。', '']
    (args.output / 'RESULTS_zh.md').write_text('\n'.join(lines))
    make_figures(args.output, reports)
    print(json.dumps({'complete': all(checks.values()), 'checks': checks}, ensure_ascii=False, indent=2))
    if args.require_complete and (not all(checks.values())):
        raise SystemExit('Required full-reproduction stages are not all complete.')
if __name__ == '__main__':
    main()
