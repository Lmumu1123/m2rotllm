"""Independently audit actual prediction rows against metadata and training pairs."""
from c2r_paths import resolve_path as _c2r_resolve_path
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
from dotenv import dotenv_values
from evaluate_mbhm import classification_metrics, corpus_summary, parse_binary, parse_class, text_metrics
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def rows(path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except Exception as error:
                raise ValueError(f'Invalid JSONL: {path}:{number}') from error

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def integer(value, minimum=0):
    return isinstance(value, int) and (not isinstance(value, bool)) and (value >= minimum)

def compare_metrics(actual, expected, path='metrics'):
    """Compare re-aggregated predictions with their saved metrics, including errors."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'Metric keys differ: {path}')
        for key, value in expected.items():
            compare_metrics(actual[key], value, path + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'Metric shape differs: {path}')
        for index, value in enumerate(expected):
            compare_metrics(actual[index], value, f'{path}/{index}')
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and (not isinstance(actual, bool)) and math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-08, abs_tol=1e-10), f'Metric differs: {path}')
    else:
        require(actual == expected, f'Metric differs: {path}')

def check_generation_fields(item, binary=False):
    require(isinstance(item['prediction'], str), 'Generated prediction must be text')
    expected = parse_binary(item['prediction']) if binary else parse_class(item['prediction'])
    require(isinstance(item['parsed_prediction'], int) and (not isinstance(item['parsed_prediction'], bool)) and (item['parsed_prediction'] == expected), 'Stored parsed prediction disagrees with generated text')
    require(integer(item['adapter_prediction']) and item['adapter_prediction'] < 10, 'Invalid adapter prediction')
    require(integer(item['generated_tokens']) and isinstance(item['truncated'], bool), 'Invalid generation length/truncation fields')

def check_heldout(path, expected, metadata):
    seen, groups = (set(), defaultdict(list))
    truncated = tokens = 0
    for item in rows(path):
        query = item['file_id']
        require(query in expected and query not in seen, 'Invalid/duplicate held-out query')
        require((item['ref_id'], item['label'], item['pair_index']) == expected[query] and item['reference_index'] == 0 and (item['split'] == 'test'), 'Wrong held-out reference/label')
        condition, label, source = metadata[query]
        require(item['condition_id'] == condition and item['label'] == label and (item['source'] == source), 'Held-out metadata mismatch')
        check_generation_fields(item)
        for group in ('all', 'source/' + source):
            groups[group].append(item)
        truncated += item['truncated']
        tokens += item['generated_tokens']
        seen.add(query)
    require(seen == set(expected), 'Incomplete held-out generation')
    saved = json.loads(path.with_name('heldout_metrics.json').read_text())
    require(saved['complete'] is True and saved['smoke_test_only'] is False and (saved['count'] == len(seen)), 'Held-out metrics are incomplete or smoke-only')
    rebuilt = {model: {name: classification_metrics(((row['label'], row[key]) for row in items)) for name, items in groups.items()} for model, key in [('generated_classification', 'parsed_prediction'), ('adapter_classification', 'adapter_prediction')]}
    compare_metrics(saved['metrics'], rebuilt)
    compare_metrics(saved['truncated_outputs'], truncated)
    compare_metrics(saved['mean_generated_tokens'], tokens / len(seen) if seen else None)
    return {'count': len(seen), 'sha256': digest(path), 'generation_fields_and_aggregate_metrics': 'passed'}

def check_corpus(path, corpus, metadata):
    seen, actual = (set(), [])
    for item in rows(path):
        identifier = item['id']
        require(identifier in corpus and identifier not in seen, 'Invalid/duplicate corpus row')
        require(all((item[key] == value for key, value in corpus[identifier].items())), 'Corpus query/prompt/reference altered')
        require(item['source'] == metadata[item['vib_id']][2], 'Corpus source mismatch')
        require(isinstance(item['reference_nll_sum'], (int, float)) and (not isinstance(item['reference_nll_sum'], bool)) and math.isfinite(item['reference_nll_sum']) and (item['reference_nll_sum'] >= 0) and integer(item['reference_token_count'], 1), 'Invalid reference likelihood')
        check_generation_fields(item, binary=item['task_id'] == 0)
        for key, expected in text_metrics(item['prediction'], item['response']).items():
            compare_metrics(item[key], expected, key)
        actual.append(item)
        seen.add(identifier)
    require(seen == set(corpus), 'Incomplete corpus generation')
    saved = json.loads(path.with_name('corpus_metrics.json').read_text())
    require(saved['complete'] is True and saved['smoke_test_only'] is False and (saved['count'] == len(corpus)) and (saved['total'] == len(corpus)), 'Corpus metrics are incomplete or smoke-only')
    compare_metrics(saved['metrics'], corpus_summary(actual, len(corpus), False)['metrics'])
    return {'count': len(seen), 'sha256': digest(path), 'generation_fields_and_aggregate_metrics': 'passed'}

def check_pairs(path, expected, metadata):
    seen = set()
    queries = Counter()
    matrices = {name: [[0] * 10 for _ in range(10)] for name in ('pre_lora_adapter', 'final_lora_adapter')}
    for item in rows(path):
        key = (item['file_id'], item['reference_index'])
        require(key not in seen and key in expected, f'Duplicate or unexpected pair {key}')
        reference, label, split, pair_index = expected[key]
        require((item['ref_id'], item['label'], item['split'], item['pair_index']) == (reference, label, split, pair_index), f'Pair differs from protocol: {key}')
        condition, true_label, source = metadata[item['file_id']]
        require(item['condition_id'] == condition and item['source'] == source and (label == true_label), f'Metadata mismatch: {key}')
        for name, matrix in matrices.items():
            probabilities = item[name]['probabilities']
            require(len(probabilities) == 10 and all((math.isfinite(p) and 0 <= p <= 1.000001 for p in probabilities)) and (abs(sum(probabilities) - 1) < 1e-05), f'Invalid probabilities: {name}/{key}')
            prediction = max(range(10), key=probabilities.__getitem__)
            require(prediction == item[name]['prediction'], f'Prediction/argmax mismatch: {key}')
            matrix[label][prediction] += 1
        seen.add(key)
        queries[item['file_id']] += 1
    require(seen == set(expected), f'Incomplete pair coverage: {path}, {len(seen)}/{len(expected)}')
    require(set(queries.values()) == {3}, f'Queries do not each have three reference draws: {path}')
    metrics = json.loads(path.with_name('vibration_metrics.json').read_text())
    require(metrics['complete'] is True and metrics['smoke_test_only'] is False and (metrics['count_pairs'] == len(seen)) and (metrics['count_unique_queries'] == len(queries)), 'Vibration metrics are incomplete or smoke-only')
    for name, matrix in matrices.items():
        actual = metrics['metrics'][name]['all']['confusion_matrix']
        require([row + [0] for row in matrix] == actual, f'Predictions disagree with aggregate confusion matrix: {name}')
    return (set(queries), {'pairs': len(seen), 'unique_queries': len(queries), 'sha256': digest(path), 'metadata_and_seeded_references': 'passed', 'probabilities_and_aggregate_counts': 'passed'})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42')))
    parser.add_argument('--tag', choices=('official', 'trained', 'both'), default='both')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/full/prediction_coverage.json')
    args = parser.parse_args()
    data = Path(_c2r_resolve_path(dotenv_values(ROOT / '.env')['MBHM_DATASET']))
    with sqlite3.connect(f"file:{data / 'metadata.sqlite'}?mode=ro", uri=True) as db:
        metadata = {i: (condition, label, source) for i, condition, label, source in db.execute('SELECT file_id,f.condition_id,label,dataset FROM file_info f JOIN "condition" c ON f.condition_id=c.condition_id ORDER BY file_id')}
    training = json.loads((args.run_root / 'pretrain/dataset.json').read_text())
    main_expected = {}
    for split, pairs in training.items():
        reference_indices = Counter()
        for query, reference, label in pairs:
            index = reference_indices[query]
            require(index < 3 and (query, index) not in main_expected, 'Duplicate/excess training query reference draw')
            reference_indices[query] += 1
            main_expected[query, index] = (reference, label, split)
    main_queries = sorted({query for query, _ in main_expected})
    healthy_conditions_pool = defaultdict(list)
    for query, (condition, label, _) in metadata.items():
        if label == 0:
            healthy_conditions_pool[condition].append(query)
    source_rng = random.Random(42)
    for query_index, query in enumerate(main_queries):
        condition, true_label, _ = metadata[query]
        split = 'train' if query_index % 10 < 7 else 'val' if query_index % 10 < 9 else 'test'
        for index in range(3):
            require(main_expected[query, index] == (source_rng.choice(healthy_conditions_pool[condition]), true_label, split), 'Training pair manifest differs from the seeded upstream healthy-reference protocol')
            main_expected[query, index] += (query_index * 3 + index,)
    healthy_conditions = {condition for condition, label, _ in metadata.values() if label == 0}
    excluded = sorted((query for query, (condition, _, _) in metadata.items() if condition not in healthy_conditions))
    healthy_sources = defaultdict(list)
    for query, (_, label, source) in metadata.items():
        if label == 0:
            healthy_sources[source].append(query)
    rng = random.Random(42)
    fallback_expected = {}
    for query_index, query in enumerate(excluded):
        _, label, source = metadata[query]
        for index in range(3):
            fallback_expected[query, index] = (rng.choice(healthy_sources[source]), label, 'excluded_from_upstream', query_index * 3 + index)
    corpus = {row['id']: row for row in json.loads((data / 'corpus.json').read_text())}
    tags = ('official', 'trained') if args.tag == 'both' else (args.tag,)
    report = {'status': 'passed', 'models': {}, 'metadata_sha256': digest(data / 'metadata.sqlite'), 'training_pairs_sha256': digest(args.run_root / 'pretrain/dataset.json')}
    for tag in tags:
        main_path = args.run_root / f'{tag}_evaluation/vibration_predictions.jsonl'
        fallback_path = args.run_root / f'{tag}_fallback/vibration_predictions.jsonl'
        supported, main_result = check_pairs(main_path, main_expected, metadata)
        additional, fallback_result = check_pairs(fallback_path, fallback_expected, metadata)
        require(not supported & additional and supported | additional == set(metadata), 'Incomplete/disjoint coverage failure')
        require(len(supported) == 122792 and len(additional) == 12724 and (len(metadata) == 135516), 'Wrong dataset scope')
        expected_test = {q: (r, label, index) for (q, draw), (r, label, split, index) in main_expected.items() if split == 'test' and draw == 0}
        heldout_path = args.run_root / f'{tag}_heldout/heldout_predictions.jsonl'
        require(len(expected_test) == 12279, 'Incorrect test-query protocol size')
        heldout_result = check_heldout(heldout_path, expected_test, metadata)
        corpus_path = args.run_root / f'{tag}_evaluation/corpus_predictions.jsonl'
        require(len(corpus) == 600, 'Incorrect corpus protocol size')
        corpus_result = check_corpus(corpus_path, corpus, metadata)
        report['models'][tag] = {'main': main_result, 'fallback': fallback_result, 'all_unique_queries': len(supported | additional), 'disjoint_union_matches_all_metadata': True, 'heldout': heldout_result, 'corpus': corpus_result}
        print(f'{tag}: all 135516 vibration queries, 12279 held-out generations and 600 corpus rows verified.', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
if __name__ == '__main__':
    main()
