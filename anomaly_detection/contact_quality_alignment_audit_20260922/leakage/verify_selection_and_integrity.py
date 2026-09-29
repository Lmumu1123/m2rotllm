from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import hashlib, json
import pandas as pd
ROOT = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/retention_alignment_20260922'))
OLD = ROOT.parent / 'four_class_chain_20260922'
OUT = Path(_c2r_resolve_path(__file__)).resolve().parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
v1 = ROOT / 'results/retained_head_v1'
c = pd.read_csv(v1 / 'candidates.csv')
checks = []
for (variant, fold), part in c.groupby(['variant', 'fold']):
    valid = part[(part.source_weight > 0) & part.retention_gate]
    best = valid.sort_values(['train_file_macro_f1', 'train_ce', 'val_acc10', 'step', 'source_weight'], ascending=[False, True, False, True, True]).iloc[0]
    selected = json.loads((v1 / 'models' / variant / fold / 'selection_before_test.json').read_text())
    matched = int(best.source_weight) == selected['source_weight'] and int(best.step) == selected['step']
    assert matched
    checks.append(dict(version='v1', variant=variant, fold=fold, source_weight=selected['source_weight'], step=selected['step'], selection_reproduces_declared_train_and_source_val_rule=True, used_local_test_in_rule=False, used_source_test_in_rule=False))
co = ROOT / 'results/retained_head_conservative'
c = pd.read_csv(co / 'candidates.csv')
for variant, part in c.groupby('variant'):
    valid = part[part.all_source_val_gate]
    best = valid.sort_values(['local_train_macro_f1', 'val_acc10', 'val_acc4', 'delta_norm'], ascending=[False, False, False, True]).iloc[0]
    selected = json.loads((co / 'models' / variant / 'all_known/selection_before_test.json').read_text())
    assert int(best.lambda_source) == selected['lambda_source'] and int(best.step) == selected['step']
    checks.append(dict(version='conservative', variant=variant, fold='all_known', source_weight=selected['lambda_source'], step=selected['step'], selection_reproduces_declared_train_and_source_val_rule=True, used_local_test_in_rule=False, used_source_test_in_rule=False))
pd.DataFrame(checks).to_csv(OUT / 'head_selection_rule_recheck.csv', index=False)
files = pd.read_csv(OUT / 'local_file_hashes.csv').set_index('file')
manifests = json.loads((OLD / 'contact/input_manifest.json').read_text()) + json.loads((OLD / 'radar/input_manifest.json').read_text())
n = 0
for row in manifests:
    p = Path(_c2r_resolve_path(row.get('file', row.get('path'))))
    assert files.loc[p.name, 'sha256'] == row['sha256']
    n += 1
codes = []
for folder, script in [(v1, ROOT / 'scripts/train_retained_head.py'), (co, ROOT / 'scripts/refine_retention.py')]:
    comp = json.loads((folder / 'completion.json').read_text())
    match = sha(script) == comp['script_sha256']
    assert match
    codes.append(dict(script=str(script), sha256=sha(script), matches_completed_training_record=True))
(OUT / 'selection_and_integrity.json').write_text(json.dumps(dict(selected_models_checked=len(checks), selected_parameters_match_declared_rule=True, local_inputs_match_previous_manifest=n, completed_training_code_matches_current=codes, scope='This verifies within-run candidate selection. It does not undo protocol redesign after previous test results or inherited cross-split references.'), ensure_ascii=False, indent=2) + '\n')
print('selection models', len(checks), 'input files', n, 'training code hashes matched')
