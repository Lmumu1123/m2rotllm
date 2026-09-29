"""Actual frozen Qwen/BearLLM generation from coarse p4 probabilities.

This adds an explicit semantic bridge; it is not the original waveform adapter,
and uniform severity probabilities contain no measured severity information.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import argparse, importlib.util, json, os, re, sys, time
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
HERE = Path(_c2r_resolve_path(__file__)).resolve().parent
ROOT = Path(_c2r_resolve_path('/home/huangyating/BearLLM'))
RUN = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42'))
ASSETS = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-assets'))
PROMPT = 'Based on the bearing state #state_place_holder#, classify the bearing into exactly one coarse category: Normal, Inner race fault, Outer race fault, or Rolling element fault. Answer with only one category. Do not state fault severity.'
COARSE = ['normal', 'inner', 'outer', 'ball']

def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def coarse_parse(text):
    text = text.lower()
    patterns = ['\\b(?:normal|healthy|fault[- ]free)\\b', '\\binner\\s+(?:race|ring)\\b', '\\bouter\\s+(?:race|ring)\\b', '\\b(?:rolling\\s+element|ball)\\b']
    matches = [i for i, p in enumerate(patterns) if re.search(p, text)]
    return matches[0] if len(matches) == 1 else -1

def p4_to_p10(p4):
    p10 = np.zeros((len(p4), 10), np.float32)
    p10[:, 0] = p4[:, 0]
    p10[:, 1:4] = p4[:, 1, None] / 3
    p10[:, 7:10] = p4[:, 2, None] / 3
    p10[:, 4:7] = p4[:, 3, None] / 3
    assert np.allclose(p10.sum(-1), 1, atol=2e-06)
    return p10

def control_rows():
    rows = []
    for label in range(4):
        rows.append(dict(id=f'oracle_{COARSE[label]}', method='oracle_coarse_onehot', fold='control', bag_id='oracle', true_label=label, **{f'p{i}': float(i == label) for i in range(4)}))
    orig = pd.read_csv(HERE / 'retrained_fcn_fixed_external/file_predictions.csv')
    for _, r in orig.iterrows():
        rows.append(dict(id='original_contact_' + r['file'], method='original_contact_fcn_bridge', fold='control', bag_id=r['bag_id'], true_label=int(r.label), state=r.state, **{f'p{i}': float(r[f'p4_{i}']) for i in range(4)}))
    return rows

def summarize_groups(df, output):
    """Do not pool external healthy/unknown examples into main diagnostic F1."""
    df = df.copy()
    for key, default in [('task', 'interface_control'), ('fold', 'control'), ('direction', 'control'), ('seed', -1), ('state', 'unspecified')]:
        if key not in df:
            df[key] = default
        else:
            df[key] = df[key].fillna(default)

    def scope(row):
        if row['state'] == 'keep':
            return 'external_unknown'
        if row['state'] == 'bigNormal':
            return 'external_normal'
        if row['fold'] == 'all_known':
            return 'external_other'
        if row['task'] == 'interface_control':
            return 'interface_control'
        return 'heldout_main'
    df['evaluation_scope'] = df.apply(scope, axis=1)
    from sklearn.metrics import f1_score, balanced_accuracy_score
    keys = ['task', 'fold', 'direction', 'method', 'seed', 'evaluation_scope']

    def one(g, values, by_state=False):
        info = dict(zip(keys + (['state'] if by_state else []), values))
        labels = [0, 1, 3] if str(info['task']).startswith('three_class') else [0, 1, 2, 3]
        known = g[g.true_label >= 0]
        metric = dict(**info, n=len(g), labelled_n=len(known), probability_fidelity=float(g.probability_faithful.mean()), unparseable=int((g.parsed_coarse_label < 0).sum()), truncated=int(g.truncated.sum()))
        metric['accuracy'] = float((known.true_label == known.parsed_coarse_label).mean()) if len(known) else None
        main = info['evaluation_scope'] in ['heldout_main', 'interface_control'] and (not by_state)
        metric['macro_f1'] = float(f1_score(known.true_label, known.parsed_coarse_label, labels=labels, average='macro', zero_division=0)) if main and len(known) else None
        metric['balanced_accuracy'] = float(balanced_accuracy_score(known.true_label, known.parsed_coarse_label)) if main and len(known) else None
        metric['macro_f1_class_ids'] = ','.join(map(str, labels)) if main else ''
        metric['external_normal_false_alarm_rate'] = float((g.parsed_coarse_label != 0).mean()) if info['evaluation_scope'] == 'external_normal' else None
        metric['unknown_detection_metric'] = None
        return metric
    groups = [one(g, values) for values, g in df.groupby(keys, dropna=False, sort=False)]
    states = [one(g, values, True) for values, g in df.groupby(keys + ['state'], dropna=False, sort=False)]
    pd.DataFrame(groups).to_csv(output / 'generation_metrics.csv', index=False)
    pd.DataFrame(states).to_csv(output / 'generation_metrics_by_state.csv', index=False)
    df.to_csv(output / 'generation_predictions.csv', index=False)
    return groups

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--inputs', type=Path, default=HERE.parent / 'results/llm_bridge_inputs.csv')
    a.add_argument('--output', type=Path, default=HERE / 'llm_bridge')
    a.add_argument('--controls-only', action='store_true')
    a.add_argument('--no-controls', action='store_true')
    a.add_argument('--device', default='cuda:0')
    a.add_argument('--batch', type=int, default=8)
    a.add_argument('--max-new-tokens', type=int, default=48)
    args = a.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [] if args.no_controls else control_rows()
    if not args.controls_only:
        if not args.inputs.is_file():
            raise FileNotFoundError(args.inputs)
        rows.extend(pd.read_csv(args.inputs, dtype={'fold': str}).to_dict(orient='records'))
    rows = [{k: None if pd.isna(v) else v for k, v in r.items()} for r in rows]
    assert rows and len({str(r['id']) for r in rows}) == len(rows)
    p4 = np.asarray([[r[f'p{i}'] for i in range(4)] for r in rows], np.float32)
    if not np.isfinite(p4).all() or (p4 < 0).any() or (not np.allclose(p4.sum(-1), 1, atol=0.0001)):
        raise ValueError('Input p0..p3 must be finite, nonnegative, sum to one')
    p4 = p4 / p4.sum(-1, keepdims=True)
    p10 = p4_to_p10(p4)
    pd.DataFrame(rows).to_csv(args.output / 'inputs_used.csv', index=False)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    original_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        upstream = module('upstream_bearllm_eval', ROOT / 'scripts/evaluate_mbhm.py')
        from transformers import AutoTokenizer
        from src.fine_tuning import signal_token_id, description_len, llm_hidden_size
        checkpoint = RUN / 'finetune/weights'
        model, adapter = upstream.load_llm(checkpoint, ASSETS / 'qwen_weights', args.device)
    finally:
        os.chdir(original_cwd)
    tokenizer = AutoTokenizer.from_pretrained(ASSETS / 'qwen_weights')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    sequence = upstream.prompt_ids(tokenizer, PROMPT)
    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(False)
    assert description_len == 5 and llm_hidden_size == 1536
    base = torch.load(checkpoint / 'vibration_adapter.pth', map_location='cpu', weights_only=True)
    saved = load_file(checkpoint / 'adapter_model.safetensors')
    cfg = json.loads((checkpoint / 'adapter_config.json').read_text())
    pre = 'base_model.model.model.embed_tokens.adapter.alignment_layer.linear3.'
    weight = base['alignment_layer.linear3.weight'].float() + saved[pre + 'lora_B.weight'].float() @ saved[pre + 'lora_A.weight'].float() * (cfg['lora_alpha'] / cfg['r'])
    bias = base['alignment_layer.linear3.bias'].float()
    test = torch.as_tensor(p10[:4], device=args.device)
    with torch.inference_mode():
        live = adapter.alignment_layer.linear3(test).float().cpu()
        merged = torch.nn.functional.linear(test.cpu(), weight, bias)
    delta = float((live - merged).abs().max())
    if not torch.allclose(live, merged, rtol=0.0002, atol=0.0002):
        raise ValueError(f'linear3 LoRA equivalence failed: max difference {delta}')
    started = time.monotonic()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch):
            q = torch.tensor(p10[start:start + args.batch], device=args.device)
            ids = sequence.to(args.device)[None].expand(len(q), -1).clone()
            attention = torch.ones_like(ids)
            mask = ids.eq(signal_token_id)
            assert torch.all(mask.sum(-1) == 5)
            embeddings = model.get_input_embeddings()(ids.masked_fill(mask, 0))
            tokens = adapter.alignment_layer.linear3(q).reshape(len(q), 5, 1536).to(embeddings.dtype)
            embeddings[mask] = tokens.reshape(-1, 1536)
            generated = model.generate(input_ids=ids, inputs_embeds=embeddings, attention_mask=attention, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            for j in range(len(q)):
                index = start + j
                tokens_out = generated[j, len(sequence):].cpu().tolist()
                text = tokenizer.decode(tokens_out, skip_special_tokens=True).strip()
                parsed = coarse_parse(text)
                r = rows[index]
                truth = int(r['true_label'])
                result = {**r, 'p10': p10[index].tolist(), 'generated_text': text, 'parsed_coarse_label': parsed, 'probability_argmax': int(p4[index].argmax()), 'probability_faithful': bool(parsed == p4[index].argmax()), 'correct': bool(parsed == truth) if truth >= 0 else None, 'truncated': len(tokens_out) >= args.max_new_tokens and tokenizer.eos_token_id not in tokens_out, 'generated_token_count': len(tokens_out)}
                outputs.append(result)
            print(f'generated {len(outputs)}/{len(rows)}', flush=True)
    with (args.output / 'generation_predictions.jsonl').open('w') as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    df = pd.DataFrame(outputs)
    df.to_csv(args.output / 'generation_predictions.csv', index=False)
    summary = summarize_groups(df, args.output)
    audit = dict(prompt=PROMPT, rows=len(rows), device=args.device, python=sys.executable, torch=torch.__version__, weights={str(checkpoint / fn): upstream.sha256(checkpoint / fn) for fn in ['vibration_adapter.pth', 'adapter_model.safetensors', 'adapter_config.json']}, qwen_model_sha256=upstream.sha256(ASSETS / 'qwen_weights/model.safetensors'), input_sha256=upstream.sha256(args.inputs) if not args.controls_only else None, script_sha256=upstream.sha256(__file__), linear3_merged_equivalence_max_abs_difference=delta, fixed_models=True, true_label_used_in_prompt_or_embeddings=False, seed=42, do_sample=False, elapsed_seconds=time.monotonic() - started, bridge='p0->q0; p1/3->q1,q2,q3; p2/3->q7,q8,q9; p3/3->q4,q5,q6; live LoRA linear3 -> 5x1536 embeddings', limitation='New coarse semantic interface, not original waveform inference. Uniform severity is a placeholder, never a severity estimate. Probability fidelity is measured separately from diagnostic accuracy.')
    upstream.write_json(args.output / 'provenance.json', audit)
    print(json.dumps(summary, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), flush=True)
if __name__ == '__main__':
    main()
