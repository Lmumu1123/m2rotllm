#!/usr/bin/env python3
"""Frozen native BearLLM p10 -> LoRA linear3 -> actual Qwen generation.

Consumes genuine ten-way model probabilities. It never expands p4 back to p10.
Ground truth is used only after generation for evaluation. Rule guards and raw
LLM diagnosis are reported separately; guards are not learned improvements.
"""
from pathlib import Path
import argparse,importlib.util,json,os,re,sys,time
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
BEAR=Path('/home/huangyating/BearLLM')
RUN=Path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42')
ASSETS=Path('/media/nas_users/huangyating/bearllm-assets')
GROUPS=[[0],[1,2,3],[7,8,9],[4,5,6]]
NAMES=['Normal','Inner race fault','Outer race fault','Rolling element fault']
PROMPT=('Based on the bearing state #state_place_holder#, classify the bearing into exactly one coarse '
        'category: Normal, Inner race fault, Outer race fault, or Rolling element fault. '
        'Answer with only one category. Do not state fault severity.')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def parse(text):
    patterns=[r'\b(?:normal|healthy|fault[- ]free)\b',r'\binner\s+(?:race|ring)\b',
              r'\bouter\s+(?:race|ring)\b',r'\b(?:rolling\s+element|ball)\b']
    found=[i for i,p in enumerate(patterns) if re.search(p,text.lower())]
    return found[0] if len(found)==1 else -1

def truth_bool(value,default=True):
    if value is None:return default
    if isinstance(value,str):
        if value.lower() in ['true','1','yes']:return True
        if value.lower() in ['false','0','no']:return False
        raise ValueError(f'Invalid boolean metadata value: {value!r}')
    return bool(value)

def controls():
    # Pure native ten-class interfaces; internal severity indices are synthetic
    # control values, never measured severity claims about local recordings.
    rows=[]
    for coarse,ten in enumerate([0,1,7,4]):
        rows.append(dict(id=f'native_p10_onehot_{ten}',task='interface_control',stage='control',
            variant='native_onehot',fold='control',method='native_p10_onehot',state='control',
            bag_id=f'synthetic_{ten}',true_label=coarse,geometry_valid=True,
            **{f'p10_{i}':float(i==ten) for i in range(10)}))
    return rows

def summarize(df,out):
    from sklearn.metrics import f1_score
    keys=['stage','variant','fold','method','evaluation_scope']
    results=[]
    for vals,g in df.groupby(keys,dropna=False,sort=False):
        base=dict(zip(keys,vals));known=g[g.true_label>=0]
        main=base['evaluation_scope'] in ['heldout_main','interface_control']
        valid=g[g.geometry_valid];vk=valid[valid.true_label>=0]
        rec=dict(**base,n=len(g),labelled_n=len(known),geometry_valid_n=len(valid),
            raw_probability_fidelity=float(g.probability_faithful.mean()),
            raw_unparseable=int((g.parsed_coarse_label<0).sum()),
            raw_truncated=int(g.truncated.sum()),raw_class_conflicts=int(g.llm_class_conflict.sum()),
            guard_fallbacks=int(g.guard_applied.sum()),
            raw_accuracy=float((known.parsed_coarse_label==known.true_label).mean()) if len(known) else None,
            raw_macro_f1=float(f1_score(known.true_label,known.parsed_coarse_label,labels=range(4),average='macro',zero_division=0)) if main and len(known) else None,
            classifier_accuracy=float((known.probability_argmax==known.true_label).mean()) if len(known) else None,
            raw_accuracy_geometry_valid=float((vk.parsed_coarse_label==vk.true_label).mean()) if len(vk) else None,
            guarded_accuracy_geometry_valid=float((vk.guarded_label==vk.true_label).mean()) if len(vk) else None,
            raw_external_normal_false_alarm_rate=float((g.parsed_coarse_label!=0).mean()) if base['evaluation_scope']=='external_normal' else None,
            learned_unknown_detection_metric=None)
        results.append(rec)
    pd.DataFrame(results).to_csv(out/'generation_metrics.csv',index=False)
    return results

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,default=ROOT/'radar/llm_original_p10_inputs.csv')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--controls-only',action='store_true')
    ap.add_argument('--no-controls',action='store_true')
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--max-new-tokens',type=int,default=48)
    args=ap.parse_args()
    if '/envs/m2vllm/' not in sys.executable:raise RuntimeError('Run in conda m2vllm')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[] if args.no_controls else controls()
    if not args.controls_only:
        rows.extend(pd.read_csv(args.inputs,dtype={'fold':str}).to_dict('records'))
    rows=[{k:None if pd.isna(v) else v for k,v in r.items()} for r in rows]
    if not rows or len({str(r['id']) for r in rows})!=len(rows):raise ValueError('Missing rows or duplicate ids')
    for r in rows:
        for k,v in [('task','four_class'),('stage','unspecified'),('variant','unspecified'),('fold','unspecified'),('method','unspecified'),('state','unspecified')]:
            if r.get(k) is None:r[k]=v
        r['geometry_valid']=truth_bool(r.get('geometry_valid'),True)
        r['true_label']=int(r.get('true_label',-1))
    p10=np.asarray([[r[f'p10_{i}'] for i in range(10)] for r in rows],np.float32)
    if not np.isfinite(p10).all() or (p10<0).any() or not np.allclose(p10.sum(-1),1,atol=1e-4):
        raise ValueError('Native p10 must be finite, nonnegative and sum to one')
    oldp=p10.copy();p10=p10/p10.sum(-1,keepdims=True)
    p4=np.stack([p10[:,g].sum(1) for g in GROUPS],axis=1)
    pd.DataFrame(rows).to_csv(args.output/'inputs_used.csv',index=False)
    torch.manual_seed(42);torch.set_num_threads(2)
    cwd=Path.cwd()
    try:
        os.chdir(BEAR);sys.path.insert(0,str(BEAR))
        upstream=module('upstream_bearllm_native_p10',BEAR/'scripts/evaluate_mbhm.py')
        from transformers import AutoTokenizer
        from src.fine_tuning import signal_token_id,description_len,llm_hidden_size
        checkpoint=RUN/'finetune/weights'
        model,adapter=upstream.load_llm(checkpoint,ASSETS/'qwen_weights',args.device)
    finally:os.chdir(cwd)
    assert description_len==5 and llm_hidden_size==1536
    tokenizer=AutoTokenizer.from_pretrained(ASSETS/'qwen_weights');tokenizer.pad_token_id=tokenizer.eos_token_id
    sequence=upstream.prompt_ids(tokenizer,PROMPT)
    model.eval();adapter.eval()
    for obj in [model,adapter]:
        for param in obj.parameters():param.requires_grad_(False)
    weight_paths=[checkpoint/fn for fn in ['vibration_adapter.pth','adapter_model.safetensors','adapter_config.json']]
    weight_hashes={str(p):upstream.sha256(p) for p in weight_paths}
    base=torch.load(checkpoint/'vibration_adapter.pth',map_location='cpu',weights_only=True)
    saved=load_file(checkpoint/'adapter_model.safetensors')
    cfg=json.loads((checkpoint/'adapter_config.json').read_text())
    if any(cfg.get(k) for k in ['use_rslora','use_dora','rank_pattern','alpha_pattern','fan_in_fan_out']):raise ValueError('Unsupported merge config')
    pre='base_model.model.model.embed_tokens.adapter.alignment_layer.linear3.'
    weight=base['alignment_layer.linear3.weight'].float()+saved[pre+'lora_B.weight'].float()@saved[pre+'lora_A.weight'].float()*(cfg['lora_alpha']/cfg['r'])
    bias=base['alignment_layer.linear3.bias'].float()
    with torch.inference_mode():
        probe=torch.as_tensor(p10[:4],device=args.device)
        live=adapter.alignment_layer.linear3(probe).float().cpu()
        merged=torch.nn.functional.linear(probe.cpu(),weight,bias)
    delta=float((live-merged).abs().max())
    if not torch.allclose(live,merged,rtol=2e-4,atol=2e-4):raise ValueError(f'linear3 equivalence failed: {delta}')
    outputs=[];started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch):
            pp=torch.tensor(p10[start:start+args.batch],device=args.device)
            ids=sequence.to(args.device)[None].expand(len(pp),-1).clone()
            mask=ids.eq(signal_token_id);attention=torch.ones_like(ids)
            assert torch.all(mask.sum(-1)==5)
            embeddings=model.get_input_embeddings()(ids.masked_fill(mask,0))
            vt=adapter.alignment_layer.linear3(pp).reshape(len(pp),5,1536).to(embeddings.dtype)
            embeddings[mask]=vt.reshape(-1,1536)
            generated=model.generate(input_ids=ids,inputs_embeds=embeddings,attention_mask=attention,
                max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
            for j in range(len(pp)):
                i=start+j;r=rows[i];gt=r['true_label']
                ts=generated[j,len(sequence):].cpu().tolist()
                raw=tokenizer.decode(ts,skip_special_tokens=True).strip();parsed=parse(raw)
                pred=int(p4[i].argmax());conflict=parsed>=0 and parsed!=pred
                if not r['geometry_valid']:
                    guard_label=-1;guard_text='Invalid radar geometry: reacquire or reprocess the target range before diagnosis.';guard_reason='invalid_geometry'
                elif parsed!=pred:
                    guard_label=pred;guard_text=NAMES[pred];guard_reason='unparseable_output' if parsed<0 else 'llm_class_conflict'
                else:guard_label=parsed;guard_text=raw;guard_reason='none'
                scope='interface_control' if r['task']=='interface_control' else 'external_unknown' if r['state']=='keep' else 'external_normal' if r['state']=='bigNormal' else 'heldout_main'
                outputs.append({**r,'p10_used':p10[i].tolist(),'p4_grouped':p4[i].tolist(),
                    'generated_text':raw,'parsed_coarse_label':parsed,'probability_argmax':pred,
                    'probability_faithful':parsed==pred,'llm_class_conflict':conflict,
                    'correct':parsed==gt if gt>=0 else None,'evaluation_scope':scope,
                    'truncated':len(ts)>=args.max_new_tokens and tokenizer.eos_token_id not in ts,
                    'generated_token_count':len(ts),'guarded_text':guard_text,'guarded_label':guard_label,
                    'guard_reason':guard_reason,'guard_applied':guard_reason!='none',
                    'diagnosis_usable':r['geometry_valid'],'guard_uses_ground_truth':False})
            print(f'generated {len(outputs)}/{len(rows)}',flush=True)
    assert all(upstream.sha256(p)==weight_hashes[str(p)] for p in weight_paths)
    df=pd.DataFrame(outputs);df.to_csv(args.output/'generation_predictions.csv',index=False)
    with (args.output/'generation_predictions.jsonl').open('w') as f:
        for row in outputs:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    metrics=summarize(df,args.output)
    upstream.write_json(args.output/'provenance.json',dict(prompt=PROMPT,rows=len(rows),python=sys.executable,
        device=args.device,torch=torch.__version__,weights=weight_hashes,weights_unchanged=True,
        qwen_model_sha256=upstream.sha256(ASSETS/'qwen_weights/model.safetensors'),
        input_snapshot_sha256=upstream.sha256(args.output/'inputs_used.csv'),script_sha256=upstream.sha256(__file__),
        linear3_merged_equivalence_max_abs_difference=delta,max_probability_rounding_normalization=float(np.max(abs(p10-oldp))),
        native_p10_used=True,p4_expanded_into_severity=False,true_label_used_in_prompt_or_embeddings=False,
        group_map=GROUPS,bridge='native p10 -> actual frozen final LoRA linear3 -> 5x1536 -> actual frozen Qwen with trained LoRA',
        fixed_models=True,do_sample=False,seed=42,elapsed_seconds=time.monotonic()-started,
        guard='Probability argmax fallback for parse/class conflict; actual geometry metadata invalidates diagnosis. Raw output is retained. No truth or fault label used by guard.',
        severity='Internal ten-way probabilities preserved, but local severity lacks ground truth and no severity accuracy is claimed.',
        limitation='Native probability interface validation; this is not new LLM training. Classifier accuracy and raw LLM fidelity are separate. Rule guard consistency is not learned model accuracy.'))
    print(json.dumps(dict(rows=len(rows),raw_probability_fidelity=float(df.probability_faithful.mean()),
        class_conflicts=int(df.llm_class_conflict.sum()),unparseable=int((df.parsed_coarse_label<0).sum()),
        invalid_geometry=int((~df.geometry_valid).sum())),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
