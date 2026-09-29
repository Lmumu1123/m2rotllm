"""Frozen BearLLM-head diagnostic controls. All fitting uses training recordings only.

This is a development replay on previously inspected recordings, not a new blind test.
The outer-race radar ROI remains invalid and is flagged throughout.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import json, hashlib, random, time, sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parent
ARCHIVE = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/retention_alignment_20260922/radar/stage_b_v0'))
CONTACT = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/retention_alignment_20260922/preprocessing/variants/v0/retrained_fcn'))
RADAR = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_chain_20260922/radar'))
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]
METHODS = ['ce_only', 'matched_bag_mse', 'full', 'label_code_mse', 'label_code_ce_mse', 'wrong_class_mse']
SEEDS = [17, 42, 73]
STEPS = 400

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(Path(_c2r_resolve_path(path)).read_bytes()).hexdigest()

def weights(meta):
    counts = meta.groupby('bag_id').size()
    return np.array([1 / counts[b] for b in meta.bag_id]) * len(meta) / len(counts)

def grouping(logits):
    return torch.stack([torch.logsumexp(logits[..., ids], -1) for ids in GROUPS], -1)

def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

class Encoder(nn.Module):

    def __init__(self, mean, scale):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 128))
        self.register_buffer('contact_mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('contact_scale', torch.tensor(scale, dtype=torch.float32))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.relu(self.contact_mean + self.contact_scale * self.net(x))

def make_label_codes(w, b, mean, scale):
    """Uses no class/recording-specific teacher features, no radar, no held-out values.

    Global train-contact mean/scale are explicitly shared across all conditions.
    Optimize four standardized codes through the actual frozen grouped head.
    A fixed small quadratic penalty keeps codes near the global feature scale.
    """
    mean = torch.tensor(mean, dtype=torch.float32)
    scale = torch.tensor(scale, dtype=torch.float32)
    z = nn.Parameter(torch.zeros(4, 128))
    opt = torch.optim.Adam([z], lr=0.05)
    labels = torch.arange(4)
    trace = []
    for step in range(400):
        h = F.relu(mean + scale * z)
        actual_z = (h - mean) / scale
        ce = F.cross_entropy(grouping(h @ w.T + b), labels)
        loss = ce + 0.001 * actual_z.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (0, 99, 199, 299, 399):
            trace.append({'step': step + 1, 'loss': float(loss.detach()), 'ce': float(ce.detach())})
    with torch.inference_mode():
        h = F.relu(mean + scale * z)
        p = grouping(h @ w.T + b).softmax(-1)
        actual_z = (h - mean) / scale
    return (h, actual_z, p, trace)

def main():
    if 'envs/m2vllm' not in sys.executable:
        raise RuntimeError('Use m2vllm')
    torch.set_num_threads(2)
    out = ROOT / 'results'
    if out.exists():
        raise RuntimeError('Refusing to overwrite controls results')
    out.mkdir()
    for name in ('windows', 'models', 'targets'):
        (out / name).mkdir()
    start = time.time()
    cm = pd.read_csv(CONTACT / 'metadata.csv')
    rm = pd.read_csv(RADAR / 'metadata.csv')
    h = np.load(CONTACT / 'contact_features.npz')['hidden_mean'].astype(np.float32)
    rz = np.load(RADAR / 'features.npz')
    x = rz['frame_shape'].astype(np.float32)
    assert h.shape == (len(cm), 128) and np.array_equal(rz['bag_id'], rm.bag_id)
    splits = [s for s in json.loads((ARCHIVE / 'splits.json').read_text()) if s['test_bags']]
    files = [CONTACT / 'metadata.csv', CONTACT / 'contact_features.npz', RADAR / 'metadata.csv', RADAR / 'features.npz', ARCHIVE / 'splits.json'] + [Path(_c2r_resolve_path(s['shared_head'])) for s in splits]
    input_hashes = {str(p): sha(p) for p in files}
    protocol = dict(methods=METHODS, seeds=SEEDS, encoder_steps=STEPS, optimizer='AdamW lr=.001 weight_decay=.01 grad_clip=5', batch='16 windows sampled with replacement per training recording', label_code_construction='400 Adam lr=.05 steps: grouped_head_CE + .001 standardized code L2; uses only frozen head and global train-contact mean/scale', all_methods_same_scalers_initialization_minibatches=True, prior_development_data=True, test_based_model_or_hyperparameter_selection=False, primary_evaluation='all fixed final-step models, two recording splits; no bearing/motor holdout', limitations=['outer-race radar ROI invalid', 'each class has one training recording per fold', 'no synchronized segment matching', 'same-class shuffling and within-class cross-recording CKA unavailable', 'label codes exploit learned shared head plus global contact scales, not label-only from scratch'], source_head='existing stage_b_v0 retained_head_v1; not the later conservative all-known head')
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'input_hashes_before.json', input_hashes)
    rm.to_csv(out / 'radar_window_metadata.csv', index=False)
    cm.to_csv(out / 'contact_window_metadata.csv', index=False)
    bag_rows, metrics, traces, checks, audits, target_checks, teacher_rows, subset_metrics = ([], [], [], [], [], [], [], [])
    for split in splits:
        fold = split['fold']
        train_bags, test_bags = (split['train_bags'], split['test_bags'])
        assert not set(train_bags) & set(test_bags)
        ctr = cm.bag_id.isin(train_bags).to_numpy()
        rtr = rm.bag_id.isin(train_bags).to_numpy()
        assert set(cm.loc[ctr].bag_id) == set(rm.loc[rtr].bag_id) == set(train_bags)
        retained = np.load(split['shared_head'])
        assert set(retained['local_training_bags'].tolist()) == set(train_bags)
        w = torch.tensor(retained['weight10'], dtype=torch.float32)
        b = torch.tensor(retained['bias10'], dtype=torch.float32)
        w_before, b_before = (w.clone(), b.clone())
        targets = np.load(Path(_c2r_resolve_path(split['shared_head'])).parent / 'local_teacher_targets.npz')
        assert np.array_equal(targets['hidden_mean'], h)
        assert np.array_equal(targets['bag_id'], cm.bag_id.to_numpy(str))
        csc = StandardScaler().fit(h[ctr], sample_weight=weights(cm[ctr]))
        rsc = StandardScaler().fit(x[rtr], sample_weight=weights(rm[rtr]))
        hz = csc.transform(h).astype(np.float32)
        xt = torch.tensor(rsc.transform(x).astype(np.float32))
        ct = np.array([hz[cm.bag_id.eq(bag)].mean(0) for bag in train_bags], dtype=np.float32)
        labels = np.array([int(cm.loc[cm.bag_id.eq(bag), 'label'].iloc[0]) for bag in train_bags])
        assert sorted(labels.tolist()) == [0, 1, 2, 3]
        prototype = np.stack([ct[labels == y].mean(0) for y in labels])
        prototype_error = float(np.abs(prototype - ct).max())
        assert prototype_error == 0
        audits.append(dict(fold=fold, class_train_bag_counts={str(y): int((labels == y).sum()) for y in range(4)}, matched_vs_class_prototype_max_abs=prototype_error, class_prototype_loss_and_gradients_identical_for_every_possible_student_output=True, same_class_cross_recording_shuffle='not possible with one training recording per class', within_class_cross_recording_CKA='not defined; no result computed'))
        centered_w = w.numpy() - w.numpy().mean(0, keepdims=True)
        _, singular, vt = np.linalg.svd(centered_w * csc.scale_[None, :], full_matrices=False)
        rank = int((singular > singular.max() * 1e-06).sum())
        basis = vt[:rank].astype(np.float32)
        with torch.inference_mode():
            teacher_logits = torch.tensor(h) @ w.T + b
            teacher_p = grouping(teacher_logits).softmax(-1).numpy()
        ct_t, yi = (torch.tensor(ct), torch.tensor(labels))
        qt = []
        for bag in train_bags:
            p10 = (teacher_logits[cm.bag_id.eq(bag).to_numpy()] / 2).softmax(-1)
            qt.append(torch.stack([p10[:, g].sum(-1) for g in GROUPS], -1).mean(0))
        qt = torch.stack(qt)
        positive = torch.tensor(labels[:, None] == labels[None, :])
        pools = [np.flatnonzero(rm.bag_id.eq(bag).to_numpy()) for bag in train_bags]
        code_h, code_z, code_p, code_trace = make_label_codes(w, b, csc.mean_, csc.scale_)
        target_checks.append(dict(fold=fold, class_predictions=code_p.argmax(-1).tolist(), true_class_probabilities=code_p.diag().tolist(), code_head_accuracy=float((code_p.argmax(-1) == torch.arange(4)).float().mean()), code_nonnegative=bool(code_h.min() >= 0), construction_trace=code_trace))
        np.savez_compressed(out / 'targets' / f'{fold}.npz', train_bags=np.array(train_bags), labels=labels, matched_teacher_z=ct, class_prototype_z=prototype, label_codes_hidden=code_h.numpy(), label_codes_z=code_z.numpy(), label_codes_probs4=code_p.numpy(), contact_mean=csc.mean_, contact_scale=csc.scale_, radar_mean=rsc.mean_, radar_scale=rsc.scale_, actual_head_basis_standardized=basis, actual_head_weight=w.numpy(), actual_head_bias=b.numpy(), contact_hidden=h, contact_standardized=hz, contact_probabilities=teacher_p)
        for bag in train_bags + test_bags:
            ids = cm.bag_id.eq(bag).to_numpy()
            p = teacher_p[ids].mean(0)
            teacher_rows.append(dict(fold=fold, bag_id=bag, role='train' if bag in train_bags else 'test', label=int(cm.loc[ids, 'label'].iloc[0]), prediction=int(p.argmax()), n_windows=int(ids.sum()), window_accuracy=float((teacher_p[ids].argmax(-1) == cm.loc[ids, 'label'].to_numpy()).mean()), **{f'p{i}': float(p[i]) for i in range(4)}))
        for method in METHODS:
            target = code_z[labels] if method.startswith('label_code') else ct_t
            if method == 'wrong_class_mse':
                target = torch.stack([ct_t[np.flatnonzero(labels == (int(y) + 1) % 4)[0]] for y in labels])
            for seed in SEEDS:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = Encoder(csc.mean_, csc.scale_)
                opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                rng = np.random.default_rng(seed)
                for step in range(STEPS):
                    ids = np.concatenate([rng.choice(pool, 16, replace=True) for pool in pools])
                    hh = model(xt[ids]).reshape(len(pools), 16, 128)
                    zz = (hh - model.contact_mean) / model.contact_scale
                    mean_z = zz.mean(1)
                    log10 = hh @ w.T + b
                    log4 = grouping(log10)
                    ce = F.cross_entropy(log4.reshape(-1, 4), yi.repeat_interleave(16))
                    mse = F.mse_loss(mean_z, target)
                    sims = F.normalize(mean_z, dim=1) @ F.normalize(ct_t, dim=1).T / 0.2
                    contrast = (torch.logsumexp(sims, 1) - torch.logsumexp(sims.masked_fill(~positive, -torch.inf), 1)).mean()
                    p10 = (log10 / 2).softmax(-1)
                    p4 = torch.stack([p10[..., g].sum(-1) for g in GROUPS], -1).mean(1)
                    kd = 4 * (qt * (qt.clamp_min(1e-10).log() - p4.clamp_min(1e-10).log())).sum(1).mean()
                    if method == 'ce_only':
                        loss = ce
                    elif method == 'full':
                        loss = ce + mse + 0.1 * contrast + 0.5 * kd
                    elif method == 'label_code_ce_mse':
                        loss = ce + mse
                    else:
                        loss = mse
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    opt.step()
                    if step % 100 == 0 or step == STEPS - 1:
                        traces.append(dict(fold=fold, method=method, seed=seed, step=step + 1, loss=float(loss.detach()), ce=float(ce.detach()), mse=float(mse.detach()), contrast=float(contrast.detach()), kd=float(kd.detach())))
                model.eval()
                with torch.inference_mode():
                    hh_t = model(xt)
                    zz = ((hh_t - model.contact_mean) / model.contact_scale).numpy()
                    logits10 = (hh_t @ w.T + b).numpy()
                    prob = grouping(torch.tensor(logits10)).softmax(-1).numpy()
                    hh = hh_t.numpy()
                key = f'{fold}__{method}__seed{seed}'
                np.savez_compressed(out / 'windows' / f'{key}.npz', hidden=hh, standardized_hidden=zz, probabilities4=prob, logits10=logits10, bag_id=rm.bag_id.to_numpy(str), row_index=np.arange(len(rm)), label=rm.label.to_numpy(), role=np.array(['train' if bag in train_bags else 'test' if bag in test_bags else 'external' for bag in rm.bag_id]), geometry_valid=~rm.geometry_roi_mismatch.to_numpy(bool))
                predictions = []
                for bag in train_bags + test_bags:
                    ids = rm.bag_id.eq(bag).to_numpy()
                    ci = cm.bag_id.eq(bag).to_numpy()
                    pred = prob[ids].mean(0)
                    ez = zz[ids].mean(0)
                    tz = hz[ci].mean(0)
                    eh = hh[ids].mean(0)
                    th = h[ci].mean(0)
                    delta = ez - tz
                    visible = delta @ basis.T @ basis
                    invisible = delta - visible
                    info = rm.loc[ids].iloc[0]
                    row = dict(fold=fold, method=method, seed=seed, bag_id=bag, role='train' if bag in train_bags else 'test', label=int(info.label), prediction=int(pred.argmax()), n_windows=int(ids.sum()), geometry_valid=not bool(info.geometry_roi_mismatch), window_accuracy=float((prob[ids].argmax(-1) == int(info.label)).mean()), mse_standardized=float(np.mean(delta ** 2)), cos_standardized=cosine(ez, tz), mse_raw=float(np.mean((eh - th) ** 2)), cos_raw=cosine(eh, th), classifier_rank=rank, visible_mse_128=float(np.mean(visible ** 2)), nullspace_mse_128=float(np.mean(invisible ** 2)), centered_logit_shift_l2=float(np.linalg.norm((eh - th) @ centered_w.T)), radar_within_bag_variance=float(zz[ids].var(0).mean()), contact_within_bag_variance=float(hz[ci].var(0).mean()), wrong_target_class=(int(info.label) + 1) % 4 if method == 'wrong_class_mse' else -1, **{f'p{i}': float(pred[i]) for i in range(4)})
                    assert abs(row['mse_standardized'] - row['visible_mse_128'] - row['nullspace_mse_128']) < 2e-05
                    predictions.append(row)
                    bag_rows.append(row)
                pdf = pd.DataFrame(predictions)
                for role in ('train', 'test'):
                    part = pdf[pdf.role.eq(role)]
                    mask = rm.bag_id.isin(part.bag_id).to_numpy()
                    scores = dict(fold=fold, method=method, seed=seed, role=role, n_bags=len(part), n_windows=int(mask.sum()), bag_accuracy=accuracy_score(part.label, part.prediction), bag_macro_f1=f1_score(part.label, part.prediction, labels=[0, 1, 2, 3], average='macro', zero_division=0), window_accuracy=accuracy_score(rm.loc[mask, 'label'], prob[mask].argmax(-1)), window_macro_f1=f1_score(rm.loc[mask, 'label'], prob[mask].argmax(-1), labels=[0, 1, 2, 3], average='macro', zero_division=0))
                    for col in ('mse_standardized', 'cos_standardized', 'cos_raw', 'visible_mse_128', 'nullspace_mse_128', 'centered_logit_shift_l2', 'radar_within_bag_variance', 'contact_within_bag_variance'):
                        scores[col] = float(part[col].mean())
                    metrics.append(scores)
                    valid = part[part.geometry_valid]
                    vmask = rm.bag_id.isin(valid.bag_id).to_numpy()
                    subset_metrics.append(dict(fold=fold, method=method, seed=seed, role=role, sensitivity='exclude_known_wrong_outer_ROI; unchanged 4-class head', truth_labels='0,1,3', n_bags=len(valid), n_windows=int(vmask.sum()), bag_accuracy=accuracy_score(valid.label, valid.prediction), bag_macro_f1_true_classes=f1_score(valid.label, valid.prediction, labels=[0, 1, 3], average='macro', zero_division=0), window_accuracy=accuracy_score(rm.loc[vmask, 'label'], prob[vmask].argmax(-1)), window_macro_f1_true_classes=f1_score(rm.loc[vmask, 'label'], prob[vmask].argmax(-1), labels=[0, 1, 3], average='macro', zero_division=0)))
                checkpoint = dict(encoder=model.state_dict(), feature='frame_shape', input_dim=128, contact_mean=csc.mean_, contact_scale=csc.scale_, radar_mean=rsc.mean_, radar_scale=rsc.scale_, head_weight=w, head_bias=b, groups=GROUPS, method=method, seed=seed, steps=STEPS, train_bags=train_bags, test_bags=test_bags, source_head=split['shared_head'], geometry_status='outer-race ROI remains invalid', protocol_path=str(out / 'protocol.json'))
                path = out / 'models' / f'{key}.pt'
                torch.save(checkpoint, path)
                loaded = torch.load(path, map_location='cpu', weights_only=False)
                rebuilt = Encoder(loaded['contact_mean'], loaded['contact_scale'])
                rebuilt.load_state_dict(loaded['encoder'])
                rebuilt.eval()
                with torch.inference_mode():
                    p_replay = grouping(rebuilt(xt) @ loaded['head_weight'].T + loaded['head_bias']).softmax(-1).numpy()
                error = float(np.abs(p_replay - prob).max())
                assert error < 1e-06
                assert torch.equal(w, w_before) and torch.equal(b, b_before)
                checks.append(dict(fold=fold, method=method, seed=seed, reload_probability_max_error=error, head_unchanged=True, hidden_nonnegative=bool(hh.min() >= 0)))
                pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
                pd.DataFrame(bag_rows).to_csv(out / 'file_predictions.csv', index=False)
                pd.DataFrame(subset_metrics).to_csv(out / 'geometry_valid_sensitivity.csv', index=False)
                print(key, 'test:', metrics[-1]['bag_accuracy'], metrics[-1]['window_accuracy'], 'mse:', round(metrics[-1]['mse_standardized'], 5), 'seconds:', round(time.time() - start, 1), flush=True)
    assert input_hashes == {str(p): sha(p) for p in files}
    pd.DataFrame(traces).to_csv(out / 'training_trace.csv', index=False)
    pd.DataFrame(teacher_rows).to_csv(out / 'contact_predictions.csv', index=False)
    write_json(out / 'prototype_audit.json', audits)
    write_json(out / 'label_code_construction.json', target_checks)
    write_json(out / 'splits.json', splits)
    write_json(out / 'verification.json', dict(models=checks, source_files_unchanged=True, original_contact_encoder_and_head_untouched=True, no_fit_on_test=True, no_fit_on_external=True, standardized_error_decomposition_checked=True, test_used_for_final_reporting_only=True, no_new_physical_samples=True, model_instances=len(checks), python=sys.executable, torch=torch.__version__, cpu_threads=2, script_sha256=sha(__file__), elapsed_seconds=time.time() - start))
    summarize(out)

def summarize(out):
    frame = pd.read_csv(out / 'metrics.csv')
    test = frame[frame.role.eq('test')]
    means = test.groupby('method', sort=False).mean(numeric_only=True)
    means.to_csv(out / 'summary_mean_over_folds_seeds.csv')
    archive = pd.read_csv(ARCHIVE / 'metrics.csv')
    replay = []
    for method, old_method in [('ce_only', 'ce_only'), ('matched_bag_mse', 'embedding_only'), ('full', 'ce_feat_1_kd')]:
        left = test[test.method.eq(method)]
        right = archive[archive.method.eq(old_method) & archive.role.eq('test')]
        both = left.merge(right, on=['fold', 'seed'], suffixes=('_new', '_old'))
        replay.append(dict(method=method, n_matches=len(both), mse_max_abs_difference=float(np.abs(both.mse_standardized_new - both.mse_standardized_old).max()), cosine_max_abs_difference=float(np.abs(both.cos_standardized_new - both.cos_standardized_old).max()), bag_accuracy_max_abs_difference=float(np.abs(both.bag_accuracy - both.accuracy).max())))
    write_json(out / 'archive_replay_comparison.json', replay)
    labels = dict(ce_only='只用真实类别训练', matched_bag_mse='匹配真实接触平均特征', full='原完整蒸馏', label_code_mse='匹配人为四类代表特征', label_code_ce_mse='真实类别＋人为四类代表特征', wrong_class_mse='匹配固定错类接触特征（负对照）')
    lines = ['# 已有数据上的 encoder 知识检验：第一轮实跑', '', '这是一轮诊断实验：检查“最后分类正确”是否足以说明雷达获得了丰富的接触知识。使用此前已多次查看的 8 条录制，并未新增盲测数据。外圈雷达距离单元仍有已知错误，因此四类高分不能作为实际物理诊断的最终成绩。', '', '每轮用四条录制训练（每类一条），另四条测试，再交换方向；每种方法重复三个种子，共 36 个固定 400 步模型。所有模型具有相同雷达输入、网络、初始化、训练采样、优化器和冻结分类头，没有用测试结果选择模型或步数。', '', '## 结果', '', '表中“片段准确率”先在各方向计算再等权平均；“整条准确率”将同次录制的预测概率平均后判断。三个种子不增加真实录制数量。', '', '| 方法 | 整条准确率 | 片段准确率 | 特征差异 MSE↓ | 特征相似度↑ | 分类头看得见的差异 | 分类头看不见的差异 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in means.iterrows():
        lines.append(f'| {labels[name]} | {row.bag_accuracy:.2%} | {row.window_accuracy:.2%} | {row.mse_standardized:.4f} | {row.cos_standardized:.4f} | {row.visible_mse_128:.4f} | {row.nullspace_mse_128:.4f} |')
    lines += ['', '“人为四类代表特征”怎样生成：不给真实的各类接触特征，只给固定分类头、四个类别编号和训练接触数据的全局均值/尺度，优化四组非负的 128 维数字，使分类头能够识别。它用到了既有分类头的知识及全局数值尺度，所以并非完全不使用接触模型。构造额外使用 400 次很小的四向量优化，雷达网络仍统一训练 400 步。', '', '“错类”负对照：训练时把正常指向内圈、内圈指向外圈、外圈指向滚动体、滚动体指向正常，不加入真实类别分类损失。这用于观察学生是否跟随给定目标，不是候选部署模型。', '', '## 可以支持什么', '', '- 按原数据、分类头和设置重跑了原来的三种方法；数值重现误差见 archive_replay_comparison.json。', '- 每类在训练时只有一条录制，所以“真实录制平均特征”就是“该类别平均特征”。两者逐元素完全相同，不能当成两个独立算法的胜负对比。', '- 人为类代表特征不包含真实的各类接触平均特征。它若也能达到高分类准确率，说明在这批数据上，成功接入分类头仍不足以证明保留了丰富的接触信息；这不等于证明原蒸馏模型已经退化成四个固定向量。', '- 教师特征匹配是否保留了故障轻重、未知故障等信息，还需要同类多条独立录制和接触侧新任务标签。当前无法从四个测试袋做有意义的类内检索/类内 CKA，也没有输出这些指标。', '', '## 指标怎么读', '', 'MSE 是两组数字的平均平方差，越小越接近；余弦相似度看组合方向，越接近 1 越相似。它们不是知识百分比。', '', '当前分类头输入 128 维，去除所有类别共同的分数平移后，有效线性方向是 9 个。报告把差异分为会改变这些分数的部分，以及不会改变这些分数的部分；两项相加等于总 MSE。后一项小也不能单独证明诊断信息丰富，需要新任务。', '', '同时保存了每条录制内的特征波动，但接触窗口与雷达窗口长度不同且没有逐段配对，因此不把两种波动之比解释为时序保真度。', '', '## 文件', '', '- metrics.csv：每方向、方法、种子的训练/测试指标。', '- file_predictions.csv：每条录制预测、误差、距离单元有效标志。', '- windows/*.npz：每个雷达片段的 128 维特征、十类分数、四类概率和录制标识。', '- models/*.pt：所有 36 个模型；旧模型没有覆盖。', '- prototype_audit.json：每类单袋及目标完全相同的核验。', '- label_code_construction.json：人为代表特征在真实分类头下的类别与概率。', '- verification.json：源文件不变、模型重载、无测试拟合等检查。', '', '注意：本轮复用了 stage_b_v0 的每折 retained_head_v1；不是随后 all_known 保守头。雷达训练全程不改接触编码器或分类头，所以没有新增接触能力遗忘。旧数据回归依赖归档分类头验收记录，不能当作新盲测回归。', '']
    (out / '第一轮对照结果_通俗说明.md').write_text('\n'.join(lines))
if __name__ == '__main__':
    main()
