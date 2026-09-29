"""Independent recomputation of contact-only Bear->Rot stitching. No import of tested script."""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
from pathlib import Path
import hashlib, json, sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
ROOT = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/encoder_validation_20260923'))
OUT = ROOT / 'results/contact_only_stitch'
BEAR = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/retention_alignment_20260922/preprocessing/variants/v0/retrained_fcn'))
ROT = ROOT / 'rotllm'
CTRL = ROOT / 'controls/results'
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]

def sha(path):
    return hashlib.sha256(Path(_c2r_resolve_path(path)).read_bytes()).hexdigest()

def sm(logit):
    exp = np.exp(logit - logit.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def fit_map(a, b, metadata, train):
    md = metadata[train]
    n = md.groupby('bag_id').size()
    weights = np.array([1 / n[k] for k in md.bag_id]) * len(md) / len(n)
    sa = StandardScaler().fit(a[train], sample_weight=weights)
    sb = StandardScaler().fit(b[train], sample_weight=weights)
    model = Ridge(alpha=1.0, fit_intercept=True, solver='svd').fit(sa.transform(a[train]), sb.transform(b[train]), sample_weight=weights)
    w = sb.scale_[:, None] * model.coef_ / sa.scale_[None, :]
    intercept = sb.mean_ + sb.scale_ * model.intercept_ - w @ sa.mean_
    return (w, intercept, sa, sb)

def main():
    recorded = json.loads((OUT / 'verification.json').read_text())
    assert all((sha(p) == h for p, h in recorded['inputs'].items()))
    cm = pd.read_csv(BEAR / 'metadata.csv')
    bm = pd.read_csv(ROT / 'metadata.csv')
    rm = pd.read_csv(CTRL / 'radar_window_metadata.csv')
    a = np.load(BEAR / 'contact_features.npz')['hidden_mean'].astype(np.float64)
    bz = np.load(ROT / 'teacher_features.npz')
    bi = {(r.bag_id, r.window_row): i for i, r in enumerate(bm.itertuples())}
    assert len(bi) == len(bm)
    positions = [bi[r.bag_id, r.window_row] for r in cm.itertuples()]
    assert len(set(positions)) == len(cm)
    matched_md = bm.iloc[positions].reset_index(drop=True)
    for field in ('file', 'window_row', 'label', 'state', 'baud', 'bag_id', 'sample_rate_hz'):
        assert np.array_equal(cm[field], matched_md[field]), field
    b32 = bz['hidden_mean'][positions]
    b = b32.astype(np.float64)
    assert np.array_equal(bz['bag_id'][positions], cm.bag_id.to_numpy(str))
    assert np.array_equal(bz['window_rows'][positions], cm.window_row.to_numpy())
    with sqlite3.connect(f'file:{ROT}/vendor/RotLLM/datasets/vibration_metadata.sqlite?mode=ro', uri=True) as con:
        notes = dict(con.execute('select label,note from label_note'))
    assert notes[0] == 'Normal State'
    for group, phrase in zip(GROUPS[1:], ('Inner Ring', 'Outer Ring', 'Ball')):
        assert all((phrase in notes[k] for k in group))
    assert all(('Gear' in notes[k] for k in range(10, 15)))
    splits = json.loads((CTRL / 'splits.json').read_text())
    saved_metrics = pd.read_csv(OUT / 'metrics.csv')
    saved_bags = pd.read_csv(OUT / 'file_predictions.csv')
    native = np.load(ROT / 'native_head15.npz')
    folds, replays = ([], [])
    maximum_metric_error = maximum_probability_error = maximum_gear_error = 0.0
    invalid_bags = set(rm.loc[rm.geometry_roi_mismatch, 'bag_id'])
    for split in splits:
        fold = split['fold']
        train = cm.bag_id.isin(split['train_bags']).to_numpy()
        test = cm.bag_id.isin(split['test_bags']).to_numpy()
        assert not (train & test).any()
        assert train.sum() + test.sum() == 58
        assert set(cm.loc[train, 'bag_id']) == set(split['train_bags'])
        w, intercept, sa, sb = fit_map(a, b, cm, train)
        saved_map = np.load(OUT / 'mappings' / f'{fold}.npz')
        map_weight_err = float(np.abs(w - saved_map['weight']).max())
        map_bias_err = float(np.abs(intercept - saved_map['bias']).max())
        assert map_weight_err < 1e-08 and map_bias_err < 1e-08
        aa, bb, mm = (a.copy(), b.copy(), cm.copy())
        aa[~train] = 100000000.0
        bb[~train] = -100000000.0
        mm.loc[~train, 'label'] = 999
        pw, pi, _, _ = fit_map(aa, bb, mm, train)
        assert np.array_equal(pw, w) and np.array_equal(pi, intercept)
        scaler = StandardScaler().fit(b32[train])
        lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=42)
        lr.fit(scaler.transform(b32[train]), cm.loc[train, 'label'])
        head = np.load(ROT / 'heads' / fold / 'head.npz')
        assert head['classes'].tolist() == lr.classes_.tolist() == [0, 1, 2, 3]
        assert set(head['train_bag_ids']) == set(split['train_bags'])
        assert set(head['test_bag_ids']) == set(split['test_bags'])
        hw = lr.coef_ / scaler.scale_[None, :]
        hb = lr.intercept_ - hw @ scaler.mean_
        head_w_err = float(np.abs(hw - head['weight4']).max())
        head_b_err = float(np.abs(hb - head['bias4']).max())
        assert head_w_err < 1e-08 and head_b_err < 1e-08
        double_transform_error = float(np.abs(sm(scaler.transform(b32).astype(float) @ head['weight4'].T + head['bias4']) - sm(b @ head['weight4'].T + head['bias4'])).max())
        folds.append(dict(fold=fold, contact_train_windows=int(train.sum()), contact_test_windows=int(test.sum()), train_recordings=len(split['train_bags']), test_recordings=len(split['test_bags']), mapping_weight_max_abs=map_weight_err, mapping_bias_max_abs=map_bias_err, heldout_feature_and_label_poisoning_leaves_mapping_bitwise_identical=True, independent_four_class_head_weight_max_abs=head_w_err, independent_four_class_head_bias_max_abs=head_b_err, scaler_already_folded_into_new4=True, hypothetical_double_scaling_probability_error=double_transform_error))
        paths = [('native_B_contact', 'none', -1, cm, b), ('A_contact_to_B', 'none', -1, cm, np.maximum(a @ w.T + intercept, 0))]
        for method in ('ce_only', 'matched_bag_mse', 'full', 'label_code_mse', 'label_code_ce_mse', 'wrong_class_mse'):
            for seed in (17, 42, 73):
                z = np.load(CTRL / 'windows' / f'{fold}__{method}__seed{seed}.npz')
                assert np.array_equal(z['bag_id'], rm.bag_id.to_numpy(str))
                assert np.array_equal(z['role'] == 'test', rm.bag_id.isin(split['test_bags']).to_numpy())
                mapped = np.maximum(z['hidden'].astype(float) @ w.T + intercept, 0)
                paths.append(('frozen_radar_A_to_B', method, seed, rm, mapped))
        for pathname, method, seed, md, hid in paths:
            take = md.bag_id.isin(split['test_bags']).to_numpy()
            validmd, validhid = (md[take].reset_index(drop=True), hid[take])
            for headname, hw, hb in [('contact_only_new4', head['weight4'], head['bias4']), ('native15_conditioned', native['weight15'], native['bias15'])]:
                probabilities = sm(validhid @ hw.T + hb)
                if headname == 'native15_conditioned':
                    gear = probabilities[:, 10:].sum(1)
                    other = (probabilities.argmax(1) >= 10).astype(float)
                    four = np.stack([probabilities[:, group].sum(1) for group in GROUPS], axis=1)
                    probabilities = four / four.sum(1, keepdims=True)
                else:
                    gear = np.zeros(len(validhid))
                    other = np.zeros(len(validhid))
                prediction = probabilities.argmax(1)
                for scope in ('all_four_development', 'known_roi_valid_subset'):
                    good = np.ones(len(validmd), dtype=bool) if scope == 'all_four_development' else ~validmd.bag_id.isin(invalid_bags).to_numpy()
                    classes = [0, 1, 2, 3] if scope == 'all_four_development' else [0, 1, 3]
                    keep = (saved_metrics.fold == fold) & (saved_metrics.path == pathname) & (saved_metrics.radar_method == method) & (saved_metrics.seed == seed) & (saved_metrics['head'] == headname) & (saved_metrics.scope == scope)
                    entry = saved_metrics[keep & saved_metrics.unit.eq('window')].iloc[0]
                    truth = validmd.label.to_numpy()
                    acc = accuracy_score(truth[good], prediction[good])
                    f1 = f1_score(truth[good], prediction[good], labels=classes, average='macro', zero_division=0)
                    maximum_metric_error = max(maximum_metric_error, abs(acc - entry.accuracy), abs(f1 - entry.macro_f1))
                    maximum_gear_error = max(maximum_gear_error, abs(gear[good].mean() - entry.mean_gear_mass), abs(other[good].mean() - entry.pred15_other_rate))
                    assert int(entry.n) == int(good.sum())
                    bagtruth, bagprediction = ([], [])
                    for bag in validmd.loc[good, 'bag_id'].unique():
                        ids = validmd.bag_id.eq(bag).to_numpy()
                        p = probabilities[ids].mean(0)
                        actual = saved_bags[(saved_bags.fold == fold) & (saved_bags.path == pathname) & (saved_bags.radar_method == method) & (saved_bags.seed == seed) & (saved_bags['head'] == headname) & (saved_bags.scope == scope) & (saved_bags.bag_id == bag)].iloc[0]
                        maximum_probability_error = max(maximum_probability_error, float(np.abs(p - actual[[f'p{i}' for i in range(4)]].to_numpy(float)).max()))
                        assert p.argmax() == actual.prediction
                        bagtruth.append(int(validmd.loc[ids, 'label'].iloc[0]))
                        bagprediction.append(int(p.argmax()))
                    row = saved_metrics[keep & saved_metrics.unit.eq('recording')].iloc[0]
                    maximum_metric_error = max(maximum_metric_error, abs(accuracy_score(bagtruth, bagprediction) - row.accuracy), abs(f1_score(bagtruth, bagprediction, labels=classes, average='macro', zero_division=0) - row.macro_f1))
                    if pathname == 'frozen_radar_A_to_B' and method in ('full', 'label_code_mse') and (headname == 'contact_only_new4'):
                        replays.append(dict(fold=fold, method=method, seed=seed, scope=scope, window_accuracy=acc, recording_accuracy=accuracy_score(bagtruth, bagprediction)))
    assert maximum_metric_error < 1e-12
    assert maximum_probability_error < 1e-08
    assert maximum_gear_error < 1e-10
    assert all((sha(p) == h for p, h in recorded['inputs'].items()))
    result = dict(status='passed_with_interpretation_limits', tested_script_sha256=sha(ROOT / 'scripts/contact_only_stitch.py'), audit_script_sha256=sha(__file__), source_files_unchanged=True, independently_recomputed_folds=folds, contact_metadata_join_exact=True, shared_raw_contact_file_and_window_rows=True, frozen_radar_rows_and_test_roles_exact=True, native15_label_groups_verified_from_official_sqlite=True, new4_head_not_double_standardized=True, max_metric_difference=maximum_metric_error, max_bag_probability_difference=maximum_probability_error, max_native_gear_diagnostic_difference=maximum_gear_error, full_vs_label_code_independent_replay=replays, audited_method_selection='All six methods, three seeds, two folds; alpha=1 and raw_rms001 fixed in script. No result-dependent selector found.', fitting_limits='Source code review and reconstruction show contact training rows only for map; four-class readout uses training contact labels. This cannot certify absence of all prior human development choices.', caveats=['same four diagnosis labels in both models; not a new-task transfer test', 'each training class has one recording; class-prototype transfer can explain benefit', 'the 25pp recording advantage of full over label-code disappears on exclusion of invalid outer ROI', 'native15_conditioned excludes gear classes and is not full fifteen-class accuracy', 'native fifteen-way head remains poor; useful results require separately trained contact four-way readout', 'mean hidden then softmax differs from mean axis softmax; the stitching script consistently uses the former', 'cross-recording splits share physical bearings, not blind cross-bearing/general-purpose proof', 'summaries average fold/seed metrics, not pooled windows'])
    (OUT / 'independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    text = f'# 接触数据校准 Bear→Rot 映射：独立审查\n\n结论：未发现本次拼接实现中的训练/测试混用、四类头重复标准化或录制汇总计算错误。独立重做了两折映射、两折接触四类分类头，并复算全部六方法、三种子、两种目标头和两种统计范围。\n\n- 指标最大差异：{maximum_metric_error:.3g}；录制四类概率最大差异：{maximum_probability_error:.3g}。\n- Bear 与 Rot 特征按录制 ID 和窗口序号对应；进一步检查原接触文件名、状态、标签、波特率、采样率，全部一致。\n- 两个 StandardScaler 与 Ridge 只使用训练接触窗口。将全部留出/外部特征改成极端数值、标签改为 999 后，独立重拟合映射逐位不变。没有雷达输入进入映射拟合。\n- 新四类头确实使用了训练接触标签；其 scaler 已折叠到 weight4/bias4，推理直接输入 Rot 隐藏特征是正确的。独立按原设置重训得到相同头。\n- 原生 15 类映射与官方 SQLite 标签一致：0 正常，1–3 内圈，7–9 外圈，4–6 滚动体；10–14 为齿轮类。此处四类概率条件化到轴承集合，不能称为完整 15 类准确率；代码另外保留了齿轮概率和原生越界预测比例。\n- 接触先平均三轴隐藏特征再过分类头，雷达对各窗口过头；录制结果均平均窗口概率。没有混用“平均隐藏特征后一次预测”和“平均窗口预测”。这条接触路径与另一实验的“每轴概率平均”不能直接混写。\n- 静态代码未发现测试标签或雷达信号进入映射，未发现依结果选 alpha、方法或种子；原始录制此前已被多轮查看，因此仍是开发实验，不是事后能证明的独立盲测。\n\n需要在论文和汇报中保留的解释边界：\n\n1. 原完整蒸馏接新四类头整段 100%，人为代码为 75%；但剔除明确错误的外圈 ROI 后，两者整段都为 100%。有效候选距离范围内的剩余窗口差异约为 99.69% 对 97.22%，不能突出 25 个百分点而隐去外圈问题。\n2. 这里的新模型仍判断相同四个类别，且映射用每类一条训练录制；结果可以由“真实的类别原型更适合另一个接触模型”解释，尚不能证明故障严重度、未知类别等类别之外知识被保留。\n3. Rot 原生 15 类头的轴承条件分类仍低；高分来自另外训练的接触四类读出，不能写成冻结原始 Rot 全模型直接达到高准确率。\n4. full 的雷达结果高于接触窗口结果不矛盾：袋均值训练和雷达窗口可能抑制接触单窗波动。这不构成雷达超过接触诊断信息量的证明。\n5. summary.csv 是对方向和种子的指标等权平均，不等于把所有窗口混在一起的总正确率。统计单位应明确；种子不是新增独立采集。\n\n没有修改主脚本、训练参数或原结果。详细复算记录见 independent_verification.json，复核脚本位于 controls/audit_stitch.py。\n'
    (OUT / 'audit.md').write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k not in ('full_vs_label_code_independent_replay', 'caveats')}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
