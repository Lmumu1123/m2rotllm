"""Read-only independent checks of saved predictions and diagnostic conclusions."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / 'results'

def main():
    metadata = pd.read_csv(ROOT / 'radar_window_metadata.csv')
    files = pd.read_csv(ROOT / 'file_predictions.csv')
    scores = pd.read_csv(ROOT / 'metrics.csv')
    geometry = pd.read_csv(ROOT / 'geometry_valid_sensitivity.csv')
    saved = sorted((ROOT / 'windows').glob('*.npz'))
    assert len(saved) == 36
    frozen_hashes = json.loads((ROOT / 'input_hashes_before.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in frozen_hashes.items())
    per_model = []
    max_bag_probability_error = 0.
    wrong_target_matched = wrong_target_total = 0
    test_rows = []
    for path in saved:
        fold, method, seed_part = path.stem.split('__')
        seed = int(seed_part.removeprefix('seed'))
        z = np.load(path)
        assert np.array_equal(z['bag_id'], metadata.bag_id.to_numpy(str))
        assert z['hidden'].shape == (len(metadata), 128)
        assert z['probabilities4'].shape == (len(metadata), 4)
        assert np.isfinite(z['hidden']).all() and np.isfinite(z['probabilities4']).all()
        assert np.abs(z['probabilities4'].sum(-1) - 1).max() < 1e-6
        assert z['hidden'].min() >= 0
        part = files[(files.fold == fold) & (files.method == method) & (files.seed == seed)]
        assert len(part) == 8
        for _, row in part.iterrows():
            ids = z['bag_id'] == row.bag_id
            p = z['probabilities4'][ids].mean(0)
            error = np.abs(p - np.array([row[f'p{i}'] for i in range(4)])).max()
            max_bag_probability_error = max(max_bag_probability_error, float(error))
            assert int(p.argmax()) == row.prediction
        hold = z['role'] == 'test'
        true = z['label'][hold]
        pred = z['probabilities4'][hold].argmax(-1)
        if method == 'wrong_class_mse':
            wrong_target_matched += int((pred == (true + 1) % 4).sum())
            wrong_target_total += len(true)
        test_rows.extend(dict(method=method, seed=seed, fold=fold, label=int(t), prediction=int(p)) for t, p in zip(true, pred))
        per_model.append(dict(path=path.name, total_saved_windows=len(metadata),
                              train_windows=int((z['role'] == 'train').sum()),
                              test_windows=int(hold.sum()),
                              all_test_windows_correct=bool((true == pred).all())))
    assert max_bag_probability_error < 1e-7
    all_test = pd.DataFrame(test_rows)
    pooled = all_test.assign(correct=all_test.label == all_test.prediction).groupby('method').agg(
        correct_seed_window_predictions=('correct', 'sum'), total_seed_window_predictions=('correct', 'size'),
        pooled_window_accuracy=('correct', 'mean'))
    pooled.to_csv(ROOT / 'pooled_window_summary.csv')
    sensitive = geometry[geometry.role.eq('test')].groupby('method', sort=False).mean(numeric_only=True)
    sensitive.to_csv(ROOT / 'geometry_valid_sensitivity_summary.csv')
    report = dict(source_artifacts_still_unchanged=True, model_count=len(saved),
                  saved_window_probability_and_metadata_checks_passed=True,
                  bag_aggregation_max_probability_difference=max_bag_probability_error,
                  wrong_class_target_following=dict(correct=wrong_target_matched, total=wrong_target_total,
                                                    fraction=wrong_target_matched / wrong_target_total),
                  independent_recordings=8, independent_bearings_not_established=True,
                  parent_archive_exact_replay=json.loads((ROOT / 'archive_replay_comparison.json').read_text()),
                  models=per_model)
    (ROOT / 'independent_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    output = ROOT / '第一轮对照结果_通俗说明.md'
    text = output.read_text()
    block = ['## 本轮新结论及距离问题敏感性检查', '',
             '结果已经实际跑出：人为四类代表特征在全部六次留出评估中，整条录制判断都正确；片段准确率按两个方向和三个种子平均为 99.53%。它与真实接触平均特征的相似度只有 0.236，真实教师匹配约为 0.723。', '',
             '**因此，在这些录制上，四分类高分确实可以由不包含真实各类接触平均特征的代码实现。原方法让平均特征更接近教师是真的，但“获得了更多可复用诊断信息”还没有被证明。** 不能把此结果写成“已证明原 encoder 塌缩”。', '',
             f'固定错类负对照对真实类别的整段和片段准确率都是 0%，但 {wrong_target_matched}/{wrong_target_total} 个留出片段预测（含三个种子的重复，即 {wrong_target_matched / wrong_target_total:.2%}）跟随指定的错误目标类别。这说明学生能够学习所给的目标对应关系；它没有验证目标是否代表真实故障知识。', '',
             '排除明确选错距离单元的外圈录制后，只统计正常、内圈、滚动体真值；模型仍输出原来四类，误报成外圈仍计错。这是敏感性检查，训练仍包含原四类，没有重训练，也不能称为完全清洁的新测试。', '',
             '| 方法 | 剩余整条准确率 | 剩余片段准确率 | 三种真实类别 macro-F1（片段） |',
             '|---|---:|---:|---:|']
    names = dict(ce_only='只用类别', matched_bag_mse='真实接触平均特征', full='完整蒸馏',
                 label_code_mse='人为四类代表特征', label_code_ce_mse='类别＋人为代表特征',
                 wrong_class_mse='错类负对照')
    for method, row in sensitive.iterrows():
        block.append(f'| {names[method]} | {row.bag_accuracy:.2%} | {row.window_accuracy:.2%} | {row.window_macro_f1_true_classes:.2%} |')
    block += ['', '每个方向有三条有效距离候选录制，总计六条；同一轴承的不同录制仍不是跨轴承或跨电机测试。', '',
              '完整蒸馏的总体片段均值为 99.69%，简单教师均值匹配为 99.14%；这次是补算旧结果的片段统计，并不是新发现的独立泛化提升。由于物理录制只有八条，暂不宣称约 0.55 个百分点差异具有统计意义。', '',
              '下一步最有价值的是同类独立录制与新诊断任务，或验证不参与雷达训练的另一个接触模型能否读取这些特征。单纯在这八条录制上反复提高四分类分数无法解决知识是否丰富的问题。', '']
    output.write_text(text.split('## 本轮新结论及距离问题敏感性检查')[0] + '\n' + '\n'.join(block))
    print(json.dumps({k:v for k,v in report.items() if k not in ('models', 'parent_archive_exact_replay')}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
