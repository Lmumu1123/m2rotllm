"""Trainable radar features and strictly recording-held-out exploratory baselines.

All writes remain in this script's directory. Existing exported inputs are read-only.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parent
SOURCE = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results'))
DISTANCE_M = 0.45
EDGES = np.unique(np.r_[5, np.arange(6, 102, 2), np.arange(105, 405, 5), np.arange(420, 801, 20)])
CLASSES = ['normal', 'inBroken', 'outBroken', 'roll']

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def markdown_table(frame):

    def cell(x):
        return f'{x:.5g}' if isinstance(x, (float, np.floating)) else str(x)
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend(('| ' + ' | '.join((cell(x) for x in row)) + ' |' for row in frame.itertuples(index=False, name=None)))
    return '\n'.join(lines)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def pool(f, p):
    """Average POWER, not logarithms; preserve 128 bands used in the prior audit."""
    return np.stack([np.mean(p[:, (f >= lo) & (f < hi)], axis=1) for lo, hi in zip(EDGES[:-1], EDGES[1:])], axis=1)

def shape(p):
    x = np.log10(np.maximum(p, 1e-14))
    return (x, x - x.mean(axis=1, keepdims=True))

def phase_frame_psd(phase, mask, f):
    """Per-frame PSD, median RX/bin power after averaging valid complete frames.

    phase is within-frame adjacent phase rate; never differentiates over gaps.
    0.5 Hz evaluation grid is zero padding, not 0.5 Hz physical resolution.
    """
    nwin, nframe, ntime, ncell = phase.shape
    taper = np.hanning(ntime)
    k = np.rint(f * 2).astype(int)
    out = []
    for i in range(nwin):
        complete = mask[i].all(axis=1)
        if not complete.any():
            raise ValueError('No complete phase frames')
        x = phase[i, complete].astype(np.float64)
        x -= x.mean(axis=1, keepdims=True)
        sp = np.fft.rfft(x * taper[None, :, None], n=4000, axis=1)
        out.append(np.median(np.mean(2 * abs(sp[:, k]) ** 2, axis=0), axis=1) / (2000 * np.sum(taper * taper)))
    return np.asarray(out)

def temporal_stats(iq, valid):
    """Dimensionless per-cell tail ratios and normalized autocorrelations.

    No across-frame products. Twelve spatial cells are correlated, so aggregate
    with a median and do not present them as 12 independent antennas.
    """
    z = iq[..., 0] + 1j * iq[..., 1]
    features = []
    for k in range(len(z)):
        x = z[k]
        m = valid[k]
        ax = abs(x[m])
        rms = np.sqrt(np.mean(ax ** 2, axis=0))
        rms = np.maximum(rms, 1e-10)
        row = [np.median(np.quantile(ax, 0.95, axis=0) / rms), np.median(np.quantile(ax, 0.99, axis=0) / rms), np.median(np.mean(ax ** 4, axis=0) / rms ** 4)]
        for lag in [1, 2, 4, 8, 16, 32, 64]:
            use = m[:, lag:] & m[:, :-lag]
            a = x[:, lag:][use]
            b = x[:, :-lag][use]
            denom = np.sqrt(np.mean(abs(a) ** 2, axis=0) * np.mean(abs(b) ** 2, axis=0))
            corr = np.mean(a * b.conj(), axis=0) / np.maximum(denom, 1e-12)
            row.append(np.median(corr.real))
        features.append(row)
    return np.asarray(features)

def extract():
    arrays = {k: [] for k in ['frame_shape', 'frame_abs', 'phase_shape', 'phase_abs', 'coherent_shape', 'time_stats', 'confound_raw_amp', 'confound_distance']}
    rows = []
    records = []
    manifest = []
    frame_curves = []
    coherent_curves = []
    phase_curves = []
    bags = json.loads((SOURCE / 'recording_bags.json').read_text())
    bags = {b['radar']: b for b in bags}
    for filename, b in sorted(bags.items()):
        p = SOURCE / filename
        j = p.with_suffix('.json')
        before = (p.stat().st_size, p.stat().st_mtime_ns)
        doc = json.loads(j.read_text())
        q = doc['quality']
        meta = doc['metadata']
        with np.load(p, allow_pickle=False) as z:
            f = z['frequency_hz']
            n = len(z['labels'])
            fp = 10.0 ** z['frame_log_power'].astype(np.float64)
            cp = z['coherent_power'].astype(np.float64)
            pp = phase_frame_psd(z['phase_rate_frames'], z['valid_phase_mask'], f)
            fa, fs = shape(pool(f, fp))
            pa, ps = shape(pool(f, pp))
            _, cs = shape(pool(f, cp))
            arrays['frame_shape'].append(fs)
            arrays['frame_abs'].append(fa)
            arrays['phase_shape'].append(ps)
            arrays['phase_abs'].append(pa)
            arrays['coherent_shape'].append(cs)
            arrays['time_stats'].append(temporal_stats(z['iq_frames'], z['valid_chirp_mask']))
            arrays['confound_raw_amp'].append(np.log10(np.maximum(z['raw_cell_power'], 1e-14)))
            arrays['confound_distance'].append(np.full((n, 1), q['range_center_m']))
            complete_fraction = float(z['valid_chirp_mask'].mean())
            spectral_use = (f >= 10) & (f <= 100)
            frame_peak = f[spectral_use][np.argmax(fp[:, spectral_use], axis=1)]
            coherent_peak = f[spectral_use][np.argmax(cp[:, spectral_use], axis=1)]
            for local, w in enumerate(doc['windows']):
                rows.append(dict(row=len(rows), recording_row=local, bag_id=b['candidate_bag_id'], recording_id=meta['recording_id'], source_npz=filename, contact_candidate=b['contact'], label=meta['label'], state=meta['state'], baud_raw=meta['baud_raw'], baud_candidate=meta['baud_candidate'], window_start_s=w['window_start_s'], duration_s=w['duration_s'], first_frame=w['first_frame'], range_center_m=q['range_center_m'], clip_fraction=q['clipped_fraction'], valid_fraction_grid=w['valid_fraction_grid'], frame_peak_10_100_hz=frame_peak[local], coherent_peak_10_100_hz=coherent_peak[local], intended_distance_m=DISTANCE_M, geometry_roi_mismatch=abs(q['range_center_m'] - DISTANCE_M) > 0.16, class_role='four_class' if meta['label'] >= 0 else 'external_healthy' if meta['state'] == 'bigNormal' else 'external_unknown_fault', condition_confirmed=False, independent_run_confirmed=False, synchronized_window_pair=False))
        records.append(dict(recording_id=meta['recording_id'], state=meta['state'], label=meta['label'], baud_candidate=meta['baud_candidate'], n_windows=n, range_center_m=q['range_center_m'], clip_fraction=q['clipped_fraction'], trailing_bytes=q['trailing_bytes'], nominal_duration_s=q['nominal_duration_s'], complete_chirp_fraction=complete_fraction, median_frame_peak_10_100_hz=float(np.median(frame_peak)), median_coherent_peak_10_100_hz=float(np.median(coherent_peak)), geometry_roi_mismatch=abs(q['range_center_m'] - DISTANCE_M) > 0.16, range_status=q['range_status'], synchronized_window_pairs=False))
        frame_curves.append(np.median(fp, axis=0))
        coherent_curves.append(np.median(cp, axis=0))
        phase_curves.append(np.median(pp, axis=0))
        if before != (p.stat().st_size, p.stat().st_mtime_ns):
            raise RuntimeError('Input changed')
        manifest.append(dict(path=str(p), bytes=before[0], mtime_ns=before[1], sha256=digest(p), sidecar_sha256=digest(j)))
        print(f"{filename}: {n} windows, range {q['range_center_m']:.3f} m", flush=True)
    data = {k: np.concatenate(v).astype(np.float32) for k, v in arrays.items()}
    data['shape_phase'] = np.concatenate([data['frame_shape'], data['phase_shape']], axis=1)
    metadata = pd.DataFrame(rows)
    rec = pd.DataFrame(records)
    data.update(labels=metadata.label.to_numpy(np.int64), bag_id=metadata.bag_id.to_numpy(str), recording_row=metadata.recording_row.to_numpy(np.int64), baud_candidate=metadata.baud_candidate.to_numpy(np.int64), band_edges_hz=EDGES)
    assert data['frame_shape'].shape[1] == 128
    assert len(np.unique(np.stack([data['bag_id'], data['recording_row'].astype(str)], axis=1), axis=0)) == len(metadata)
    assert all((np.isfinite(v).all() for k, v in data.items() if v.dtype.kind in 'fiu'))
    np.savez_compressed(ROOT / 'features.npz', **data)
    np.savez_compressed(ROOT / 'recording_spectra.npz', frequency_hz=f, frame_power=np.asarray(frame_curves), coherent_power=np.asarray(coherent_curves), phase_frame_power=np.asarray(phase_curves), recording_id=rec.recording_id.to_numpy(str))
    metadata.to_csv(ROOT / 'metadata.csv', index=False)
    rec.to_csv(ROOT / 'recording_quality.csv', index=False)
    write_json(ROOT / 'input_manifest.json', manifest)
    write_json(ROOT / 'feature_schema.json', dict(main_feature='frame_shape', optional_feature='shape_phase', row_mapping='metadata.csv row order', array_shapes={k: list(v.shape) for k, v in data.items()}, band_edges_hz=EDGES.tolist(), frequency_range_hz=[5, 800], band_pooling='mean power over [left,right), then log10 and per-window mean removal; 800 endpoint excluded', frame_resolution_hz=2000 / 192, phase_frame_resolution_hz=2000 / 191, interpolation_grid_hz=0.5, coherent_branch='conditional on cross-frame coherence; not main student input', abs_caveat='frame_abs still comes from amplitude-normalized IQ; it is not physical displacement/acceleration power', time_stat_columns=['amplitude_q95_over_rms', 'amplitude_q99_over_rms', 'amplitude_fourth_over_rms4'] + [f'normalized_complex_autocorr_real_lag_{lag}' for lag in [1, 2, 4, 8, 16, 32, 64]], confound_policy='range and raw amplitude are diagnostic baselines, excluded from frame_shape and shape_phase', raw_adc_available_in_input_directory=False, split_policy='recording-level held out by serial-baud group; baud is not an environment or sampling-rate change', labels={c: i for i, c in enumerate(CLASSES)}, pairing='user confirmed simultaneous collection; no packet timebase or synchronized window pairing'))
    return (data, metadata, rec)

def baselines(data, metadata):
    methods = ['confound_distance', 'confound_raw_amp', 'frame_abs', 'frame_shape', 'phase_shape', 'shape_phase', 'coherent_shape', 'time_stats']
    scores = []
    pred = []
    for source, target in [(115200, 460800), (460800, 115200)]:
        tr = ((metadata.baud_candidate == source) & (metadata.label >= 0)).to_numpy()
        te = ((metadata.baud_candidate == target) & (metadata.label >= 0)).to_numpy()
        y = data['labels']
        tm = metadata.loc[te]
        weights = np.zeros(tr.sum())
        trm = metadata.loc[tr]
        for bag in trm.bag_id.unique():
            which = (trm.bag_id == bag).to_numpy()
            weights[which] = 1 / which.sum()
        weights *= len(weights) / weights.sum()
        for method in methods:
            scaler = StandardScaler().fit(data[method][tr], sample_weight=weights)
            model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=3000, random_state=42)
            model.fit(scaler.transform(data[method][tr]), y[tr], sample_weight=weights)
            prob = model.predict_proba(scaler.transform(data[method][te]))
            file_y = []
            file_p = []
            for bag in tm.bag_id.unique():
                m = (tm.bag_id == bag).to_numpy()
                truth = int(tm.loc[m, 'label'].iloc[0])
                pp = prob[m].mean(0)
                estimate = int(model.classes_[pp.argmax()])
                file_y.append(truth)
                file_p.append(estimate)
                pred.append(dict(source_baud=source, target_baud=target, method=method, bag_id=bag, y_true=truth, y_pred=estimate, **{f'p_{c}': float(pp[i]) for i, c in enumerate(CLASSES)}))
            scores.append(dict(source_baud=source, target_baud=target, method=method, n_train_recordings=4, n_test_recordings=4, n_train_windows=int(tr.sum()), n_test_windows=int(te.sum()), file_accuracy=accuracy_score(file_y, file_p), file_macro_f1=f1_score(file_y, file_p, labels=range(4), average='macro', zero_division=0), window_accuracy_secondary=accuracy_score(y[te], model.predict(scaler.transform(data[method][te])))))
    metrics = pd.DataFrame(scores)
    metrics.to_csv(ROOT / 'baseline_metrics.csv', index=False)
    pd.DataFrame(pred).to_csv(ROOT / 'baseline_predictions.csv', index=False)
    write_json(ROOT / 'baseline_protocol.json', dict(C=1.0, C_selected_on_target=False, model='source-standardized logistic regression', train_sample_weight='equal recording total weight', decision='mean window probability per recording', excluded_from_training='all target windows', validation='no independent validation recordings; fixed C and methods, exploratory only', caveat='baud_candidate is serial-baud group, not environment; external records label -1 excluded'))
    return metrics

def figures(rec, metrics):
    d = np.load(ROOT / 'recording_spectra.npz', allow_pickle=False)
    f = d['frequency_hz']
    colors = ['#277DA1', '#43AA8B', '#F8961E', '#9B5DE5']
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for b, marker in [(115200, 'o'), (460800, 's')]:
        rr = rec[(rec.baud_candidate == b) & (rec.label >= 0)]
        ax[0].scatter(rr.label + (0.05 if b == 460800 else -0.05), rr.range_center_m, label=str(b), marker=marker, s=65)
        ax[1].scatter(rr.label + (0.05 if b == 460800 else -0.05), rr.clip_fraction * 100, marker=marker, s=65)
        ax[2].scatter(rr.label + (0.05 if b == 460800 else -0.05), rr.median_coherent_peak_10_100_hz, marker=marker, s=65)
    for a in ax:
        a.set_xticks(range(4), CLASSES, rotation=20)
        a.grid(alpha=0.2)
    ax[0].set_ylabel('Selected range center (m)')
    ax[0].legend(title='Filename group')
    ax[1].set_ylabel('Near-full-scale int16 values (%)')
    ax[2].set_ylabel('Coherent 10–100 Hz peak (Hz)')
    ax[2].set_title('Conditional diagnostic, not verified RPM')
    fig.tight_layout()
    savefig(fig, '01_quality_confound')
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for r, b in enumerate([115200, 460800]):
        for i, record in rec.iterrows():
            if record.baud_candidate != b or record.label < 0:
                continue
            for j, key in enumerate(['frame_power', 'phase_frame_power']):
                yy = np.log10(np.maximum(d[key][i], 1e-14))
                yy -= yy.mean()
                ax[r, j].plot(f, yy, label=record.state, c=colors[int(record.label)], linewidth=1)
        ax[r, 0].set_ylabel(f'{b}\nCentered log10 power')
        for a in ax[r]:
            a.grid(alpha=0.2)
            a.set_xlim(5, 400)
    ax[0, 0].set_title('Within-frame IQ spectrum (resolution ≈10.4 Hz)')
    ax[0, 1].set_title('Within-frame phase-rate spectrum')
    for a in ax[-1]:
        a.set_xlabel('Frequency (Hz)')
    ax[0, 0].legend(ncol=2)
    fig.tight_layout()
    savefig(fig, '02_recording_spectra')
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, (s, t) in enumerate([(115200, 460800), (460800, 115200)]):
        m = metrics[(metrics.source_baud == s) & (metrics.target_baud == t)]
        ax.bar(np.arange(len(m)) + (i - 0.5) * 0.36, m.file_macro_f1, width=0.36, label=f'{s} → {t}')
    ax.set_xticks(np.arange(len(m)), m.method, rotation=25, ha='right')
    ax.set_ylim(0, 1.06)
    ax.set_ylabel('Recording macro-F1')
    ax.grid(axis='y', alpha=0.2)
    ax.legend()
    ax.set_title('Exploratory: only four held-out recordings per direction')
    fig.tight_layout()
    savefig(fig, '03_recording_baselines')

def savefig(fig, name):
    (ROOT / 'figures').mkdir(exist_ok=True)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(ROOT / 'figures' / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

def report(data, meta, rec, metrics):
    geometry_supplied = bool((rec.range_status == 'geometry_supplied').all())
    history = '本次从原始 bin 使用统一物理 ROI 重新导出；历史宽 ROI 版本外圈与 keep 曾选择约 0.884 m。统一 ROI 只能排除该显著选门问题，仍须核对实际距离峰、波形质量与反射来源。' if geometry_supplied else '外圈与 keep 历史导出约 0.884 m，不符合用户确认的四十多厘米目标距离；当前结果仅验证软件链路和混杂风险。'
    range_mismatch = int(rec.geometry_roi_mismatch.sum())
    four = rec[rec.label >= 0]
    clip_min = float(four.clip_fraction.min() * 100)
    clip_max = float(four.clip_fraction.max() * 100)
    lines = ['# 四类毫米波特征提取与录制级探索验证', '', f'输入为已导出的雷达数据，{len(rec)} 个录制、{len(meta)} 个不重叠的 2 s 窗口；四类为 {len(four)} 个录制、{int((meta.label >= 0).sum())} 窗。只读输入；未修改原始数据。', '用户确认 keep 为保持座/底座故障、bigNormal 为另一台正常电机。均保留原标签 -1，独立用于未知故障/外部正常检查，不纳入四类监督。用户确认两模态同时采集，但仍无逐包时间轴及同步窗口对应关系。', f'**几何状态：{history}** 本次距离容限 {DISTANCE_M:.2f}±0.16 m 外的录制数为 {range_mismatch}。', '', '## 可训练接口', '', '`features.npz` 的每一行对应 `metadata.csv` 同一行；`bag_id` 为原雷达 recording_id。', '- 主分支 `frame_shape[N,128]`：12 个相邻 RX/bin 单元的逐帧 IQ 功率谱，先做固定频带功率均值，再 log10，再减去本窗口频带均值。', '- 可选 `shape_phase[N,256]`：主分支拼接 `phase_shape[N,128]`。相位速率只由同一帧的相邻有效 chirp 计算；按 191 点帧分别计算 PSD，先平均完整帧，再对 12 个单元取中位数。', '- `frame_abs` / `phase_abs` 是未减跨频带均值的对照；IQ 已经过逐单元幅度归一化，前者不是物理绝对振动能量。', '- `coherent_shape` 是有跨帧相干假设的独立对照，不能自动作为主分支；`time_stats[N,10]` 是无量纲幅度尾部统计和帧内归一化自相关。', '- `confound_distance`、`confound_raw_amp` 仅作混杂基线，不进入主学生。去掉距离字段不能消除信号中已有的距离影响。', '', '频带固定为 5–100 Hz 附近 2 Hz、100–400 Hz 5 Hz、400–800 Hz 20 Hz，共 128 个频带；首带为 5–6 Hz。', '帧内 192 点、2 kHz，频谱真实分辨能力约 10.42 Hz；0.5 Hz 只是补零求值网格，不能声称分辨 0.5 Hz 的故障边带。每帧独立计算，不跨缺口拼接。', '', '## 数据质量与混杂', '', markdown_table(rec), '', '历史预处理使用 0.15–2 m 搜索最强峰。每份 NPZ 只保留所选峰附近 3 个 bin，无法仅靠该导出恢复其他距离门；修改 ROI 必须回到原始 bin。', '外部电机的频谱与四类测试台不同，须同时核对转速/负载等工况，不能把域差异全部归因于结构变化。', f'原始 int16 的近满量程占比四类为 {clip_min:.3f}%–{clip_max:.3f}%；该值是饱和风险指标，不能单独证明每个值均发生硬削顶。相位非线性/谐波不能直接解释为故障。下一批需单独降低接收增益并检查峰值分布。', f'推荐所有类别统一使用 {DISTANCE_M:.2f}±0.16 m 的物理 ROI；随后人工检查目标峰。不能按类别选门，不能仅修复某个类别而对其他类别维持另一套筛选规则。', '图中 10–100 Hz 主峰是条件性相干谱诊断，不是测得的转速；当前未提供独立转速记录，不能确认各类同速。', '', '![质量与混杂](figures/01_quality_confound.png)', '', '![频谱](figures/02_recording_spectra.png)', '', '## 录制级探索基线', '', '按串口波特率文件组 115200 与 460800 双向训练/测试。波特率仅是串口参数，不表示雷达采样率或环境变化。每组每类仅 1 个录制；不随机划窗、不让同一个 bag 跨集合。', '这只是录制分组的探索验证，不能称为跨环境泛化或独立轴承泛化：独立 run、转速和轴承实体均未确认；几何问题见上面的实测状态。', '固定 LogisticRegression C=1，source-only StandardScaler，训练窗口按录制等权；无目标调参。测试录制概率为窗口概率均值。', '每方向只有 4 个测试录制，每错一个 accuracy 变化 25 个百分点；窗口数不增加独立样本数量。', '', markdown_table(metrics), '', '![录制级探索基线](figures/03_recording_baselines.png)', '', '如果距离或原始幅值基线有效，说明采集混杂可预测标签；不能把主方法相似或更高的分数直接解释为捕捉到了故障机制。', '测试已用于此次方法对照；后续方法选择后必须重新采独立盲测数据，不能反复针对这 8 个录制调参后作为最终论文结果。', '', '## 最小补采要求', '', '1. 四类在同一转速、同一距离、同一测点/角度/增益下，每类至少 3 次重新启动或重新安装录制；记录轴承实体，实体留出需要每类至少 2 个实体。', '2. 同类别跨距离和跨环境采集，使距离与标签交叉；每种故障都覆盖同样距离，不允许某类别只出现于某距离。', '3. 保存原始 bin、实际 cfg、ADC/loss 日志、实际转速/距离，以及接触式逐包时间信息；先验证 ROI 与饱和，再训练学生。', '4. 教师-学生训练当前只能使用训练 bag 的分布/原型监督。精细逐窗对齐必须依赖同步证据；不要人为把第 i 个雷达窗配给第 i 个接触窗。', '', '## 复现', '', '```bash', 'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\\n/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \\\npython /home/huangyating/anomaly_detection/four_class_chain_20260922/radar/extract_features.py \\\n  --source ' + str(SOURCE) + ' \\\n  --output /path/to/new_empty_radar_output', '```', '', '`input_manifest.json` 记录各输入 NPZ/JSON 哈希；`feature_schema.json` 记录接口、频带与限制；`baseline_protocol.json` 记录固定对照协议。']
    (ROOT / '毫米波特征与探索验证.md').write_text('\n'.join(lines) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True, help='New or empty feature-output directory')
    parser.add_argument('--distance-m', type=float, default=DISTANCE_M)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    ROOT = args.output.resolve()
    DISTANCE_M = args.distance_m
    if not np.isfinite(DISTANCE_M) or DISTANCE_M <= 0.04:
        parser.error('distance-m must be finite and >0.04')
    if ROOT.exists() and any(ROOT.iterdir()):
        parser.error('output must be new or empty; no overwrite')
    ROOT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    data, meta, rec = extract()
    metrics = baselines(data, meta)
    figures(rec, metrics)
    report(data, meta, rec, metrics)
    write_json(ROOT / 'runtime.json', dict(python=sys.version, executable=sys.executable, numpy=np.__version__, seconds=time.time() - t, script_sha256=digest(Path(_c2r_resolve_path(__file__))), source=str(SOURCE), output=str(ROOT), distance_m=DISTANCE_M, threads=2, window_count=len(meta)))
    print(metrics.to_string(index=False))
    print('Complete.', flush=True)
