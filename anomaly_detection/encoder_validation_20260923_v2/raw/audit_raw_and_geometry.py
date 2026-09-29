"""Read-only capture audit after fixed physical ROI re-export. No model selection."""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
from pathlib import Path
import json, hashlib, importlib.util, collections
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parent
CAP = Path(_c2r_resolve_path('/media/nas_users/huangyating/data'))
OLD = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results'))
NEW = ROOT.parent / 'geometry_corrected/results'
OLD_F = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_chain_20260922/radar'))
NEW_F = ROOT.parent / 'geometry_corrected/radar'
spec = importlib.util.spec_from_file_location('pre', OLD.parent / 'preprocess.py')
pre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre)

def read(p):
    return json.loads(p.read_text())

def write(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''):
            h.update(b)
    return h.hexdigest()

def contiguous_runs(path):
    """Audit only: maximal valid counter-consecutive runs within a packet, no repairs."""
    tokens = path.read_bytes().split(b';')
    runs = []
    first = None
    prev = None
    length = 0
    pos = 0

    def finish():
        nonlocal first, prev, length
        if length:
            runs.append(dict(byte_start=first[0], counter_start=first[1], counter_end=prev, n=length))
        first = None
        prev = None
        length = 0
    for i, t in enumerate(tokens):
        here = pos
        pos += len(t) + 1
        m = pre.RECORD.fullmatch(t)
        if i == len(tokens) - 1 or m is None:
            finish()
            continue
        marker, c, *xyz = m.groups()
        c = int(c)
        x = np.asarray([float(v) for v in xyz])
        if not 1 <= c <= 4096 or not np.isfinite(x).all() or (marker and c != 1):
            finish()
            continue
        if prev is None or c != prev + 1:
            finish()
            first = (here, c)
        prev = c
        length += 1
    finish()
    return runs

def contact_audit():
    manifest = {r['file']: r for r in read(OLD / 'input_manifest.json')}
    summary = read(OLD / 'summary.json')
    rows = []
    wins = []
    runrows = []
    for rec in summary:
        if rec['modality'] != 'contact':
            continue
        path = CAP / rec['file']
        digest = sha(path)
        assert digest == manifest[path.name]['sha256']
        packets, q = pre.parse_packets(path)
        oldj = read(OLD / Path(_c2r_resolve_path(rec['output'])).with_suffix('.json'))
        assert q == oldj['quality']
        raw = np.stack([p[1][48:4048].astype(np.float32) for p in packets])
        with np.load(OLD / rec['output']) as z:
            eq = bool(np.array_equal(raw, z['raw_xyz']))
            assert eq
        runs = contiguous_runs(path)
        eligible = [r for r in runs if r['n'] >= 4000]
        complete_positions = {p[0]['byte_start'] for p in packets}
        extra = [r for r in eligible if not (r['counter_start'] == 1 and r['n'] == 4096 and (r['byte_start'] in complete_positions))]
        issues = collections.Counter((x['reason'] for x in q['issues']))
        entry = dict(file=path.name, state=rec['state'], label=rec['label'], baud_candidate=rec['baud_candidate'], sha256=digest, sha_matches_archive=True, raw_xyz_bitwise_equal=True, complete_packets=len(packets), saved_1s_windows=len(raw), packet_starts=q['packet_starts_observed'], valid_syntax_rows=q['valid_syntax_rows'], complete_packet_sample_rows=len(packets) * 4096, valid_syntax_rows_outside_complete_packets=q['valid_syntax_rows'] - len(packets) * 4096, usable_sample_rows=len(raw) * 4000, intentionally_trimmed_rows=len(raw) * 96, issue_count=q['issue_count'], invalid_record_count=issues.get('invalid_record', 0), counter_discontinuity_count=issues.get('counter_discontinuity', 0), contiguous_runs_ge4000=len(eligible), additional_contiguous_1s_candidates=len(extra), median_AC_RMS_x=float(np.median(np.std(raw[:, :, 0], axis=1))), median_AC_RMS_y=float(np.median(np.std(raw[:, :, 1], axis=1))), median_AC_RMS_z=float(np.median(np.std(raw[:, :, 2], axis=1))))
        rows.append(entry)
        extra_starts = {r['byte_start'] for r in extra}
        for r in runs:
            runrows.append(dict(file=path.name, **r, eligible_1s=r['n'] >= 4000, additional_1s_candidate=r['byte_start'] in extra_starts))
        for idx, x in enumerate(raw.astype(np.float64)):
            ac = x - x.mean(0)
            rms = np.sqrt(np.mean(ac * ac, axis=0))
            crest = np.max(abs(ac), axis=0) / rms
            kur = kurtosis(ac, axis=0, fisher=True, bias=False)
            for a, axis in enumerate('xyz'):
                wins.append(dict(file=path.name, state=rec['state'], label=rec['label'], baud_candidate=rec['baud_candidate'], window=idx, axis=axis, dc_mean=float(x[:, a].mean()), dc_to_ac_ratio=float(abs(x[:, a].mean()) / rms[a]), ac_rms=float(rms[a]), excess_kurtosis=float(kur[a]), crest_factor=float(crest[a]), unchanged_adjacent_fraction=float(np.mean(np.diff(x[:, a]) == 0)), value_min=float(x[:, a].min()), value_max=float(x[:, a].max())))
    pd.DataFrame(rows).to_csv(ROOT / 'contact_raw_identity_and_quality.csv', index=False)
    pd.DataFrame(wins).to_csv(ROOT / 'contact_window_quality.csv', index=False)
    pd.DataFrame(runrows).to_csv(ROOT / 'contact_contiguous_run_inventory.csv', index=False)
    return rows

def radar_audit():
    old = read(OLD / 'summary.json')
    new = {r['file']: r for r in read(NEW / 'summary.json')}
    rows = []
    profiles = []
    with np.load(OLD_F / 'features.npz') as a, np.load(NEW_F / 'features.npz') as b:
        assert np.array_equal(a['bag_id'], b['bag_id'])
        assert np.array_equal(a['recording_row'], b['recording_row'])
        changed = np.max(abs(a['frame_shape'] - b['frame_shape']), axis=1)
        md = pd.read_csv(NEW_F / 'metadata.csv')
        delta_by_record = {g: float(changed[r.index].max()) for g, r in md.groupby('recording_id')}
    for oldr in old:
        if oldr['modality'] != 'radar':
            continue
        name = oldr['file']
        nr = new[name]
        oj = read(OLD / Path(_c2r_resolve_path(oldr['output'])).with_suffix('.json'))
        nj = read(NEW / Path(_c2r_resolve_path(nr['output'])).with_suffix('.json'))
        oq = oj['quality']
        nq = nj['quality']
        path = CAP / name
        x = np.memmap(path, dtype='<i2', mode='r', shape=(3 * pre.FRAME_BYTES // 2,))
        f = pre.decode_range(x)
        profile = np.mean(abs(f) ** 2, axis=(0, 1))
        profiles.append(profile)
        oi = int(np.argmin(abs(pre.RANGE - oq['range_center_m'])))
        ni = int(np.argmin(abs(pre.RANGE - nq['range_center_m'])))
        rows.append(dict(file=name, state=oldr['state'], label=oldr['label'], baud_candidate=oldr['baud_candidate'], old_range_m=oq['range_center_m'], new_range_m=nq['range_center_m'], range_changed=oi != ni, old_windows=oldr['windows'], new_windows=nr['windows'], complete_frames=nq['complete_frames'], trailing_bytes=nq['trailing_bytes'], unused_complete_frames=nq['unused_complete_frames'], near_full_scale_int16_fraction=nq['clipped_fraction'], zero_int16_fraction=nq['zero_int16_fraction'], rejected_windows=len(nq['rejected_windows']), first3frames_new_peak_db_below_old=float(10 * np.log10(profile[oi] / profile[ni])), maximum_128feature_absolute_change=delta_by_record[oldr['recording_id']]))
    pd.DataFrame(rows).to_csv(ROOT / 'radar_geometry_before_after.csv', index=False)
    np.savez_compressed(ROOT / 'initial_range_profiles.npz', power=np.asarray(profiles), range_m=pre.RANGE, files=np.asarray([r['file'] for r in rows]))
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    for ax, r, p in zip(axes.flat, rows, profiles):
        ax.plot(pre.RANGE, 10 * np.log10(np.maximum(p, 1e-09) / p.max()), c='#277DA1', lw=1.25)
        ax.axvspan(0.29, 0.61, color='#43AA8B', alpha=0.15)
        ax.axvline(r['old_range_m'], c='#F8961E', ls='--', label='Previous peak')
        ax.axvline(r['new_range_m'], c='#9B5DE5', ls=':', label='Fixed-ROI peak')
        ax.set_title(f"{r['state']} / {r['baud_candidate']}")
        ax.set_xlim(0.15, 1.4)
        ax.set_ylim(-55, 2)
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel('Range (m)')
    fig.supylabel('Power relative to strongest pilot range cell (dB)')
    fig.suptitle('Same 0.29–0.61 m physical search range for every recording; first 3 frames')
    fig.tight_layout()
    fig.savefig(ROOT / 'radar_range_before_after.png', dpi=180)
    fig.savefig(ROOT / 'radar_range_before_after.pdf')
    plt.close(fig)
    return rows

def main():
    contact = contact_audit()
    radar = radar_audit()
    c = pd.DataFrame(contact)
    r = pd.DataFrame(radar)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = [x['state'] + ' / ' + str(x['baud_candidate']) for x in contact]
    xx = np.arange(len(names))
    axes[0].bar(xx - 0.18, c.packet_starts, width=0.36, label='Observed packet starts', color='#277DA1')
    axes[0].bar(xx + 0.18, c.complete_packets, width=0.36, label='Complete 4096-point packets', color='#43AA8B')
    axes[1].bar(xx, c.invalid_record_count + c.counter_discontinuity_count, color='#F8961E')
    for ax in axes:
        ax.set_xticks(xx, names, rotation=60, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.2)
    axes[0].set_ylabel('Packet count')
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Invalid-record + counter-gap events')
    fig.tight_layout()
    fig.savefig(ROOT / 'contact_capture_integrity.png', dpi=180)
    plt.close(fig)
    verification = dict(status='pass', raw_DAT_SHA_match_count=int(c.sha_matches_archive.sum()), raw_DAT_count=len(c), complete_contact_packets=int(c.complete_packets.sum()), four_class_contact_packets=int(c.loc[c.label >= 0, 'complete_packets'].sum()), old_raw_xyz_bitwise_equal_count=int(c.raw_xyz_bitwise_equal.sum()), additional_contiguous_4000_sample_candidates=int(c.additional_contiguous_1s_candidates.sum()), radar_windows_old=int(r.old_windows.sum()), radar_windows_new=int(r.new_windows.sum()), changed_range_recordings=r.loc[r.range_changed, 'file'].tolist(), bitwise_unchanged_feature_recordings=r.loc[r.maximum_128feature_absolute_change == 0, 'file'].tolist(), numerically_close_feature_recordings_at_absolute_tolerance_1e_5=r.loc[r.maximum_128feature_absolute_change < 1e-05, 'file'].tolist(), previous_export_numpy='2.0.2 on macOS Python3.9.6', current_export_numpy=np.__version__, all_ranges_in_shared_ROI=bool(((r.new_range_m >= 0.29) & (r.new_range_m <= 0.61)).all()), no_model_accuracy_based_quality_filter=True, caveats=['ADC packing assumes archived 2I2Q conjugated headerless framing; raw identity does not independently verify firmware layout.', 'No DCA packet-loss log, no per-packet absolute contact timestamps, no one-to-one window alignment inferred.', 'Amplitude clipping flags count near int16 limits; ADC bit alignment and analog gain must be checked before physical saturation claims.', 'All recordings previously inspected; this is development reprocessing, not a new blind test.'])
    write(ROOT / 'audit_verification.json', verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
