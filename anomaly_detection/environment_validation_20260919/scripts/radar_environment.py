"""Reproducible AWR1843/DCA1000 environment study; no fault-diagnosis claims.

The parser assumes reordered, headerless ADC payload, 2I/2Q LVDS layout,
four non-interleaved RX, 256 samples, 192 chirps per 100 ms, 500 us/chirp.
No raw data are modified. Incomplete tails are reported and excluded.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import argparse, hashlib, json, re, time, os
import numpy as np
import pandas as pd
from scipy import signal, ndimage
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parents[1]
DATA = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/data9.18'))
FS = 2000.0
CHIRPS = 192
GRID = 200
ADC = 256
RX = 4
FRAME_BYTES = CHIRPS * ADC * RX * 4
RANGE_AXIS = np.arange(512) * 3430000 / 512 * 299792458 / (2 * 49990000000000.0)

def metadata(path):
    m = re.fullmatch('(room|narrow)-(off|\\d+r)-(\\d+)cm-(.+)\\.bin', path.name)
    if not m:
        raise ValueError(f'Unrecognized filename: {path.name}')
    env, speed, distance, suffix = m.groups()
    return dict(file=path.name, environment=env, rpm=0 if speed == 'off' else int(speed[:-1]), distance_cm=int(distance), label='off' if speed == 'off' else speed[:-1], suffix=suffix, bearing_state='not_inferred' if speed == 'off' else 'normal')

def snapshot():
    return [dict(**metadata(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(DATA.glob('*.bin'))]

def decode(raw, layout='2i2q'):
    """Convert int16 payload to [chirp, RX, fast-time sample]."""
    if len(raw) % (ADC * RX * 2):
        raise ValueError('Input must contain whole chirps')
    if layout == '2i2q':
        a = raw.reshape(-1, 4).astype(np.float32)
        z = np.empty((len(a), 2), dtype=np.complex64)
        z[:, 0] = a[:, 0] + 1j * a[:, 2]
        z[:, 1] = a[:, 1] + 1j * a[:, 3]
        return z.reshape(-1, RX, ADC)
    if layout == 'iqiq':
        a = raw.reshape(-1, 2).astype(np.float32)
        return (a[:, 0] + 1j * a[:, 1]).reshape(-1, RX, ADC)
    raise ValueError(layout)

def range_fft(z, conjugate=False):
    if conjugate:
        z = z.conj()
    z = z - z.mean(-1, keepdims=True)
    return np.fft.fft(z * np.hanning(ADC), n=512, axis=-1).astype(np.complex64)

def audit_prefix():
    """Read fixed prefixes only for parser plausibility while uploads finish."""
    rows = []
    for p in sorted(DATA.glob('*.bin')):
        raw = np.fromfile(p, dtype='<i2', count=FRAME_BYTES // 2 * 2)
        raw = raw[:len(raw) // (ADC * RX * 2) * (ADC * RX * 2)]
        if not len(raw):
            continue
        meta = metadata(p)
        for layout in ['2i2q', 'iqiq']:
            for conjugate in [False, True]:
                rf = range_fft(decode(raw, layout), conjugate)
                power = np.mean(abs(rf) ** 2, axis=(0, 1))
                d = meta['distance_cm'] / 100
                roi = (RANGE_AXIS >= max(0.04, d - 0.16)) & (RANGE_AXIS <= d + 0.16)
                near = (RANGE_AXIS >= 0.04) & (RANGE_AXIS <= 2.0)
                indices = np.flatnonzero(roi)
                peak = indices[np.argmax(power[indices])]
                rows.append(dict(**meta, layout=layout, conjugate=conjugate, prefix_bytes=len(raw) * 2, roi_peak_m=RANGE_AXIS[peak], roi_to_near_median_db=10 * np.log10(power[peak] / np.median(power[near])), roi_fraction=float(power[roi].sum() / power[near].sum()), clip_fraction=float(np.mean(abs(raw.astype(np.int32)) >= 32760))))
    pd.DataFrame(rows).to_csv(ROOT / 'results/parser_prefix_audit.csv', index=False)
    print(pd.DataFrame(rows).groupby(['layout', 'conjugate'])[['roi_to_near_median_db', 'roi_fraction']].median().to_string(), flush=True)

def masked_sinusoid_power(y, mask, fs=FS):
    """Harmonic LS on the actual regular grid with holes, incl. intercept.

    y: [grid positions, channels], complex or real; false mask positions ignored.
    Fits an intercept, cos, sin for each frequency separately. This does NOT
    invert arbitrary spectral leakage or prove cross-frame phase coherence.
    The FFT evaluates sufficient statistics, never treats holes as measurements.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    valid = mask.sum()
    f = np.fft.rfftfreq(n, 1 / fs)
    k = np.flatnonzero((f >= 5) & (f <= 800))
    w = mask.astype(float)
    mw = np.fft.fft(w)
    yy = np.where(mask[:, None], y, 0)
    yf = np.fft.fft(yy, axis=0)
    cc = mw[k].real
    ss = -mw[k].imag
    c2 = mw[2 * k % n].real
    s2 = -mw[2 * k % n].imag
    gram = np.empty((len(k), 3, 3))
    gram[:, 0, 0] = valid
    gram[:, 0, 1] = gram[:, 1, 0] = cc
    gram[:, 0, 2] = gram[:, 2, 0] = ss
    gram[:, 1, 1] = (valid + c2) / 2
    gram[:, 2, 2] = (valid - c2) / 2
    gram[:, 1, 2] = gram[:, 2, 1] = s2 / 2
    rhs = np.empty((len(k), 3, y.shape[1]), dtype=complex)
    rhs[:, 0] = yy.sum(0)
    rhs[:, 1] = (yf[k] + yf[-k % n]) / 2
    rhs[:, 2] = (yf[-k % n] - yf[k]) / 2j
    beta = np.linalg.solve(gram, rhs)
    power = (abs(beta[:, 1]) ** 2 + abs(beta[:, 2]) ** 2) / 2
    return (f[k], power.real)

def pool_bins(f, power):
    edges = np.unique(np.r_[5, np.arange(6, 102, 2), np.arange(105, 405, 5), np.arange(420, 801, 20)])
    return np.asarray([np.mean(power[(f >= a) & (f < b)], axis=0) for a, b in zip(edges[:-1], edges[1:])])

def extract_file(path, conjugate=False, seconds=2):
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    nframes = before[0] // FRAME_BYTES
    if nframes < seconds * 10:
        raise ValueError(f'{path.name}: insufficient complete frames')
    raw = np.memmap(path, dtype='<i2', mode='r', shape=(nframes * FRAME_BYTES // 2,))
    meta = metadata(path)
    d = meta['distance_cm'] / 100
    ridx = np.flatnonzero((RANGE_AXIS >= max(0.04, d - 0.16)) & (RANGE_AXIS <= d + 0.16))
    pilot = range_fft(decode(np.asarray(raw[:FRAME_BYTES // 2 * min(3, nframes)])), conjugate)
    rp = np.mean(abs(pilot) ** 2, axis=(0, 1))
    center = ridx[np.argmax(rp[ridx])]
    bins = np.array([max(1, center - 1), center, min(254, center + 1)])
    selected_pilot = pilot[:, :, bins]
    best = np.argmax(np.mean(abs(selected_pilot) ** 2, axis=0).reshape(-1))
    all_windows = []
    powers = []
    count_clipped = count_zero = count_int = 0
    frame_jumps = []
    within_steps = []
    frames_per_window = int(seconds * 10)
    for first in range(0, nframes - frames_per_window + 1, frames_per_window):
        block = np.asarray(raw[first * FRAME_BYTES // 2:(first + frames_per_window) * FRAME_BYTES // 2])
        count_clipped += int(np.count_nonzero(abs(block.astype(np.int32)) >= 32760))
        count_zero += int(np.count_nonzero(block == 0))
        count_int += len(block)
        rf = range_fft(decode(block), conjugate)[:, :, bins].reshape(frames_per_window, CHIRPS, -1)
        within_steps.append(float(np.median(abs(np.angle(rf[:, 1:, best] * rf[:, :-1, best].conj())))))
        frame_jumps.append(float(np.median(abs(np.angle(rf[1:, 0, best] * rf[:-1, -1, best].conj())))))
        raw_chirps = block.reshape(frames_per_window, CHIRPS, -1)
        valid_chirp = np.any(raw_chirps != 0, axis=-1)
        mag = np.median(abs(rf), axis=(0, 1))
        iq = rf / np.maximum(mag, 1)
        iq -= np.mean(iq, axis=1, keepdims=True)
        grid = np.zeros((frames_per_window, GRID, rf.shape[-1]), dtype=np.complex64)
        mask = np.zeros((frames_per_window, GRID), dtype=bool)
        grid[:, :CHIRPS] = iq
        mask[:, :CHIRPS] = valid_chirp
        f, p = masked_sinusoid_power(grid.reshape(-1, rf.shape[-1]), mask.reshape(-1))
        multi = np.median(p, axis=1)
        phase = np.angle(rf[:, 1:, best] * rf[:, :-1, best].conj()) * FS
        phase -= np.mean(phase, axis=1, keepdims=True)
        pg = np.zeros((frames_per_window, GRID))
        pm = np.zeros_like(pg, dtype=bool)
        pg[:, 1:CHIRPS] = phase
        pm[:, 1:CHIRPS] = valid_chirp[:, 1:] & valid_chirp[:, :-1]
        _, pp = masked_sinusoid_power(pg.ravel(), pm.ravel())
        single = pp[:, 0]
        raw_amp = np.log10(np.mean(abs(rf) ** 2, axis=(0, 1)) + 1)
        multi_b = pool_bins(f, multi)
        phase_b = pool_bins(f, single)
        complete = valid_chirp.all(axis=1)
        if not np.any(complete):
            raise ValueError(f'{path.name}: no fully valid frame in a window')
        fspec = np.fft.fft(iq[complete] * np.hanning(CHIRPS)[None, :, None], n=2048, axis=1)
        fframe = np.fft.fftfreq(2048, 1 / FS)
        positive = np.flatnonzero((fframe >= 5) & (fframe <= 800))
        frame_power = np.median(np.mean(abs(fspec[:, positive]) ** 2 + abs(fspec[:, -positive % 2048]) ** 2, axis=0), axis=-1)
        frame_b = pool_bins(fframe[positive], frame_power)
        abs_feat = np.log10(multi_b + 1e-14)
        shape_feat = abs_feat - np.mean(abs_feat)
        phase_feat = np.log10(phase_b + 1e-12)
        features = dict(raw_amplitude=raw_amp, phase_single=phase_feat, iq_multi_abs=abs_feat, iq_multi_shape=shape_feat, iq_frame_shape=np.log10(frame_b + 1e-14) - np.mean(np.log10(frame_b + 1e-14)))
        row = dict(**meta, window_start_s=first / 10, window_duration_s=seconds, valid_fraction=float(mask.mean()), range_peak_m=float(RANGE_AXIS[center]))
        all_windows.append((row, features))
        powers.append((multi_b, f, multi, single, frame_b))
    after = (path.stat().st_size, path.stat().st_mtime_ns)
    if after != before:
        raise RuntimeError(f'{path.name} changed during processing; discard results')
    qc = dict(**meta, size=before[0], mtime_ns=before[1], full_frames=nframes, nominal_complete_duration_s=nframes / 10, trailing_bytes=before[0] % FRAME_BYTES, complete_windows=len(all_windows), roi_peak_m=float(RANGE_AXIS[center]), selected_bins=bins.tolist(), best_channel=int(best), clip_fraction=count_clipped / max(1, count_int), zero_fraction=count_zero / max(1, count_int), median_within_chirp_phase_step_rad=float(np.median(within_steps)), median_between_frame_phase_step_rad=float(np.median(frame_jumps)), median_range_profile=rp[:128].tolist())
    return (all_windows, powers, qc)

def extract(conjugate=False):
    initial = snapshot()
    rows = []
    feats = {}
    powers = []
    frame_powers = []
    spectra = []
    qc = []
    started = time.monotonic()
    for i, p in enumerate(sorted(DATA.glob('*.bin'))):
        windows, pp, q = extract_file(p, conjugate)
        qc.append(q)
        for (row, ff), (power, f, spec, phase, frame_b) in zip(windows, pp):
            rows.append(row)
            powers.append(power)
            frame_powers.append(frame_b)
            spectra.append(spec)
            for key, v in ff.items():
                feats.setdefault(key, []).append(v)
        print(f"{i + 1}/{len(initial)} {p.name}: {len(windows)} windows, {q['nominal_complete_duration_s']:.1f}s, tail {q['trailing_bytes']} bytes", flush=True)
    if snapshot() != initial:
        raise RuntimeError('Directory changed during extraction; rerun after upload is complete')
    pd.DataFrame(rows).to_csv(ROOT / 'results/windows.csv', index=False)
    np.savez_compressed(ROOT / 'results/features.npz', **{k: np.array(v) for k, v in feats.items()}, power_binned=np.array(powers), frame_power_binned=np.array(frame_powers), spectra=np.array(spectra), frequencies=f)
    (ROOT / 'results/quality.json').write_text(json.dumps(qc, ensure_ascii=False, indent=2))
    (ROOT / 'results/input_manifest.json').write_text(json.dumps(initial, ensure_ascii=False, indent=2))
    (ROOT / 'results/processing_config.json').write_text(json.dumps(dict(layout='2i2q', conjugate=conjugate, fs_slow_hz=FS, chirps_per_frame=CHIRPS, grid_slots_per_frame=GRID, frame_s=0.1, adc_samples=ADC, rx=RX, window_s=2, mask_method='harmonic least squares with intercept; frame-centred IQ', roi='known distance +/- 0.16 m; maximum mean range power; adjacent 3 bins', source='TI SWRA581B, section 9.2, assumed applicable to this DCA payload', wall_seconds=time.monotonic() - started), indent=2))

def self_check():
    rng = np.random.default_rng(24)
    a = rng.integers(-3000, 3000, size=(2, RX, ADC), dtype=np.int16)
    b = rng.integers(-3000, 3000, size=(2, RX, ADC), dtype=np.int16)
    c = (a.astype(float) + 1j * b).reshape(-1, 2)
    packed = np.c_[c[:, 0].real, c[:, 1].real, c[:, 0].imag, c[:, 1].imag].astype('<i2').ravel()
    assert np.array_equal(decode(packed), a + 1j * b)
    n = 4000
    t = np.arange(n) / FS
    m = np.arange(n) % GRID < CHIRPS
    y = (2 * np.cos(2 * np.pi * 33 * t) + 0.4 * np.sin(2 * np.pi * 33 * t) + 3)[:, None]
    f, p = masked_sinusoid_power(y, m)
    k = np.argmin(abs(f - 33))
    design = np.c_[np.ones(n), np.cos(2 * np.pi * f[k] * t), np.sin(2 * np.pi * f[k] * t)]
    coeff = np.linalg.lstsq(design[m], y[m], rcond=None)[0]
    direct = float((abs(coeff[1]) ** 2 + abs(coeff[2]) ** 2).item() / 2)
    assert abs(p[k, 0] - direct) < 1e-10
    assert abs(p[k, 0] - 2.08) < 1e-10
    assert f[np.argmax(p[:, 0])] == 33
    ff = np.fft.fftfreq(2048, 1 / FS)
    ff = ff[(ff >= 5) & (ff <= 800)]
    assert np.isfinite(pool_bins(ff, np.ones(len(ff)))).all()
    print('Passed known-layout round trip and masked LS versus direct regression.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['snapshot', 'audit-prefix', 'extract', 'self-check'])
    parser.add_argument('--conjugate', action='store_true')
    args = parser.parse_args()
    (ROOT / 'results').mkdir(exist_ok=True, parents=True)
    if args.command == 'snapshot':
        s = snapshot()
        (ROOT / 'results/upload_snapshot.json').write_text(json.dumps(s, indent=2))
        print(json.dumps(dict(files=len(s), GB=sum((x['size'] for x in s)) / 1000000000.0, newest_mtime_ns=max((x['mtime_ns'] for x in s))), indent=2))
    elif args.command == 'audit-prefix':
        audit_prefix()
    elif args.command == 'extract':
        extract(args.conjugate)
    else:
        self_check()
