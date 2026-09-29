"""Offline, unpaired contact/radar export. Never alter raw recordings.

python preprocess.py --data /path/to/data --output results [--distance-m 0.8]
Only the archived R0 radar configuration is supported. NumPy + SciPy required.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime

import numpy as np
import scipy
from scipy.fft import dct
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent
LABELS = {'normal': 0, 'inBroken': 1, 'outBroken': 2, 'roll': 3}
NUMBER = rb'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
RECORD = re.compile(rb'(!?)(\d+),(' + NUMBER + rb'),(' + NUMBER + rb'),(' + NUMBER + rb')')
ADC, RX, CHIRPS, GRID, FRAME_BYTES = 256, 4, 192, 200, 786432
FREQ = np.arange(5., 800.01, .5)
RANGE = np.arange(512) * 3430000 / 512 * 299792458 / (2 * 49.99e12)


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def metadata(path):
    stamp, state, baud = path.stem.rsplit('-', 2)
    fmt = '%Y%m%d-%H%M%S' if path.suffix == '.bin' else '%Y_%m_%d_%H-%M-%S'
    dt = datetime.strptime(stamp, fmt)
    # Keep raw spelling; only this one observed typo is mapped for candidate pairing.
    effective = 115200 if baud == '11520' else int(baud)
    return dict(file=path.name, recording_id=path.stem, state=state,
                label=LABELS.get(state, -1), baud_raw=baud, baud_candidate=effective,
                filename_start_local=dt.isoformat(), timezone='Asia/Shanghai',
                timestamp_semantics='file start only; not packet sample time',
                condition_id=None, bearing_entity_id=None, independent_run_id=None,
                split='unassigned', four_class_eligible=state in LABELS)


def dcn(x):
    """[4000,3] -> [3,24000]; no spectral magnitude or time-domain padding."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (4000, 3) or not np.isfinite(x).all():
        raise ValueError('Need exactly 4000 finite, contiguous XYZ samples')
    x = x - x.mean(axis=0)
    if np.any(np.mean(x*x, axis=0) <= 1e-16):
        raise ValueError('Degenerate axis AC energy')
    c = dct(x, type=2, axis=0, norm='backward').T
    out = np.zeros((3, 24000), dtype=np.float32)
    out[:, :4000] = c * (.01 * np.sqrt(24000) / np.linalg.norm(c, axis=1))[:, None]
    return out


def parse_packets(path):
    """Only explicitly complete 1..4096 sequences; no cross-gap salvage/repair."""
    raw = path.read_bytes()
    parts = raw.split(b';')
    packets, rows, issues = [], [], []
    start = None
    offset = 0
    starts = 0
    valid_rows = 0
    for i, token in enumerate(parts):
        pos = offset
        offset += len(token) + 1
        if not token and i == len(parts)-1:
            continue
        m = RECORD.fullmatch(token)
        reason = None
        if i == len(parts)-1:
            reason = 'unterminated_tail'
        elif m is None:
            reason = 'invalid_record'
        else:
            marker, counter, *xyz = m.groups()
            counter = int(counter)
            xyz = np.asarray(list(map(float, xyz)))
            if not 1 <= counter <= 4096 or not np.isfinite(xyz).all():
                reason = 'counter_or_nonfinite'
            elif marker and counter != 1:
                reason = 'unexpected_packet_marker'
        if reason:
            issues.append(dict(byte=pos, reason=reason, token=token[:100].decode('ascii', errors='replace')))
            rows, start = [], None
            continue
        valid_rows += 1
        if counter == 1:
            if rows:
                issues.append(dict(byte=pos, reason='packet_restart_before_4096', retained_rows=len(rows)))
            starts += 1
            rows, start = [], dict(packet_ordinal_observed=starts, byte_start=pos)
        if start is None:
            continue
        if counter != len(rows) + 1:
            issues.append(dict(byte=pos, reason='counter_discontinuity', expected=len(rows)+1, observed=counter))
            rows, start = [], None
            continue
        rows.append(xyz)
        if counter == 4096:
            packets.append((start, np.asarray(rows, dtype=np.float64)))
            rows, start = [], None
    return packets, dict(valid_syntax_rows=valid_rows, packet_starts_observed=starts,
                         complete_packets=len(packets), issue_count=len(issues), issues=issues)


def contact_export(path, dest):
    meta = metadata(path)
    packets, quality = parse_packets(path)
    raw, features, common, rows = [], [], [], []
    rejected = []
    for pmeta, packet in packets:
        x = packet[48:4048]
        try:
            feat = dcn(x)
        except ValueError as e:
            rejected.append(dict(**pmeta, reason=str(e)))
            continue
        # Separate magnitude-power diagnostic branch; NOT BearLLM DCN.
        f, psd = welch(x-x.mean(0), fs=4000, window='hann', nperseg=4000,
                       noverlap=0, nfft=8000, axis=0)
        common.append(psd[(f >= 5) & (f <= 800)].T)
        raw.append(x.astype(np.float32))
        features.append(feat)
        rows.append(dict(**meta, **pmeta, row=len(rows), sample_start_in_packet=48,
                         relative_packet_start_s=.012, duration_s=1., sample_rate_hz=4000,
                         absolute_window_start=None, ac_rms=np.std(x, axis=0).tolist(),
                         value_min=packet.min(0).tolist(), value_max=packet.max(0).tolist()))
    n = len(rows)
    np.savez_compressed(dest, raw_xyz=np.asarray(raw, dtype=np.float32).reshape(n, 4000, 3),
        dcn=np.asarray(features, dtype=np.float32).reshape(n, 3, 24000),
        common_power=np.asarray(common, dtype=np.float32).reshape(n, 3, len(FREQ)),
        common_frequency_hz=FREQ, available_dcn_mask=np.arange(24000) < 4000,
        labels=np.full(n, meta['label'], np.int64), axis_order=np.array(['x', 'y', 'z']))
    save_json(dest.with_suffix('.json'), dict(metadata=meta, windows=rows, quality=quality,
              rejected_complete_packets=rejected, units='unconfirmed original sensor units',
              timing='packet-relative only; packet gaps unknown', ready_for_reference_pairing=True))
    return dict(**meta, modality='contact', windows=n, complete_packets=len(packets),
                issues=quality['issue_count'], nominal_observed_window_seconds=n,
                output=dest.name)


def decode_range(raw):
    """Archived 2I/2Q, four RX, little-endian int16; conjugate as prior audit."""
    a = raw.reshape(-1, 4).astype(np.float32)
    z = np.empty((len(a), 2), np.complex64)
    z[:, 0] = a[:, 0] - 1j*a[:, 2]
    z[:, 1] = a[:, 1] - 1j*a[:, 3]
    z = z.reshape(-1, RX, ADC)
    z -= z.mean(-1, keepdims=True)
    return np.fft.fft(z * np.hanning(ADC), n=512, axis=-1).astype(np.complex64)


def masked_power(y, mask):
    """2 s harmonic least squares, actual 2 kHz grid with missing observations."""
    y = np.asarray(y).reshape(4000, -1)
    w = np.asarray(mask, dtype=bool).reshape(4000)
    k = np.rint(FREQ*2).astype(int)
    mw = np.fft.fft(w.astype(float))
    yy = np.where(w[:, None], y, 0)
    yf = np.fft.fft(yy, axis=0)
    c, s = mw[k].real, -mw[k].imag
    c2, s2 = mw[2*k].real, -mw[2*k].imag
    gram = np.empty((len(k), 3, 3))
    gram[:, 0, 0] = w.sum()
    gram[:, 0, 1] = gram[:, 1, 0] = c
    gram[:, 0, 2] = gram[:, 2, 0] = s
    gram[:, 1, 1], gram[:, 2, 2] = (w.sum()+c2)/2, (w.sum()-c2)/2
    gram[:, 1, 2] = gram[:, 2, 1] = s2/2
    rhs = np.empty((len(k), 3, y.shape[1]), complex)
    rhs[:, 0] = yy.sum(0)
    rhs[:, 1] = (yf[k]+yf[-k])/2
    rhs[:, 2] = (yf[-k]-yf[k])/(2j)
    beta = np.linalg.solve(gram, rhs)
    return (abs(beta[:, 1])**2+abs(beta[:, 2])**2)/2


def radar_export(path, dest, distance):
    meta = metadata(path)
    size = path.stat().st_size
    nframes = size // FRAME_BYTES
    raw = np.memmap(path, dtype='<i2', mode='r', shape=(nframes*FRAME_BYTES//2,))
    lo, hi = ((max(.04, distance-.16), distance+.16) if distance is not None else (.15, 2.))
    roi = np.flatnonzero((RANGE >= lo) & (RANGE <= hi) & (np.arange(512) < 255))
    if not len(roi) or nframes < 20:
        raise ValueError('No ROI bins or fewer than 20 complete frames')
    pilot = decode_range(raw[:min(3, nframes)*FRAME_BYTES//2])
    profile = np.mean(abs(pilot)**2, axis=(0, 1))
    center = int(roi[np.argmax(profile[roi])])
    bins = np.arange(center-1, center+2)
    if bins.min() < 1 or bins.max() > 255:
        raise ValueError('ROI at unsupported FFT edge')
    best = int(np.argmax(np.mean(abs(pilot[:, :, bins])**2, axis=0)))
    iq_all, masks, phase_all, phase_masks, powers, frame_powers, rows = [], [], [], [], [], [], []
    absolute, rejected = [], []
    clipped, zeros, total = 0, 0, 0
    for first in range(0, nframes-19, 20):
        block = np.asarray(raw[first*FRAME_BYTES//2:(first+20)*FRAME_BYTES//2])
        valid = np.any(block.reshape(20, CHIRPS, -1) != 0, axis=-1)
        clipped += int(np.count_nonzero(abs(block.astype(np.int32)) >= 32760))
        zeros += int(np.count_nonzero(block == 0)); total += block.size
        if valid.mean() < .99 or not np.any(valid.all(axis=1)):
            rejected.append(dict(first_frame=first, reason='less_than_99_percent_observed_chirps_or_no_complete_frame'))
            continue
        rf = decode_range(block)[:, :, bins].reshape(20, CHIRPS, 12)
        scale = np.maximum(np.median(abs(rf[valid]), axis=0), 1)
        iq = rf/scale
        for j in range(20):
            iq[j] -= iq[j, valid[j]].mean(axis=0)
        iq[~valid] = 0
        grid = np.zeros((20, GRID, 12), np.complex64)
        mask = np.zeros((20, GRID), bool)
        grid[:, :CHIRPS], mask[:, :CHIRPS] = iq, valid
        p = np.median(masked_power(grid, mask), axis=1)
        # No phase differences across frame boundaries or missing samples.
        phase = np.angle(rf[:, 1:] * rf[:, :-1].conj()) * 2000
        pm = valid[:, 1:] & valid[:, :-1]
        for j in range(20):
            phase[j] -= phase[j, pm[j]].mean(axis=0)
        phase[~pm] = 0
        spec = np.fft.fft(iq[valid.all(axis=1)] * np.hanning(CHIRPS)[None, :, None], n=4000, axis=1)
        k = np.rint(FREQ*2).astype(int)
        fp = np.median(np.mean(abs(spec[:, k])**2 + abs(spec[:, -k])**2, axis=0), axis=1)
        # Hann normalization gives relative frame power density, not displacement PSD.
        fp /= 2000 * np.sum(np.hanning(CHIRPS)**2)
        amp = np.mean(abs(rf[valid])**2, axis=0)
        iq_all.append(np.stack([iq.real, iq.imag], axis=-1))
        masks.append(valid); phase_all.append(phase); phase_masks.append(pm)
        powers.append(p); frame_powers.append(fp); absolute.append(amp)
        rows.append(dict(**meta, row=len(rows), first_frame=first, window_start_s=first/10,
                         duration_s=2., valid_fraction_grid=float(mask.mean()),
                         frame_boundary_phase_step_rad=float(np.median(abs(np.angle(rf[1:, 0, best]*rf[:-1, -1, best].conj())))),
                         within_frame_phase_step_rad=float(np.median(abs(phase[:, :, best]))/2000)))
    n = len(rows)
    p = np.asarray(powers, np.float32).reshape(n, len(FREQ))
    fp = np.asarray(frame_powers, np.float32).reshape(n, len(FREQ))
    logp, logfp = np.log10(p+1e-14), np.log10(fp+1e-14)
    np.savez_compressed(dest,
        iq_frames=np.asarray(iq_all, np.float32).reshape(n, 20, 192, 12, 2),
        valid_chirp_mask=np.asarray(masks, bool).reshape(n, 20, 192),
        phase_rate_frames=np.asarray(phase_all, np.float32).reshape(n, 20, 191, 12),
        valid_phase_mask=np.asarray(phase_masks, bool).reshape(n, 20, 191),
        frame_log_shape=logfp-logfp.mean(-1, keepdims=True), frame_log_power=logfp,
        coherent_log_shape=logp-logp.mean(-1, keepdims=True), coherent_power=p,
        raw_cell_power=np.asarray(absolute, np.float32).reshape(n, 12),
        frequency_hz=FREQ, range_bins=bins, cell_rx=np.repeat(np.arange(4), 3),
        cell_bin=np.tile(bins, 4), observed_time_s=np.arange(20)[:, None]*.1+np.arange(192)[None, :]/2000,
        labels=np.full(n, meta['label'], np.int64))
    quality = dict(complete_frames=nframes, trailing_bytes=size % FRAME_BYTES,
        unused_complete_frames=nframes % 20, clipped_fraction=clipped/max(total, 1),
        zero_int16_fraction=zeros/max(total, 1), range_center_m=float(RANGE[center]),
        range_roi_m=[lo, hi], range_status='geometry_supplied' if distance is not None else 'provisional_geometry_unconfirmed',
        top_roi_peaks_m=RANGE[roi[np.argsort(profile[roi])[-5:][::-1]]].tolist(),
        rejected_windows=rejected, parser_status='assumed_2I2Q_conjugated_frame_aligned_headerless',
        packet_loss_logs_available=False, nominal_duration_s=nframes/10)
    save_json(dest.with_suffix('.json'), dict(metadata=meta, windows=rows, quality=quality,
        frame_spectrum_resolution_hz=2000/192,
        coherent_spectrum_status='conditional_on_cross_frame_coherence; diagnostic_ablation',
        primary_input='frame_log_shape OR frame-wise iq with masks; not BearLLM DCN'))
    return dict(**meta, modality='radar', windows=n, output=dest.name, **{k: quality[k] for k in
                ['complete_frames', 'trailing_bytes', 'nominal_duration_s', 'range_center_m', 'range_status', 'clipped_fraction']})


def check_config(path):
    expected = {'channelCfg': ['15', '7', '0'], 'adcCfg': ['2', '1'],
        'adcbufCfg': ['-1', '0', '1', '1', '1'],
        'profileCfg': '0 77 420 5 80 0 0 49.99 1 256 3430 0 0 48'.split(),
        'chirpCfg': '0 0 0 0 0 0 0 7'.split(), 'frameCfg': '0 0 192 0 100 1 0'.split(),
        'lvdsStreamCfg': '-1 0 1 0'.split()}
    lines = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith('%')]
    for key, values in expected.items():
        if [p[1:] for p in lines if p[0] == key] != [values]:
            raise ValueError(f'Unsupported radar config at {key}; no silent timing reuse')


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data', type=Path, required=True)
    a.add_argument('--output', type=Path, default=ROOT/'results')
    a.add_argument('--config', type=Path, default=ROOT.parents[1]/'research_proposals/contact_radar_alignment_20260919/雷达原始配置.cfg')
    a.add_argument('--distance-m', type=float)
    a.add_argument('--modality', choices=['both', 'contact', 'radar'], default='both')
    args = a.parse_args()
    check_config(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new/empty output directory to preserve prior exports')
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.data.iterdir() if p.suffix in ('.DAT', '.bin') and
                   (args.modality == 'both' or (p.suffix == '.DAT') == (args.modality == 'contact')))
    summary, inputs = [], []
    for path in files:
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        dest = args.output/(path.stem+'.npz')
        print(f'Processing {path.name}', flush=True)
        result = contact_export(path, dest) if path.suffix == '.DAT' else radar_export(path, dest, args.distance_m)
        digest = sha256(path)
        if before != (path.stat().st_size, path.stat().st_mtime_ns):
            raise RuntimeError(f'Raw input changed during processing: {path}')
        inputs.append(dict(file=path.name, bytes=before[0], mtime_ns=before[1], sha256=digest))
        summary.append(result)
        print(f'  windows={result["windows"]}', flush=True)
        save_json(args.output/'summary.json', summary)
    candidates = []
    for c in [r for r in summary if r['modality'] == 'contact']:
        choices = [r for r in summary if r['modality'] == 'radar' and r['state'] == c['state'] and r['baud_candidate'] == c['baud_candidate']]
        if len(choices) == 1:
            r = choices[0]
            delta = (datetime.fromisoformat(c['filename_start_local'])-datetime.fromisoformat(r['filename_start_local'])).total_seconds()
            candidates.append(dict(contact=c['output'], radar=r['output'], label=c['label'],
                candidate_bag_id=r['recording_id'], filename_start_delta_s=delta,
                pairing='candidate_recording_bag_only', synchronized_window_pairs=False,
                condition_confirmed=False, independent_run_confirmed=False))
    save_json(args.output/'recording_bags.json', candidates)
    save_json(args.output/'input_manifest.json', inputs)
    save_json(args.output/'runtime.json', dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        preprocessing_sha256=sha256(Path(__file__)), config_sha256=sha256(args.config),
        command=sys.argv, distance_m=args.distance_m, class_map=LABELS,
        keep_and_bigNormal='label -1; excluded pending confirmation',
        reference_policy='not auto-assigned; require training-only independently recorded matched normal reference',
        time_alignment='none; never infer sampling gaps from baud or file size'))
    print('Complete. Review summary.json, recording_bags.json and each quality record.', flush=True)


if __name__ == '__main__':
    main()
