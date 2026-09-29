"""Fixed, label-free frame-local radar representations for the September R2 data.

Reads immutable exported complex range-FFT data. No model, global scaler,
cross-frame interpolation, cross-frame phase differentiation, or label-dependent
ROI selection is used. All nominal two-second windows remain in the output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import h5py
import numpy as np
from scipy import fft

ROOT = Path(__file__).resolve().parent
CLASSES = ["normal", "inBroken", "outBroken", "roll"]
FS = 5000.0
NFFT = 1024
FRAMES = 50
EDGES = np.linspace(5.0, 2000.0, 129)
FREQUENCIES = np.fft.rfftfreq(NFFT, 1 / FS)
NUISANCE_COLUMNS = [
    "log10_mean_raw_power", "log10_median_raw_power",
    "log10_q10_raw_power", "log10_q90_raw_power",
    "log10_mean_dynamic_power", "log10_median_dynamic_power",
    *[f"log10_mean_raw_power_rx{i}" for i in range(4)],
    "selected_range_mean_m", "declared_distance_m",
]


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def band_weights(f=FREQUENCIES, edges=EDGES):
    """Exact integral of piecewise-linear spectral interpolation in each band."""
    weights = np.zeros((len(edges) - 1, len(f)), dtype=np.float64)
    for band, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        nodes = np.r_[left, f[(f > left) & (f < right)], right]
        nodal_weights = np.zeros(len(nodes))
        widths = np.diff(nodes)
        nodal_weights[:-1] += widths / 2
        nodal_weights[1:] += widths / 2
        index = np.clip(np.searchsorted(f, nodes, side="right") - 1, 0, len(f) - 2)
        frac = (nodes - f[index]) / (f[index + 1] - f[index])
        np.add.at(weights[band], index, nodal_weights * (1 - frac))
        np.add.at(weights[band], index + 1, nodal_weights * frac)
        weights[band] /= right - left
    assert np.allclose(weights.sum(1), 1)
    return weights


WEIGHTS = band_weights()


def spectral_shape(power):
    pooled = WEIGHTS @ power
    # The floor is numerical protection, not fitted to a class or data split.
    logged = np.log10(np.maximum(pooled, 1e-20))
    return (logged - logged.mean()).astype(np.float32)


def _folded_psd(x):
    """Average complete-frame periodograms; median across correlated cells."""
    n = x.shape[1]
    taper = np.hanning(n).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    transformed = fft.fft(x * taper[None, :, None], n=NFFT, axis=1, workers=1)
    power = np.abs(transformed) ** 2 / (FS * np.sum(taper * taper))
    folded = power[:, :NFFT // 2 + 1].copy()
    folded[:, 1:-1] += power[:, -1:NFFT // 2:-1]
    return np.median(folded.mean(axis=0), axis=-1)


def features_for_window(iq, valid, range_m, declared_distance_cm):
    """No operation ever joins samples from adjacent radar frames."""
    assert iq.ndim == 4 and iq.shape[1:3] == (192, 4)
    assert valid.shape == iq.shape[:2]
    finite_frames = np.isfinite(iq).all(axis=(1, 2, 3))
    complete = valid.all(axis=1) & finite_frames
    quality = {
        "complete_frame_fraction": float(complete.mean()),
        "valid_chirp_fraction": float(valid.mean()),
        "nonfinite_frame_count": int((~finite_frames).sum()),
        "spectral_estimate_available": bool(complete.any()),
    }
    if not complete.any():
        quality.update(near_zero_amplitude_fraction=1., near_pi_increment_fraction=0.,
                       usable_spatial_cell_fraction=0., median_raw_amplitude=0.,
                       max_to_median_raw_amplitude=0.)
        return np.zeros(128, np.float32), np.zeros(128, np.float32), np.zeros(12, np.float32), quality, np.zeros(len(FREQUENCIES))

    raw = iq[complete].astype(np.complex128)
    x = raw.reshape(len(raw), 192, -1)
    amplitude = np.abs(x)
    scale = np.median(amplitude, axis=(0, 1))
    good_cells = np.isfinite(scale) & (scale > 1e-12)
    quality["usable_spatial_cell_fraction"] = float(good_cells.mean())
    quality["near_zero_amplitude_fraction"] = float(
        (amplitude < .1 * np.maximum(scale, 1e-12)[None, None, :]).mean())
    quality["median_raw_amplitude"] = float(np.median(amplitude))
    quality["max_to_median_raw_amplitude"] = float(
        amplitude.max() / max(float(np.median(amplitude)), 1e-12))
    normalized = x[:, :, good_cells] / scale[good_cells][None, None, :] if good_cells.any() else np.zeros((*x.shape[:2], 1), complex)
    complex_power = _folded_psd(normalized)
    # Adjacent phase increment is computed strictly within each frame. No unwrap.
    increments = np.angle(normalized[:, 1:] * normalized[:, :-1].conj())
    quality["near_pi_increment_fraction"] = float((np.abs(increments) > .8 * np.pi).mean())
    phase_power = _folded_psd(increments)
    raw_power = np.abs(raw) ** 2
    dynamic_power = np.abs(raw - raw.mean(axis=1, keepdims=True)) ** 2
    positive = [raw_power.mean(), np.median(raw_power), np.quantile(raw_power, .1),
                np.quantile(raw_power, .9), dynamic_power.mean(), np.median(dynamic_power),
                *raw_power.mean(axis=(0, 1, 3)).tolist()]
    nuisance = np.r_[np.log10(np.maximum(positive, 1e-20)), np.mean(range_m), declared_distance_cm / 100].astype(np.float32)
    return spectral_shape(complex_power), spectral_shape(phase_power), nuisance, quality, complex_power


def fixed_geometry_indices(handle, distance_cm):
    """Five contiguous bins centered closest to the declared physical distance."""
    ranges = handle["range_m"][:]
    if len(ranges) < 5:
        raise ValueError("Wide ROI must have at least five bins")
    middle = int(np.argmin(np.abs(ranges - distance_cm / 100)))
    start = max(0, min(middle - 2, len(ranges) - 5))
    return np.arange(start, start + 5)


def check_cfg(handle):
    cfg = json.loads(handle.attrs["cfg_json"])
    if not (cfg["chirps_per_frame"] == 192 and cfg["n_rx"] == 4
            and np.isclose(cfg["within_frame_sample_rate_hz"], FS)
            and np.isclose(cfg["frame_period_s"], .04)
            and cfg["window_frames"] == FRAMES):
        raise ValueError("Expected declared R2 5000 Hz, 192 chirps, 40 ms frame, 50-frame windows")
    if not handle.attrs.get("export_complete", False):
        raise ValueError("Incomplete HDF5 export")
    return cfg


def extract(data_root, output, limit=None):
    begin = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((data_root / "upload/radar").glob("*/radar.h5"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError("No radar.h5 files")
    arrays = {k: [] for k in ["complex_shape", "phase_shape", "nuisance",
                             "geometry_complex_shape", "geometry_phase_shape"]}
    rows, records, input_manifest, curves = [], [], [], []
    for index, path in enumerate(paths):
        doc = json.loads((path.parent / "recording.json").read_text())
        metadata, qc = doc["metadata"], doc["radar"]
        rid = doc["recording_id"]
        distance = float(metadata["distance_cm"])
        wide_path = data_root / "wide_cache_local" / f"{rid}.h5"
        before = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in (path, wide_path)}
        record_curves, record_quality = [], []
        with h5py.File(path, "r") as compact, h5py.File(wide_path, "r") as wide:
            cfg = check_cfg(compact)
            assert check_cfg(wide)["cfg_sha256"] == cfg["cfg_sha256"]
            starts = compact["window_first_frame"][:]
            assert np.array_equal(compact["window_frame_count"][:], np.full(len(starts), FRAMES))
            assert np.array_equal(starts, np.arange(compact["iq"].shape[0] // FRAMES) * FRAMES)
            ranges = compact["range_m"][:]
            wide_indices = fixed_geometry_indices(wide, distance)
            fixed_ranges = wide["range_m"][:][wide_indices]
            for start in starts:
                sl = slice(int(start), int(start) + FRAMES)
                valid = compact["valid_chirp"][sl]
                features = features_for_window(compact["iq"][sl], valid, ranges, distance)
                geometric_iq = wide["iq"][sl][..., wide_indices]
                fixed = features_for_window(geometric_iq, wide["valid_chirp"][sl], fixed_ranges, distance)
                for key, value in zip(["complex_shape", "phase_shape", "nuisance"], features[:3]):
                    arrays[key].append(value)
                arrays["geometry_complex_shape"].append(fixed[0])
                arrays["geometry_phase_shape"].append(fixed[1])
                row = dict(row=len(rows), recording_id=rid, contact_id=doc["contact_id"],
                           label=CLASSES.index(metadata["label"]), state=metadata["label"],
                           rpm=int(metadata["rpm"]), distance_cm=distance, repeat=int(metadata["repeat"]),
                           window_first_frame=int(start), window_start_s=float(start * .04),
                           selected_center_range_m=qc["selected_center_range_m"],
                           selected_range_mean_m=float(ranges.mean()),
                           fixed_range_mean_m=float(fixed_ranges.mean()),
                           ADC_near_limit_fraction=qc["ADC_near_limit_fraction"],
                           roi_edge_flag="selected_peak_at_physical_ROI_edge_review_range_profile" in qc["quality_flags"],
                           mirror_flag="negative_frequency_mirror_dominates_check_IQ_convention" in qc["quality_flags"],
                           **features[3])
                row.update({f"geometry_{key}": val for key, val in fixed[3].items()})
                rows.append(row)
                record_quality.append(features[3])
                record_curves.append(features[4])
            records.append(dict(recording_id=rid, contact_id=doc["contact_id"],
                                label=CLASSES.index(metadata["label"]), state=metadata["label"],
                                rpm=int(metadata["rpm"]), distance_cm=distance, repeat=int(metadata["repeat"]),
                                n_frames=compact["iq"].shape[0], n_windows=len(starts),
                                discarded_full_frames=0,
                                retained_tail_frames_not_in_full_window=compact["iq"].shape[0] % FRAMES,
                                selected_center_range_m=qc["selected_center_range_m"],
                                fixed_range_mean_m=float(fixed_ranges.mean()),
                                ADC_near_limit_fraction=qc["ADC_near_limit_fraction"],
                                quality_flags=";".join(qc["quality_flags"]),
                                **{f"mean_{key}": float(np.mean([q[key] for q in record_quality])) for key in record_quality[0]}))
            curves.append(np.median(record_curves, axis=0))
        for p in (path, wide_path):
            assert before[str(p)] == [p.stat().st_size, p.stat().st_mtime_ns], "Input changed"
        input_manifest.append(dict(recording_id=rid, sources=before,
                                   recording_json_sha256=hashlib.sha256((path.parent / "recording.json").read_bytes()).hexdigest(),
                                   source_adc_sha256=qc["source_sha256"], cfg_sha256=cfg["cfg_sha256"]))
        print(f"{index+1}/{len(paths)} {rid}: {len(starts)} windows", flush=True)

    output_arrays = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    for key in ["rpm", "distance_cm", "repeat", "recording_id", "contact_id", "window_first_frame"]:
        output_arrays[key] = np.asarray([r[key] for r in rows])
    output_arrays["labels"] = np.asarray([r["label"] for r in rows], dtype=np.int64)
    output_arrays["band_edges_hz"] = EDGES
    output_arrays["quality_spectral_available"] = np.asarray([r["spectral_estimate_available"] for r in rows], dtype=bool)
    assert all(np.isfinite(v).all() for v in output_arrays.values() if v.dtype.kind in "fciu")
    np.savez_compressed(output / "features.npz", **output_arrays)
    np.savez_compressed(output / "recording_spectra.npz", frequency_hz=FREQUENCIES,
                        complex_power=np.asarray(curves), recording_id=np.asarray([r["recording_id"] for r in records]))
    save_csv(output / "metadata.csv", rows)
    save_csv(output / "recording_quality.csv", records)
    save_json(output / "input_manifest.json", input_manifest)
    save_json(output / "feature_schema.json", dict(
        schema="r2_frame_local_v1", main_feature="complex_shape", labels=dict(zip(CLASSES, range(4))),
        default_secondary="phase_shape", geometry_sensitivity=["geometry_complex_shape", "geometry_phase_shape"],
        source=str(data_root), array_shapes={k: list(v.shape) for k, v in output_arrays.items()},
        radar_fs_within_frame_hz=FS, frame_period_s=.04, frame_chirps=192, window_frames=FRAMES,
        window_wall_seconds=2., observed_chirps_per_full_window=FRAMES * 192, nominal_missing_fraction=.04,
        fft_size=NFFT, fft_evaluation_spacing_hz=FS / NFFT,
        frame_unwindowed_rayleigh_spacing_hz=FS / 192,
        phase_increment_unwindowed_rayleigh_spacing_hz=FS / 191,
        spectral_resolution_note="Zero padding and 128 bands do not create 128 independent frequency measurements; Hann broadens the main lobe.",
        band_edges_hz=EDGES.tolist(), band_pooling="mean of linearly interpolated PSD integrated exactly over each band; log10; subtract window mean",
        spectral_estimation="within-complete-frame mean removal, Hann, 1024 FFT, fold +/- complex power; average frames, median spatial cells",
        spatial_normalization="divide each spatial cell by its original amplitude median in that window before frame demeaning",
        phase_branch="within-frame angle(z[t] * conj(z[t-1])); mean removal; never across gaps; no absolute displacement calibration",
        geometry_branch="5 wide-ROI bins fixed nearest declared physical distance; same PSD and features; no label-dependent or metric-selected ROI",
        nuisance_columns=NUISANCE_COLUMNS,
        nuisance_caveat="Absolute amplitude can contain both fault signal and acquisition effects; this is a diagnostic baseline, not a proof of pure nuisance.",
        scalar_fit="none; train-fold StandardScaler belongs to downstream experiment",
        filtering="No recordings/windows removed. Incomplete/nonfinite frames omitted from that window's PSD only; fully unobservable windows use zeros plus false quality flag.",
        contact_association="shared continuous-session reference, not synchronized window pairs",
        grouping_caveat="repeat indices are per-condition file-time ordering, not confirmed independent campaigns; only 12 contact sessions",
        quality_thresholds={"near_zero_amplitude": "<0.1 x per-cell window median", "near_pi_increment": "abs(angle) > 0.8*pi"},
        elapsed_seconds=time.monotonic()-begin,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(f"Finished: {len(rows)} windows, {len(records)} recordings in {time.monotonic()-begin:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/media/nas_users/huangyating/data"))
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    extract(args.data_root, args.output, args.limit)
