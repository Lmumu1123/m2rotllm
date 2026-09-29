"""Streaming AWR1843/DCA1000 export; no hardware commands are sent.

Input contract: little-endian, signed 16-bit, headerless *reordered ADC* files,
two-lane [I0,I1,Q0,Q1] grouping and non-interleaved RX blocks per chirp.
``iiqq_neg`` reproduces the project's historical I-iQ convention; ``iiqq_pos``
is an explicit alternative. A file-size check cannot verify these assumptions,
UDP packet loss, or whether capture began at a frame boundary. Retain capture
logs and original files. An ordered list of split files is read as one stream.

The HDF5 ``iq`` dataset is [frame, chirp, rx, range], complex64. Its only signal
operations are per-chirp fast-time mean removal, a symmetric Hann window, and a
512-point range FFT. It is NOT slow-time demeaned, standardized, phase unwrapped,
resampled, converted to acceleration, or interpolated across frame gaps.
``valid_chirp`` only flags wholly zero ADC chirps; it is not a packet-loss audit.
``frame_start_s`` and ``chirp_offset_s`` are nominal CFG-relative timestamps.
Complete nominal two-second window indices are supplied, while every complete
frame, including the final short window, remains in ``iq``. Compact and optional
wide-ROI files use the same schema, with different ``range_bin_indices``.

Dependencies: Python 3.9+, NumPy, h5py. This module does not infer fault labels.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import h5py
import numpy as np


SCHEMA_VERSION = "c2r-radar-roi-v1"
LIGHT_SPEED_M_S = 299792458.0
NFFT = 512


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _one(commands, name):
    rows = commands.get(name, [])
    if len(rows) != 1:
        raise ValueError(f"CFG requires exactly one {name}; found {len(rows)}")
    return rows[0]


def _integer(value, name):
    x = float(value)
    if not math.isfinite(x) or x != int(x):
        raise ValueError(f"{name} must be an integer")
    return int(x)


def parse_cfg(path):
    """Validate the supported ADC layout and derive all timing from the CFG.

    Supports the project's single-TX R1/R2 timing, not TDM-MIMO, multiple
    profiles/chirp definitions, advanced frames, real ADC or HSI headers.
    The R3 39-ms period does not fit this exact-two-second window schema.
    The nominal R2 rate is derived, never silently substituted for a CFG value.
    """
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    commands = {}
    for line in text.splitlines():
        line = line.split("%", 1)[0].split("#", 1)[0].strip()
        if line:
            words = line.split()
            commands.setdefault(words[0], []).append(words[1:])
    if "advFrameCfg" in commands or "subFrameCfg" in commands:
        raise ValueError("Advanced/subframe configurations are unsupported")
    channel = _one(commands, "channelCfg")
    adc = _one(commands, "adcCfg")
    adcbuf = _one(commands, "adcbufCfg")
    profile = _one(commands, "profileCfg")
    chirp = _one(commands, "chirpCfg")
    frame = _one(commands, "frameCfg")
    lvds = _one(commands, "lvdsStreamCfg")
    output_mode = _one(commands, "dfeDataOutputMode")
    if output_mode != ["1"]:
        raise ValueError("Only frame-mode dfeDataOutputMode 1 is supported")
    if not (len(channel) >= 3 and len(adc) == 2 and len(adcbuf) == 5
            and len(profile) >= 14 and len(chirp) == 8 and len(frame) >= 7
            and len(lvds) == 4):
        raise ValueError("CFG has an unsupported command argument count")
    rx_mask, enabled_tx_mask = [int(x) for x in channel[:2]]
    chirp_tx_mask = int(chirp[7])
    if rx_mask != 15 or enabled_tx_mask not in (1, 2, 4):
        raise ValueError("This exporter requires four RX and exactly one enabled TX")
    if chirp_tx_mask != enabled_tx_mask:
        raise ValueError("chirpCfg TX must equal the single channelCfg TX")
    if [int(x) for x in adc] != [2, 1]:
        raise ValueError("Only adcCfg 2 1 (16-bit complex 1x) is supported")
    if [int(x) for x in adcbuf[1:4]] != [0, 1, 1]:
        raise ValueError("Expected adcbufCfg complex output, sampleSwap=1, noninterleaved RX")
    if [int(x) for x in lvds[1:]] != [0, 1, 0]:
        raise ValueError("Only headerless hardware ADC LVDS streaming is supported")
    if any(float(x) != 0 for x in chirp[3:7]):
        raise ValueError("Per-chirp parameter variations are unsupported")
    if int(chirp[0]) != int(chirp[1]) or int(frame[0]) != int(frame[1]):
        raise ValueError("Exactly one repeated chirp index is required")
    if int(chirp[0]) != int(frame[0]) or int(chirp[2]) != int(profile[0]):
        raise ValueError("Profile/chirp/frame indices do not agree")
    adc_samples = _integer(profile[9], "ADC samples")
    if adc_samples != 256:
        raise ValueError("This verified exporter supports exactly 256 ADC samples per chirp")
    chirps_per_frame = _integer(frame[2], "chirps per frame")
    if chirps_per_frame < 2:
        raise ValueError("At least two chirps per frame are required")
    idle_us, adc_start_us, ramp_us = [float(profile[i]) for i in (2, 3, 4)]
    slope_mhz_us, sample_ksps = float(profile[7]), float(profile[10])
    period_s = float(frame[4]) * 1e-3
    if not all(math.isfinite(x) and x > 0 for x in
               (idle_us + ramp_us, ramp_us, slope_mhz_us, sample_ksps, period_s)):
        raise ValueError("Invalid timing, slope or ADC sample rate")
    if idle_us < 0 or adc_start_us < 0:
        raise ValueError("Idle time and ADC start time must be nonnegative")
    chirp_period_s = (idle_us + ramp_us) * 1e-6
    active_s = chirps_per_frame * chirp_period_s
    if active_s > period_s + 1e-10:
        raise ValueError("Chirp block is longer than the frame period")
    adc_rate_hz = sample_ksps * 1e3
    if adc_start_us + adc_samples / adc_rate_hz * 1e6 > ramp_us + 1e-6:
        raise ValueError("ADC acquisition extends beyond the configured ramp")
    window_frames = int(round(2.0 / period_s))
    if window_frames < 1 or not math.isclose(window_frames * period_s, 2.0,
                                            rel_tol=0, abs_tol=1e-8):
        raise ValueError("Frame period must divide two seconds for this export schema")
    grid_per_frame = period_s / chirp_period_s
    grid_integer = int(round(grid_per_frame))
    if not math.isclose(grid_per_frame, grid_integer, rel_tol=0, abs_tol=1e-6):
        grid_integer = None
    frame_bytes = chirps_per_frame * 4 * adc_samples * 2 * 2
    range_bin_m = LIGHT_SPEED_M_S * adc_rate_hz / (2 * slope_mhz_us * 1e12 * NFFT)
    return {
        "cfg_name": path.name, "cfg_sha256": hashlib.sha256(raw).hexdigest(),
        "cfg_text": text, "rx_mask": rx_mask, "rx_indices": [0, 1, 2, 3],
        "tx_mask": chirp_tx_mask, "n_rx": 4, "adc_samples": adc_samples,
        "adc_bits": 16, "complex_adc": True, "adc_rate_hz": adc_rate_hz,
        "start_frequency_ghz": float(profile[1]), "slope_mhz_us": slope_mhz_us,
        "idle_us": idle_us, "adc_start_us": adc_start_us, "ramp_us": ramp_us,
        "hpf_corner_1_code": int(profile[11]), "hpf_corner_2_code": int(profile[12]),
        "rx_gain_db": float(profile[13]), "chirps_per_frame": chirps_per_frame,
        "chirp_period_s": chirp_period_s, "within_frame_sample_rate_hz": 1 / chirp_period_s,
        "frame_period_s": period_s, "frame_rate_hz": 1 / period_s,
        "occupied_chirp_block_s": active_s,
        "observed_first_to_last_chirp_s": (chirps_per_frame - 1) * chirp_period_s,
        "nominal_frame_gap_s": max(0.0, period_s - active_s),
        "nominal_gap_fraction": max(0.0, (period_s - active_s) / period_s),
        "nominal_grid_slots_per_frame": grid_integer,
        "missing_nominal_grid_slots_per_frame": (
            grid_integer - chirps_per_frame if grid_integer is not None else None),
        "average_observed_chirps_per_second": chirps_per_frame / period_s,
        "frame_bytes": frame_bytes, "adc_average_bytes_per_second": frame_bytes / period_s,
        "range_fft_size": NFFT, "range_bin_spacing_m": range_bin_m,
        "sampled_bandwidth_hz": slope_mhz_us * 1e12 * adc_samples / adc_rate_hz,
        "nominal_range_resolution_m": LIGHT_SPEED_M_S / (
            2 * slope_mhz_us * 1e12 * adc_samples / adc_rate_hz),
        "window_duration_s": 2.0, "window_frames": window_frames,
        "configured_finite_frame_count": int(frame[3]),
    }


class _ConcatReader:
    """Read ordered DCA split parts across arbitrary byte boundaries."""

    def __init__(self, paths, hashing=False):
        self.paths = paths
        self.hashing = hashing
        self.index = 0
        self.stream = None
        self.combined = hashlib.sha256()
        self.hashes = [hashlib.sha256() for _ in paths]
        self.counts = [0] * len(paths)

    def read(self, n):
        chunks, left = [], n
        while left and self.index < len(self.paths):
            if self.stream is None:
                self.stream = self.paths[self.index].open("rb")
            block = self.stream.read(left)
            if block:
                chunks.append(block)
                left -= len(block)
                self.counts[self.index] += len(block)
                if self.hashing:
                    self.combined.update(block)
                    self.hashes[self.index].update(block)
            else:
                self.stream.close()
                self.stream = None
                self.index += 1
        return b"".join(chunks)

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def _stat_signature(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)


def _decode(block, cfg, iq_order):
    words = np.frombuffer(block, dtype="<i2")
    values = words.reshape(-1, 4).astype(np.float32)
    z = np.empty((len(values), 2), dtype=np.complex64)
    sign = -1 if iq_order == "iiqq_neg" else 1
    z[:, 0] = values[:, 0] + sign * 1j * values[:, 2]
    z[:, 1] = values[:, 1] + sign * 1j * values[:, 3]
    nframes = len(block) // cfg["frame_bytes"]
    z = z.reshape(nframes, cfg["chirps_per_frame"], cfg["n_rx"], cfg["adc_samples"])
    valid = np.any(words.reshape(nframes, cfg["chirps_per_frame"], -1) != 0, axis=-1)
    z -= z.mean(axis=-1, keepdims=True)
    z *= np.hanning(cfg["adc_samples"]).astype(np.float32)
    spectrum = np.fft.fft(z, n=NFFT, axis=-1).astype(np.complex64)
    saturated = int(np.count_nonzero(np.abs(words.astype(np.int32)) >= 32760))
    return spectrum, valid, saturated, words.size


def _profiles(spectrum, valid):
    """Equal valid-chirp weighting within each frame; preserve all 512 bins."""
    weights = valid[:, :, None, None]
    count = valid.sum(axis=1)
    denom = np.maximum(count, 1)[:, None, None, None]
    frame_mean = (spectrum * weights).sum(axis=1, keepdims=True) / denom
    dynamic = spectrum - frame_mean
    static_power = (np.abs(spectrum).astype(np.float64) ** 2 * weights).sum(axis=(1, 2))
    dynamic_power = (np.abs(dynamic).astype(np.float64) ** 2 * weights).sum(axis=(1, 2))
    divisor = np.maximum(count * spectrum.shape[2], 1)[:, None]
    return static_power / divisor, dynamic_power / divisor


def _create_h5(path, cfg, nframes, bins, output_kind, iq_order):
    handle = h5py.File(path, "x")
    try:
        nc, nr, nb = cfg["chirps_per_frame"], cfg["n_rx"], len(bins)
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["output_kind"] = output_kind
        handle.attrs["cfg_json"] = _json(cfg)
        handle.attrs["iq_order"] = iq_order
        handle.attrs["input_contract"] = "headerless_reordered_adc_2lane_rx_blocks"
        handle.attrs["iq_axes"] = "frame,chirp,rx,range"
        handle.attrs["iq_units"] = "uncalibrated_ADC_counts_after_fast_time_Hann_range_FFT"
        handle.attrs["processing"] = "fast_time_mean_removal;symmetric_Hann_256;FFT_512"
        handle.attrs["timestamp_semantics"] = "nominal_CFG_relative;frame_boundary_assumed;not_hardware_timestamps"
        handle.attrs["valid_chirp_semantics"] = "not_all_ADC_words_zero;not_proof_of_no_packet_loss"
        handle.create_dataset("iq", shape=(nframes, nc, nr, nb), dtype="complex64",
                              chunks=(1, nc, nr, nb), compression="gzip", compression_opts=3,
                              shuffle=True)
        handle.create_dataset("valid_chirp", shape=(nframes, nc), dtype="bool",
                              chunks=(min(256, nframes), nc), compression="gzip")
        for name in ("range_profile_mean_power", "range_profile_dynamic_power"):
            handle.create_dataset(name, shape=(nframes, 256), dtype="float32",
                                  chunks=(min(128, nframes), 256), compression="gzip", shuffle=True)
        handle.create_dataset("range_bin_indices", data=np.asarray(bins, dtype=np.int32))
        handle.create_dataset("range_m", data=np.asarray(bins) * cfg["range_bin_spacing_m"])
        handle.create_dataset("range_profile_m", data=np.arange(256) * cfg["range_bin_spacing_m"])
        handle.create_dataset("rx_indices", data=np.asarray(cfg["rx_indices"], dtype=np.int8))
        handle.create_dataset("frame_index", data=np.arange(nframes, dtype=np.int64))
        handle.create_dataset("frame_start_s", data=np.arange(nframes) * cfg["frame_period_s"])
        handle.create_dataset("chirp_offset_s", data=np.arange(nc) * cfg["chirp_period_s"])
        nf = cfg["window_frames"]
        first = np.arange(nframes // nf, dtype=np.int64) * nf
        handle.create_dataset("window_first_frame", data=first)
        handle.create_dataset("window_frame_count", data=np.full(len(first), nf, dtype=np.int32))
        handle.create_dataset("window_start_s", data=first * cfg["frame_period_s"])
        return handle
    except Exception:
        handle.close()
        raise


def process_radar(raw_path, cfg_path, output_h5, distance_cm, *, iq_order="iiqq_neg",
                  roi_half_width_m=0.12, compact_bins=5, batch_frames=4,
                  archive_h5=None, max_frames=None):
    """Export one recording (or an explicitly ordered split-file list).

    ``max_frames`` creates a clearly marked pilot export and hashes only the
    processed byte prefix. Existing output files are refused. Returns a JSON-safe
    QC dict, also embedded as ``qc_json`` in both HDF5 files. No class labels are
    read, no scaler is fitted, and quality flags never silently delete recordings.
    A crash may leave a UUID-suffixed temporary file, never a success-looking file.
    """
    cfg = parse_cfg(cfg_path)
    paths = ([Path(raw_path)] if isinstance(raw_path, (str, os.PathLike))
             else [Path(p) for p in raw_path])
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Provide at least one distinct, explicitly ordered raw part")
    signatures = [_stat_signature(p) for p in paths]
    source_bytes = sum(s[0] for s in signatures)
    available, tail_bytes = divmod(source_bytes, cfg["frame_bytes"])
    if available < 1:
        raise ValueError("No complete ADC frame; check input format and CFG")
    if iq_order not in ("iiqq_neg", "iiqq_pos"):
        raise ValueError("iq_order must be iiqq_neg or iiqq_pos")
    if not isinstance(batch_frames, int) or batch_frames < 1:
        raise ValueError("batch_frames must be a positive integer")
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    used_frames = min(available, max_frames) if max_frames is not None else available
    is_partial = used_frames < available
    if not isinstance(compact_bins, int) or compact_bins < 3 or compact_bins % 2 != 1:
        raise ValueError("compact_bins must be an odd integer >= 3")
    distance_m = float(distance_cm) / 100.0
    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("distance_cm must be finite and positive")
    if not math.isfinite(roi_half_width_m) or roi_half_width_m <= 0:
        raise ValueError("roi_half_width_m must be finite and positive")
    ranges = np.arange(256) * cfg["range_bin_spacing_m"]
    roi_min = max(ranges[1], distance_m - roi_half_width_m)
    roi_max = min(ranges[-1], distance_m + roi_half_width_m)
    wide_bins = np.flatnonzero((ranges >= roi_min) & (ranges <= roi_max))
    if len(wide_bins) < compact_bins:
        raise ValueError("Physical ROI is too small/out of range for the compact bin count")
    output_h5 = Path(output_h5)
    targets = [output_h5] + ([Path(archive_h5)] if archive_h5 is not None else [])
    if len({p.resolve() for p in targets}) != len(targets):
        raise ValueError("Compact and archive output paths must differ")
    source_resolved = {p.resolve() for p in paths} | {Path(cfg_path).resolve()}
    if any(p.resolve() in source_resolved for p in targets):
        raise ValueError("Output must not overwrite a raw source or CFG")
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    # Fixed-duration, label-free selection; frame means are removed ONLY here.
    pilot_frames = min(used_frames, int(math.ceil(1.0 / cfg["frame_period_s"])))
    pilot_static, pilot_dynamic = np.zeros(NFFT), np.zeros(NFFT)
    pilot_nonempty = 0
    with contextlib.closing(_ConcatReader(paths)) as reader:
        for first in range(0, pilot_frames, batch_frames):
            count = min(batch_frames, pilot_frames - first)
            block = reader.read(count * cfg["frame_bytes"])
            if len(block) != count * cfg["frame_bytes"]:
                raise IOError("Source shortened during pilot read")
            spectrum, valid, _, _ = _decode(block, cfg, iq_order)
            static, dynamic = _profiles(spectrum, valid)
            nonempty = valid.any(axis=1)
            pilot_static += static[nonempty].sum(axis=0)
            pilot_dynamic += dynamic[nonempty].sum(axis=0)
            pilot_nonempty += int(nonempty.sum())
    pilot_static /= max(pilot_nonempty, 1)
    pilot_dynamic /= max(pilot_nonempty, 1)
    flags = []
    if pilot_nonempty == 0 or pilot_dynamic[wide_bins].max() <= 0:
        center = int(wide_bins[np.argmin(abs(ranges[wide_bins] - distance_m))])
        flags.append("no_pilot_dynamic_energy_center_uses_declared_distance")
    else:
        center = int(wide_bins[np.argmax(pilot_dynamic[wide_bins])])
    start = max(int(wide_bins[0]), min(center - compact_bins // 2,
                                     int(wide_bins[-1]) - compact_bins + 1))
    selected = np.arange(start, start + compact_bins, dtype=np.int32)
    if center in (int(wide_bins[0]), int(wide_bins[-1])):
        flags.append("selected_peak_at_physical_ROI_edge_review_range_profile")
    mirrored = (-wide_bins) % NFFT
    pos_dynamic, neg_dynamic = float(pilot_dynamic[wide_bins].sum()), float(pilot_dynamic[mirrored].sum())
    pos_static, neg_static = float(pilot_static[wide_bins].sum()), float(pilot_static[mirrored].sum())
    dynamic_ratio = neg_dynamic / max(pos_dynamic, 1e-30)
    static_ratio = neg_static / max(pos_static, 1e-30)
    if dynamic_ratio > 4 or static_ratio > 4:
        flags.append("negative_frequency_mirror_dominates_check_IQ_convention")
    if tail_bytes:
        flags.append("trailing_incomplete_frame_retained_only_in_raw_source")
    if is_partial:
        flags.append("pilot_only_partial_recording_export")
    if used_frames < cfg["window_frames"]:
        flags.append("no_complete_two_second_window")

    temps = [p.with_name(p.name + ".partial-" + uuid.uuid4().hex) for p in targets]
    handles, reader = [], None
    saturated_words = adc_words = zero_chirps = 0
    try:
        handles.append(_create_h5(temps[0], cfg, used_frames, selected, "compact", iq_order))
        if archive_h5 is not None:
            handles.append(_create_h5(temps[1], cfg, used_frames, wide_bins, "wide_physical_roi", iq_order))
        for handle in handles:
            handle.create_dataset("pilot_range_m", data=np.arange(256) * cfg["range_bin_spacing_m"])
            handle.create_dataset("pilot_positive_mean_power", data=pilot_static[:256])
            handle.create_dataset("pilot_positive_dynamic_power", data=pilot_dynamic[:256])
            handle.create_dataset("pilot_negative_mirror_mean_power", data=pilot_static[(-np.arange(256)) % NFFT])
            handle.create_dataset("pilot_negative_mirror_dynamic_power", data=pilot_dynamic[(-np.arange(256)) % NFFT])
        reader = _ConcatReader(paths, hashing=True)
        for first in range(0, used_frames, batch_frames):
            count = min(batch_frames, used_frames - first)
            block = reader.read(count * cfg["frame_bytes"])
            if len(block) != count * cfg["frame_bytes"]:
                raise IOError("Source shortened during export")
            spectrum, valid, clipped, nwords = _decode(block, cfg, iq_order)
            static, dynamic = _profiles(spectrum, valid)
            saturated_words += clipped
            adc_words += nwords
            zero_chirps += int((~valid).sum())
            sl = slice(first, first + count)
            for handle, bins in zip(handles, [selected, wide_bins]):
                handle["iq"][sl] = spectrum[..., bins]
                handle["valid_chirp"][sl] = valid
                handle["range_profile_mean_power"][sl] = static[:, :256].astype(np.float32)
                handle["range_profile_dynamic_power"][sl] = dynamic[:, :256].astype(np.float32)
        if not is_partial:
            tail = reader.read(tail_bytes)
            if len(tail) != tail_bytes or reader.read(1):
                raise IOError("Source size changed during export")
        reader.close()
        if [_stat_signature(p) for p in paths] != signatures:
            raise IOError("Source file metadata changed during processing; outputs not committed")
        if parse_cfg(cfg_path)["cfg_sha256"] != cfg["cfg_sha256"]:
            raise IOError("CFG changed during processing; outputs not committed")
        if zero_chirps:
            flags.append("all_zero_ADC_chirps_flagged_review_capture_loss_logs")
        saturation_fraction = saturated_words / max(adc_words, 1)
        if saturation_fraction > 1e-4:
            flags.append("ADC_saturation_candidate_fraction_above_0.0001")
        peak_order = wide_bins[np.argsort(pilot_dynamic[wide_bins])[::-1][:5]]
        part_sources = []
        for i, path in enumerate(paths):
            part_complete = reader.counts[i] == signatures[i][0]
            part_sources.append({"path": str(path), "name": path.name,
                                 "source_bytes": signatures[i][0], "hashed_bytes": reader.counts[i],
                                 "hash_scope": "complete_file" if part_complete else "processed_prefix",
                                 "sha256": reader.hashes[i].hexdigest() if reader.counts[i] else None})
        qc = {
            "schema_version": SCHEMA_VERSION, "status": "pilot_partial" if is_partial else "complete_export",
            "source_parts": part_sources, "source_bytes": source_bytes,
            "source_sha256": reader.combined.hexdigest(),
            "source_hash_scope": "processed_prefix" if is_partial else "concatenated_complete_sources",
            "hashed_bytes": sum(reader.counts), "cfg_sha256": cfg["cfg_sha256"],
            "input_contract": "headerless_reordered_ADC;2lane_IIQQ;RX_blocks;frame_boundary_assumed",
            "iq_order": iq_order, "capture_packet_loss_status": "unknown_check_DCA_logs",
            "frames_available": available, "frames_exported": used_frames,
            "trailing_incomplete_frame_bytes": tail_bytes,
            "nominal_export_duration_s": used_frames * cfg["frame_period_s"],
            "nominal_last_observed_chirp_s": (used_frames - 1) * cfg["frame_period_s"]
                + cfg["observed_first_to_last_chirp_s"],
            "within_frame_sample_rate_hz": cfg["within_frame_sample_rate_hz"],
            "frame_rate_hz": cfg["frame_rate_hz"],
            "frame_period_s": cfg["frame_period_s"],
            "chirp_period_s": cfg["chirp_period_s"],
            "chirps_per_frame": cfg["chirps_per_frame"],
            "nominal_frame_gap_s": cfg["nominal_frame_gap_s"],
            "nominal_gap_fraction": cfg["nominal_gap_fraction"],
            "adc_average_bytes_per_second": cfg["adc_average_bytes_per_second"],
            "complete_two_second_windows": used_frames // cfg["window_frames"],
            "retained_tail_frames_after_last_full_window": used_frames % cfg["window_frames"],
            "all_zero_ADC_chirps": zero_chirps,
            "valid_chirp_fraction": 1 - zero_chirps / (used_frames * cfg["chirps_per_frame"]),
            "ADC_near_limit_word_count": saturated_words, "ADC_word_count": adc_words,
            "ADC_near_limit_fraction": saturation_fraction,
            "declared_distance_cm": float(distance_cm), "physical_roi_bounds_m": [roi_min, roi_max],
            "roi_half_width_m": float(roi_half_width_m), "wide_roi_bin_count": len(wide_bins),
            "selected_center_bin": center, "selected_center_range_m": float(ranges[center]),
            "compact_range_bin_indices": selected.tolist(),
            "compact_range_bounds_m": [float(ranges[selected[0]]), float(ranges[selected[-1]])],
            "roi_selection_rule": "max_first_1s_per_frame_demeaned_power_within_declared_physical_ROI",
            "roi_selection_pilot_frames": pilot_frames,
            "roi_selection_pilot_nominal_seconds": pilot_frames * cfg["frame_period_s"],
            "pilot_negative_to_positive_dynamic_ratio": dynamic_ratio,
            "pilot_negative_to_positive_mean_ratio": static_ratio,
            "pilot_top_dynamic_bins": [{"bin": int(b), "range_m": float(ranges[b]),
                                        "relative_ADC_FFT_power": float(pilot_dynamic[b])} for b in peak_order],
            "compact_IQ_uncompressed_bytes": used_frames * cfg["chirps_per_frame"] * 4 * len(selected) * 8,
            "wide_IQ_uncompressed_bytes": used_frames * cfg["chirps_per_frame"] * 4 * len(wide_bins) * 8,
            "compact_output": output_h5.name,
            "wide_output": str(archive_h5) if archive_h5 is not None else None,
            "quality_flags": flags,
            "quality_note": "Flags request review; no class-dependent filtering and no files/windows silently discarded.",
        }
        for handle in handles:
            handle.attrs["qc_json"] = _json(qc)
            handle.attrs["export_complete"] = True
            handle.flush()
            handle.close()
        handles = []
        # Each file is atomically committed. A system crash between the two renames
        # can leave only one final file; the manifest/CLI must check both outputs.
        for temp, target in zip(temps, targets):
            if target.exists():
                raise FileExistsError(f"Output appeared during processing: {target}")
            os.replace(temp, target)
        return qc
    finally:
        if reader is not None:
            reader.close()
        for handle in handles:
            handle.close()
        for temp in temps:
            if temp.exists():
                temp.unlink()
