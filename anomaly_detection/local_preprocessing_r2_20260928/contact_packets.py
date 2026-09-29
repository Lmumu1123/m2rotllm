"""Lossless numeric export of complete contact packets, with explicit gaps.

This module intentionally does not normalize signals, join packets, select an
axis, compute a teacher embedding, or infer sample timestamps from file names.
Input is the semicolon-delimited ASCII format used by this acquisition device.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np


NUMBER = rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RECORD = re.compile(
    rb"(!?)(\d+),(" + NUMBER + rb"),(" + NUMBER + rb"),(" + NUMBER + rb")"
)
SCHEMA_VERSION = "contact_packets_v1"
MAX_TOKEN_BYTES = 65536
MAX_RECORDED_ISSUES = 1000


def _stat_identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _tokens(handle, digest):
    """Yield token, start, termination status and true length; bounded memory."""
    start = 0
    length = 0
    retained = bytearray()
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
        parts = block.split(b";")
        for index, part in enumerate(parts):
            length += len(part)
            if len(retained) < MAX_TOKEN_BYTES:
                retained.extend(part[: MAX_TOKEN_BYTES - len(retained)])
            if index < len(parts) - 1:
                yield bytes(retained), start, True, length
                start += length + 1
                length = 0
                retained.clear()
    if length:
        yield bytes(retained), start, False, length


def _packet_statistics(packet):
    """Stable per-axis summaries; constant-axis kurtosis remains undefined."""
    minimum = packet.min(axis=0)
    maximum = packet.max(axis=0)
    constant = minimum == maximum
    scale = np.max(np.abs(packet), axis=0)
    normalized = packet / np.where(scale > 0, scale, 1.0)
    center = normalized - normalized.mean(axis=0)
    variance = np.mean(center * center, axis=0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = normalized.mean(axis=0) * scale
        std = np.sqrt(variance) * scale
        kurtosis = np.mean(center ** 4, axis=0) / (variance ** 2)
    kurtosis[constant] = np.nan
    return {
        "mean_xyz": mean,
        "std_xyz": std,
        "minimum_xyz": minimum,
        "maximum_xyz": maximum,
        "kurtosis_pearson_xyz": kurtosis,
        "constant_axis": constant,
        "raw_finite": np.isfinite(packet).all(axis=0),
        "summary_finite": np.isfinite(mean) & np.isfinite(std)
        & (np.isfinite(kurtosis) | constant),
    }


def process_contact(
    raw_path,
    output_h5,
    *,
    sample_rate_hz=4000,
    packet_samples=4096,
):
    """Export explicitly complete 1..packet_samples packets, returning QC.

    A malformed record invalidates its current partial packet. A new counter 1
    starts a new candidate. Missing counters are never interpolated and partial
    packets are never combined. Original DAT archival is the caller's job.
    """
    raw_path = Path(raw_path)
    output_h5 = Path(output_h5)
    if isinstance(sample_rate_hz, bool) or not float(sample_rate_hz).is_integer():
        raise ValueError("sample_rate_hz must be a positive integer")
    if isinstance(packet_samples, bool) or not float(packet_samples).is_integer():
        raise ValueError("packet_samples must be a positive integer")
    sample_rate_hz, packet_samples = int(sample_rate_hz), int(packet_samples)
    if sample_rate_hz <= 0 or packet_samples <= 0:
        raise ValueError("sample_rate_hz and packet_samples must be positive")
    if output_h5.exists():
        raise FileExistsError(f"Output already exists: {output_h5}")
    initial_stat = _stat_identity(raw_path)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temporary_name = tempfile.mkstemp(
        prefix=output_h5.name + ".", suffix=".partial", dir=output_h5.parent
    )
    os.close(handle_fd)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    reasons = Counter()
    issues = []
    counter_counts = np.zeros(packet_samples + 1, dtype=np.int64)
    stats = dict(
        valid_syntax_rows=0,
        packet_starts_observed=0,
        complete_packets=0,
        incomplete_candidates=0,
        orphan_valid_rows=0,
        packets_constant_any_axis=0,
        packets_constant_all_axes=0,
        packets_summary_nonfinite=0,
    )
    packet = np.empty((packet_samples, 3), dtype=np.float64)
    count = 0
    packet_start = None
    packet_ordinal = None

    def issue(reason, position, **details):
        reasons[reason] += 1
        if len(issues) < MAX_RECORDED_ISSUES:
            issues.append(dict(byte=int(position), reason=reason, **details))

    def invalidate_candidate():
        nonlocal count, packet_start, packet_ordinal
        if count:
            stats["incomplete_candidates"] += 1
        count, packet_start, packet_ordinal = 0, None, None

    def append(dataset, value, row):
        dataset.resize(row + 1, axis=0)
        dataset[row] = value

    try:
        with h5py.File(temporary, "w") as h5:
            h5.attrs.update(
                schema_version=SCHEMA_VERSION,
                source_name=raw_path.name,
                source_size_bytes=initial_stat[2],
                source_mtime_ns=initial_stat[3],
                sample_rate_hz=sample_rate_hz,
                packet_samples=packet_samples,
                axis_order="x,y,z",
                original_units="unconfirmed original sensor units",
                timing="within-packet only; wall-clock packet gaps unknown",
                timestamp_semantics="no absolute sample timestamps inferred",
                packet_duration_nominal_s=packet_samples / sample_rate_hz,
                numeric_transform="none; ASCII numbers parsed as float64",
                window_available=sample_rate_hz <= packet_samples,
            )
            if sample_rate_hz <= packet_samples:
                first = (packet_samples - sample_rate_hz) // 2
                h5.attrs["window_slice"] = np.array(
                    [first, first + sample_rate_hz], dtype=np.int64
                )
                h5.attrs["window_duration_s"] = 1.0
                h5.attrs["window_slice_semantics"] = "zero-based [start, stop)"
            raw = h5.create_dataset(
                "raw_xyz", shape=(0, packet_samples, 3),
                maxshape=(None, packet_samples, 3), dtype="<f8",
                chunks=(1, packet_samples, 3), compression="gzip",
                compression_opts=4, shuffle=True,
            )
            indices = {
                key: h5.create_dataset(
                    key, shape=(0,), maxshape=(None,), dtype="<i8", chunks=True
                )
                for key in ("packet_ordinal_observed", "byte_start", "byte_end_exclusive")
            }
            summary_group = h5.create_group("packet_stats")
            summary_datasets = {}
            for key in ("mean_xyz", "std_xyz", "minimum_xyz", "maximum_xyz",
                        "kurtosis_pearson_xyz", "constant_axis", "raw_finite",
                        "summary_finite"):
                dtype = "?" if key in ("constant_axis", "raw_finite", "summary_finite") else "<f8"
                summary_datasets[key] = summary_group.create_dataset(
                    key, shape=(0, 3), maxshape=(None, 3), dtype=dtype, chunks=True
                )
            with raw_path.open("rb") as source:
                for token, position, terminated, token_length in _tokens(source, digest):
                    match = RECORD.fullmatch(token) if token_length <= MAX_TOKEN_BYTES else None
                    reason = None
                    if not terminated:
                        reason = "unterminated_tail"
                    elif token_length > MAX_TOKEN_BYTES:
                        reason = "oversized_record"
                    elif match is None:
                        reason = "invalid_record"
                    else:
                        marker, index_bytes, *numbers = match.groups()
                        try:
                            index = int(index_bytes)
                            xyz = np.asarray([float(item) for item in numbers], dtype=np.float64)
                        except (ValueError, OverflowError):
                            reason = "numeric_parse_error"
                        else:
                            if not 1 <= index <= packet_samples or not np.isfinite(xyz).all():
                                reason = "counter_or_nonfinite"
                            elif marker and index != 1:
                                reason = "unexpected_packet_marker"
                    if reason:
                        issue(reason, position,
                              token=token[:100].decode("ascii", errors="replace"),
                              token_length_bytes=int(token_length))
                        invalidate_candidate()
                        continue
                    stats["valid_syntax_rows"] += 1
                    counter_counts[index] += 1
                    if index == 1:
                        if count:
                            issue("packet_restart_before_end", position, retained_rows=count)
                            invalidate_candidate()
                        stats["packet_starts_observed"] += 1
                        packet_start = position
                        packet_ordinal = stats["packet_starts_observed"]
                    if packet_start is None:
                        stats["orphan_valid_rows"] += 1
                        continue
                    if index != count + 1:
                        issue("counter_discontinuity", position,
                              expected=count + 1, observed=index)
                        invalidate_candidate()
                        continue
                    packet[count] = xyz
                    count += 1
                    if count == packet_samples:
                        row = stats["complete_packets"]
                        append(raw, packet, row)
                        append(indices["packet_ordinal_observed"], packet_ordinal, row)
                        append(indices["byte_start"], packet_start, row)
                        append(indices["byte_end_exclusive"], position + token_length + 1, row)
                        packet_stats = _packet_statistics(packet)
                        for key, value in packet_stats.items():
                            append(summary_datasets[key], value, row)
                        stats["packets_constant_any_axis"] += int(packet_stats["constant_axis"].any())
                        stats["packets_constant_all_axes"] += int(packet_stats["constant_axis"].all())
                        stats["packets_summary_nonfinite"] += int(not packet_stats["summary_finite"].all())
                        stats["complete_packets"] += 1
                        count, packet_start, packet_ordinal = 0, None, None
            if count:
                issue("incomplete_packet_at_eof", initial_stat[2], retained_rows=count,
                      packet_byte_start=packet_start)
                invalidate_candidate()
            if _stat_identity(raw_path) != initial_stat:
                raise RuntimeError(f"Source changed during processing: {raw_path}")
            qc = dict(
                schema_version=SCHEMA_VERSION,
                source_name=raw_path.name,
                source_size_bytes=int(initial_stat[2]),
                source_sha256=digest.hexdigest(),
                source_unchanged_during_processing=True,
                output_name=output_h5.name,
                sample_rate_hz=sample_rate_hz,
                packet_samples=packet_samples,
                one_second_window_available=sample_rate_hz <= packet_samples,
                window_slice=([first, first + sample_rate_hz]
                              if sample_rate_hz <= packet_samples else None),
                units="unconfirmed original sensor units",
                timing="packet-relative only; absolute times and packet gaps unknown",
                **stats,
                issue_count=int(sum(reasons.values())),
                issue_reason_counts=dict(reasons),
                issues_recorded=len(issues),
                issues_truncated=sum(reasons.values()) > len(issues),
                issues=issues,
                complete_packet_sample_seconds=stats["complete_packets"] * packet_samples / sample_rate_hz,
                recording_wall_duration_s=None,
                ready_for_contact_model=bool(stats["complete_packets"] > 0
                                             and sample_rate_hz <= packet_samples),
            )
            h5.create_dataset("observed_counter_counts", data=counter_counts)
            h5.create_dataset("qc_json", data=json.dumps(qc, ensure_ascii=False, allow_nan=False),
                              dtype=h5py.string_dtype("utf-8"))
            h5.attrs["source_sha256"] = digest.hexdigest()
            h5.attrs["export_complete"] = True
            h5.flush()
        if output_h5.exists():
            raise FileExistsError(f"Output appeared during processing: {output_h5}")
        os.replace(temporary, output_h5)
        return qc
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
