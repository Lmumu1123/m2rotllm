"""Audit every MBHM metadata/corpus record and (by default) every signal value.

The published signals have already undergone DCN; no preprocessing is applied.
The audit distinguishes data integrity from the narrower healthy-reference
coverage of the upstream training/evaluation loader.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time

import h5py
import numpy as np
import pandas as pd

from download_demo_assets import verify


CORPUS_LABEL_NAMES = [
    "Fault-Free", "Minor Inner Ring Fault", "Moderate Inner Ring Fault", "Severe Inner Ring Fault",
    "Minor Ball Fault", "Moderate Ball Fault", "Severe Ball Fault",
    "Minor Outer Ring Fault", "Moderate Outer Ring Fault", "Severe Outer Ring Fault",
]


def counts(values):
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/mbhm_audit.json"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    started = time.monotonic()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "scope": "metadata_only" if args.metadata_only else "complete",
        "checks": {}, "warnings": [],
    }
    checks = report["checks"]
    manifest = json.loads((args.data_dir / "mbhm_manifest.json").read_text())
    report["dataset_revision"] = manifest["revision"]
    report["file_verification"] = {}
    for item in manifest["files"]:
        name = item["rfilename"]
        if args.metadata_only and name == "data.hdf5":
            continue
        valid = verify(args.data_dir / name, item)
        report["file_verification"][name] = {
            "bytes": item["size"], "verified": valid,
            "hash_type": "sha256" if "lfs" in item else "git_blob_sha1",
            "expected_hash": item["lfs"]["sha256"] if "lfs" in item else item["blobId"],
        }
        print(f"Hash verified={valid}: {name}", flush=True)
    checks["all_requested_file_hashes_match"] = all(
        item["verified"] for item in report["file_verification"].values())

    with sqlite3.connect(f"file:{args.data_dir / 'metadata.sqlite'}?mode=ro", uri=True) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchall()
        checks["sqlite_integrity"] = integrity == [("ok",)]
        checks["sqlite_foreign_keys"] = not conn.execute("PRAGMA foreign_key_check").fetchall()
        tables = dict(conn.execute("SELECT name,sql FROM sqlite_master WHERE type='table'"))
        df = pd.read_sql_query(
            "SELECT f.file_id,f.condition_id,f.label,c.dataset,c.code AS bearing_code,"
            "c.channel,c.rpm,c.load FROM file_info f JOIN condition c USING(condition_id) "
            "ORDER BY file_id", conn)
        total = conn.execute("SELECT COUNT(*) FROM file_info").fetchone()[0]
        condition_count = conn.execute("SELECT COUNT(*) FROM condition").fetchone()[0]
        labels = dict(conn.execute("SELECT label,note FROM label_note ORDER BY label"))
    checks["all_file_rows_join_conditions"] = len(df) == total
    checks["file_ids_contiguous_from_zero"] = bool(np.array_equal(df.file_id, np.arange(total)))
    checks["valid_label_ids"] = bool(df.label.isin(labels).all())
    parquet = pd.read_parquet(args.data_dir / "metadata.parquet").sort_values("file_id").reset_index(drop=True)
    checks["sqlite_parquet_identical"] = list(parquet.columns) == list(df.columns) and parquet.equals(df)
    healthy_conditions = set(df.loc[df.label == 0, "condition_id"].tolist())
    eligible = df.condition_id.isin(healthy_conditions)
    excluded = df.loc[~eligible]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    excluded_path = args.output.parent / "mbhm_excluded_missing_reference.csv"
    excluded.to_csv(excluded_path, index=False)
    eligible_rows = df.loc[eligible].reset_index(drop=True)
    split_names = np.where(np.arange(len(eligible_rows)) % 10 < 7, "train",
                           np.where(np.arange(len(eligible_rows)) % 10 < 9, "val", "test"))
    source_details = {}
    for source, subset in df.groupby("dataset"):
        subset_eligible = subset.condition_id.isin(healthy_conditions)
        source_details[source] = {
            "samples": len(subset), "conditions": subset.condition_id.nunique(),
            "label_counts": counts(subset.label),
            "eligible_label_counts": counts(subset.loc[subset_eligible, "label"]),
            "excluded_label_counts": counts(subset.loc[~subset_eligible, "label"]),
            "same_condition_healthy_reference_samples": int(subset_eligible.sum()),
            "missing_same_condition_healthy_reference_samples": int((~subset_eligible).sum()),
        }
    report["metadata"] = {
        "schema": tables, "sample_count": total, "condition_count": condition_count,
        "source_count": int(df.dataset.nunique()), "sources": source_details,
        "label_names": labels, "label_counts": counts(df.label),
        "parquet_shape": list(parquet.shape),
    }
    report["official_reference_coverage"] = {
        "eligible_unique_samples": int(eligible.sum()),
        "excluded_unique_samples": int((~eligible).sum()),
        "eligible_conditions": len(healthy_conditions),
        "excluded_conditions": int(excluded.condition_id.nunique()),
        "excluded_by_source": counts(excluded.dataset),
        "excluded_label_counts": counts(excluded.label),
        "eligible_sources_with_only_healthy_queries": [
            source for source, detail in source_details.items()
            if set(detail["eligible_label_counts"]) == {"0"}
        ],
        "excluded_manifest": str(excluded_path.resolve()),
        "unique_samples_per_official_split": counts(split_names),
        "pairs_per_official_split": {k: v * 3 for k, v in counts(split_names).items()},
        "split_rule": "Enumerate eligible samples in file_id order; index%10<7 train, <9 val, otherwise test. Upstream creates 3 random same-condition healthy reference pairs per sample.",
        "published_fixed_split_or_reference_manifest": False,
        "reference_pool_rule": "Healthy references are selected from the entire same-condition pool before splitting; references are not constrained to the query split.",
    }
    if len(excluded):
        report["warnings"].append(
            f"The upstream healthy-reference rule excludes {len(excluded)} samples. "
            "Data integrity coverage includes them, but official paired model evaluation cannot. "
            "Any cross-condition reference fallback must be reported separately.")
    report["warnings"].append(
        "The release contains no fixed train/val/test or reference-pair manifest; "
        "regenerated splits and references cannot establish independence from published model training.")

    corpus = json.loads((args.data_dir / "corpus.json").read_text())
    by_id = df.set_index("file_id")
    violations = []
    for row in corpus:
        problems = []
        for field in ("id", "task_id", "instruction", "label_id", "vib_id", "ref_id", "response", "condition_id"):
            if field not in row:
                problems.append(f"missing {field}")
        if problems:
            violations.append({"id": row.get("id"), "problems": problems})
            continue
        if row["vib_id"] not in by_id.index or row["ref_id"] not in by_id.index:
            problems.append("vib_id/ref_id not in metadata")
        else:
            vib, ref = by_id.loc[row["vib_id"]], by_id.loc[row["ref_id"]]
            if int(vib.label) != row["label_id"]:
                problems.append("vibration label mismatch")
            if int(vib.condition_id) != row["condition_id"]:
                problems.append("vibration condition mismatch")
            if int(ref.label) != 0:
                problems.append("reference not healthy")
            if int(ref.condition_id) != row["condition_id"]:
                problems.append("reference condition mismatch")
        if not isinstance(row["instruction"], str) or not row["instruction"].strip():
            problems.append("empty instruction")
        if not isinstance(row["response"], str) or not row["response"].strip():
            problems.append("empty response")
        if row["task_id"] not in range(4):
            problems.append("unknown task")
        if "#state_place_holder#" not in row["instruction"]:
            problems.append("missing upstream state placeholder")
        if row["task_id"] == 0 and row["response"] != ("no" if row["label_id"] == 0 else "yes"):
            problems.append("binary response inconsistent with label")
        if row["task_id"] == 1 and (
                row["label_id"] not in range(len(CORPUS_LABEL_NAMES))
                or row["response"] != CORPUS_LABEL_NAMES[row["label_id"]]):
            problems.append("diagnosis response inconsistent with label")
        if problems:
            violations.append({"id": row["id"], "problems": problems})
    checks["corpus_records_valid"] = not violations
    checks["corpus_ids_unique"] = len({r["id"] for r in corpus}) == len(corpus)
    checks["corpus_ids_contiguous_from_zero"] = [r["id"] for r in corpus] == list(range(len(corpus)))
    eligible_split = dict(zip(eligible_rows.file_id.tolist(), split_names.tolist()))
    report["corpus"] = {
        "sample_count": len(corpus), "task_counts": counts(r["task_id"] for r in corpus),
        "task_names": {0: "fault_detection", 1: "fault_diagnosis", 2: "maintenance_recommendation", 3: "risk_assessment"},
        "label_counts": counts(r["label_id"] for r in corpus),
        "label_names": dict(enumerate(CORPUS_LABEL_NAMES)),
        "task_label_counts": {str(t): counts(r["label_id"] for r in corpus if r["task_id"] == t) for t in range(4)},
        "source_counts": counts(by_id.loc[r["vib_id"], "dataset"] for r in corpus),
        "vibration_split_membership_under_regenerated_split": counts(eligible_split.get(r["vib_id"], "excluded") for r in corpus),
        "unique_vibration_ids": len({r["vib_id"] for r in corpus}),
        "unique_reference_ids": len({r["ref_id"] for r in corpus}),
        "violations": violations,
    }
    corpus_training_count = report["corpus"]["vibration_split_membership_under_regenerated_split"].get("train", 0)
    if corpus_training_count:
        report["warnings"].append(
            f"{corpus_training_count}/{len(corpus)} published corpus vibration queries belong to the "
            "regenerated upstream training split. Corpus replay is not an independent held-out benchmark.")

    if not args.metadata_only:
        with h5py.File(args.data_dir / "data.hdf5", "r") as handle:
            signals = handle["vibration"]
            checks["hdf5_expected_shape"] = signals.shape == (total, 24000)
            checks["hdf5_float32"] = signals.dtype == np.dtype("float32")
            info = {"keys": list(handle.keys()), "shape": list(signals.shape),
                    "dtype": str(signals.dtype), "chunks": signals.chunks,
                    "compression": signals.compression, "preprocessing": "Already DCN; do not apply DCN again."}
            finite_bad, zero_rows, energy_bad = [], [], []
            energies = np.empty(signals.shape[0], dtype=np.float64)
            minima, maxima, max_dc = [], [], []
            for start in range(0, signals.shape[0], args.batch_size):
                batch = signals[start:start + args.batch_size]
                row_finite = np.isfinite(batch).all(axis=1)
                finite_bad.extend((np.flatnonzero(~row_finite) + start).tolist())
                energy = np.einsum("ij,ij->i", batch, batch, dtype=np.float64)
                energies[start:start + len(batch)] = energy
                zero_rows.extend((np.flatnonzero(energy == 0) + start).tolist())
                energy_bad.extend((np.flatnonzero(~np.isclose(energy, 2.4, rtol=1e-4, atol=1e-6)) + start).tolist())
                minima.append(float(np.nanmin(batch)))
                maxima.append(float(np.nanmax(batch)))
                max_dc.append(float(np.nanmax(np.abs(batch[:, 0]))))
                print(f"Validated signal values: {start + len(batch)}/{signals.shape[0]}", flush=True)
            checks["hdf5_all_values_finite"] = not finite_bad
            checks["hdf5_no_zero_signals"] = not zero_rows
            checks["hdf5_dcn_energy_matches"] = not energy_bad
            info.update({
                "values_checked": int(signals.size), "rows_checked": signals.shape[0],
                "min": min(minima), "max": max(maxima), "max_abs_dc_coefficient": max(max_dc),
                "energy_expected": 2.4, "energy_quantiles": dict(zip(
                    ["min", "p01", "median", "p99", "max"],
                    np.quantile(energies, [0, .01, .5, .99, 1]).tolist())),
                "nonfinite_row_ids": finite_bad, "zero_energy_row_ids": zero_rows,
                "unexpected_energy_row_ids": energy_bad,
            })
            report["hdf5"] = info

    report["passed"] = all(checks.values())
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Audit passed={report['passed']}; report={args.output}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
