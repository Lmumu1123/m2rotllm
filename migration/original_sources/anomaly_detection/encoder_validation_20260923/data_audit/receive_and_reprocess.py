#!/home/huangyating/miniconda3/envs/m2vllm/bin/python
"""Check uploaded original captures; reprocess only when --run is explicit.

This is a one-shot command, not a watcher. The existing reprocessor validates
full raw SHA256 before exporting. Optional DAT are not needed: verified archived
contact NPZ/JSON are reused unchanged by the existing reprocessor.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / "four_class_preprocessing_20260921/results"
SCRIPT = ROOT / "four_class_chain_20260922/scripts/reprocess_geometry.py"
CONFIG = ROOT / "four_class_preprocessing_20260921/雷达原始配置.cfg"
FRAME_BYTES = 256 * 4 * 192 * 2 * 2  # Verified against preprocess.py: 786432.


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stat(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "raw_four_class")
    parser.add_argument("--output", type=Path, default=HERE.parent / "geometry_corrected_r0_045m")
    parser.add_argument("--report", type=Path, default=HERE / "upload_readiness.json")
    parser.add_argument("--stability-seconds", type=float, default=2.,
                        help="One interval between two stat checks, range 0.25..30 seconds")
    parser.add_argument("--verify-sha256", action="store_true",
                        help="Also hash size-matching inputs during dry-run; --run always gets full SHA verification")
    parser.add_argument("--run", action="store_true", help="Execute fixed 0.45m ROI reprocessing after readiness passes")
    args = parser.parse_args()
    if not .25 <= args.stability_seconds <= 30:
        parser.error("--stability-seconds must be between 0.25 and 30")
    for name in ("raw_dir", "output", "report"):
        setattr(args, name, getattr(args, name).resolve())
    # Keep the readiness report outside captures and archived source data.
    if args.report == args.output or args.output in args.report.parents:
        parser.error("--report must be outside --output (reprocessor requires new/empty output)")
    if args.report == args.raw_dir or args.raw_dir in args.report.parents or SOURCE in args.report.parents:
        parser.error("--report must be outside capture directory and archived results")
    if args.report.exists() and args.report.suffix.lower() != ".json":
        parser.error("Existing report may only be a JSON report")

    manifest_path = SOURCE / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {row["file"]: row for row in manifest}
    required = [name for name in expected if name.endswith(".bin")]
    optional = [name for name in expected if name.endswith(".DAT")]
    if len(required) != 12 or len(optional) != 12:
        raise RuntimeError("Archived manifest no longer has the expected 12 radar and 12 contact identities")
    found = {name: [] for name in expected}
    unknown = []
    if args.raw_dir.is_dir():
        for p in args.raw_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".bin", ".dat"):
                if p.name in found:
                    found[p.name].append(p.resolve())
                else:
                    unknown.append(str(p))
    first = {}
    for name, paths in found.items():
        if len(paths) == 1:
            first[name] = stat(paths[0])
    if first:
        print(f"Checking file stability once over {args.stability_seconds:g} seconds...", flush=True)
        time.sleep(args.stability_seconds)
    rows = []
    for name, row in expected.items():
        radar = name in required
        paths = found[name]
        item = dict(file=name, required=radar, expected_bytes=row["bytes"],
                    expected_sha256=row["sha256"], paths=list(map(str, paths)))
        if not paths:
            item["status"] = "missing" if radar else "optional_absent"
        elif len(paths) > 1:
            item["status"] = "duplicate_filename"
        else:
            p = paths[0]
            try:
                second = stat(p)
                item.update(second)
                item["stable_between_checks"] = first[name] == second
                item["size_matches_archive"] = second["bytes"] == row["bytes"]
                if radar:
                    item["complete_frames"] = second["bytes"] // FRAME_BYTES
                    item["trailing_bytes"] = second["bytes"] % FRAME_BYTES
                    item["expected_trailing_bytes"] = row["bytes"] % FRAME_BYTES
                    item["minimum_20_complete_frames"] = item["complete_frames"] >= 20
                    item["tail_policy"] = "Archived captures include partial-frame tails; matching original length and SHA is required, not zero remainder"
                item["status"] = "unstable" if not item["stable_between_checks"] else "size_mismatch" if not item["size_matches_archive"] else "ready_for_full_hash_verification"
                if radar and item["complete_frames"] < 20:
                    item["status"] = "fewer_than_20_complete_frames"
                if args.verify_sha256 and item["status"] == "ready_for_full_hash_verification":
                    print(f"Verifying SHA256: {name}", flush=True)
                    digest = sha(p)
                    item["sha256"] = digest
                    item["stable_during_hash"] = second == stat(p)
                    item["status"] = "unstable" if not item["stable_during_hash"] else "sha256_verified" if digest == row["sha256"] else "sha256_mismatch"
            except OSError as exc:
                item["status"] = "read_error"
                item["error"] = str(exc)
        rows.append(item)
    accepted = {"ready_for_full_hash_verification", "sha256_verified"}
    blockers = [f"{r['file']}: {r['status']}" for r in rows if r["required"] and r["status"] not in accepted]
    output_nonempty = args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir()))
    if output_nonempty:
        blockers.append(f"Output already exists and is not an empty directory: {args.output}")
    command = [sys.executable, str(SCRIPT), "--raw-dir", str(args.raw_dir), "--output", str(args.output),
               "--distance-m", ".45", "--config", str(CONFIG), "--existing-results", str(SOURCE)]
    # Existing inventory checks configuration and archived contact export presence.
    existing_check = subprocess.run(command + ["--check-only"], text=True, capture_output=True)
    try:
        existing_result = json.loads(existing_check.stdout)
    except json.JSONDecodeError:
        existing_result = dict(stdout=existing_check.stdout, stderr=existing_check.stderr)
    if existing_check.returncode and not blockers:
        blockers.append("Existing reprocessor input/configuration check failed; see existing_check")
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), mode="run" if args.run else "dry_run",
        raw_dir=str(args.raw_dir), raw_dir_exists=args.raw_dir.is_dir(),
        output=str(args.output), report=str(args.report), fixed_distance_m=.45, fixed_candidate_roi_m=[.29, .61],
        required_bin_count=12, required_bin_accepted=sum(r["required"] and r["status"] in accepted for r in rows),
        optional_DAT_required_for_ROI_reprocessing=False,
        optional_DAT_warning_count=sum(not r["required"] and r["status"] not in accepted | {"optional_absent"} for r in rows),
        stability_seconds=args.stability_seconds, dry_run_full_hash_requested=args.verify_sha256,
        existing_script_always_checks_all_raw_SHA256_before_export=True,
        archive_manifest_sha256=sha(manifest_path), existing_script_sha256=sha(SCRIPT), entry_script_sha256=sha(Path(__file__)),
        inputs=rows, unknown_capture_names=unknown, blockers=blockers,
        existing_check=dict(returncode=existing_check.returncode, result=existing_result),
        reprocessing_command=command,
        status="blocked" if blockers else "ready_for_verified_reprocessing", reprocessing_executed=False)

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    save()
    print(f"Required radar files ready: {report['required_bin_accepted']}/12")
    if blockers:
        print("Not ready. No reprocessing performed.")
        for reason in blockers:
            print(f"  - {reason}")
        print(f"Report: {args.report}")
        return 2
    if not args.run:
        print("Readiness checks passed. This was a dry-run; no signal export was performed.")
        print("Use --run for reprocessing; all original SHA256 hashes are verified before output creation.")
        print(f"Report: {args.report}")
        return 0
    print("Running verified fixed-geometry export; existing data/model files are not overwritten.", flush=True)
    result = subprocess.run(command)
    report.update(reprocessing_executed=True, reprocessing_returncode=result.returncode,
                  status="complete" if result.returncode == 0 else "reprocessing_failed")
    save()
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
