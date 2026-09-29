#!/usr/bin/env python3
"""Local, recording-level R2 export. No training, label-based filtering or raw edits."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import platform
from datetime import datetime, timezone

import h5py
import numpy as np
from radar_r2 import parse_cfg, process_radar
from contact_packets import process_contact

ROOT = Path(__file__).resolve().parent
VERSION = "r2-local-20260928-v1"
FIELDS = ["recording_id", "label", "rpm", "distance_cm", "repeat", "radar_path",
          "radar_parts", "contact_path", "contact_id", "pairing_level", "paired_confirmed", "session_id", "bearing_id",
          "environment", "contact_fs_hz", "contact_unit", "cfg_path", "radar_format",
          "iq_order", "aux_paths", "notes"]
LABELS = ["normal", "inBroken", "outBroken", "roll"]


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def csv_write(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def template(args):
    dest = Path(args.output)
    if dest.exists():
        raise ValueError(f"Refusing to overwrite: {dest}")
    rows = []
    for label in LABELS:
        for rpm in [1000, 2000, 3000]:
            for cm in [20, 40, 80]:
                for repeat in [1, 2, 3, 4]:
                    rows.append(dict(recording_id=f"{label}_{rpm}r_{cm}cm_rep{repeat}",
                                     label=label, rpm=rpm, distance_cm=cm, repeat=repeat,
                                     contact_id=f"{label}_{rpm}r_session", pairing_level="continuous_session",
                                     contact_fs_hz=4000, iq_order="iiqq_neg"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    csv_write(dest, rows, FIELDS)
    print(f"Created {len(rows)} planned recordings: {dest}", flush=True)


def inventory(args):
    root = Path(args.data_root).expanduser().resolve()
    dest = Path(args.output).resolve()
    if not root.is_dir() or dest.exists():
        raise ValueError("Input root must exist; inventory output must be new")
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".bin", ".dat", ".cfg", ".json", ".log", ".txt", ".csv"}:
            rows.append(dict(path=p.relative_to(root).as_posix(), size_bytes=p.stat().st_size,
                             extension=p.suffix.lower(), modified_ns=p.stat().st_mtime_ns))
    csv_write(dest, rows, ["path", "size_bytes", "extension", "modified_ns"])
    print(f"Inventoried {len(rows)} files; no pairing was inferred: {dest}")


def resolve_input(root, value):
    p = Path(value).expanduser()
    p = p.resolve() if p.is_absolute() else (root / p).resolve()
    if not p.is_file():
        raise ValueError(f"File not found: {p}")
    return p


def json_list(value, name):
    if not value.strip():
        return []
    x = json.loads(value)
    if not isinstance(x, list) or not all(isinstance(y, str) and y for y in x):
        raise ValueError(f"{name} must be a JSON list of file paths")
    return x


def read_jobs(args):
    root = Path(args.data_root).expanduser().resolve()
    with Path(args.manifest).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS[:8]).issubset(reader.fieldnames or []):
            raise ValueError("Use the supplied manifest_template_144.csv column names")
        rows = [{k: (v or "").strip() for k, v in row.items() if k is not None} for row in reader]
    jobs, unused, seen_ids, source_owner, contacts, contact_names = [], [], set(), {}, {}, {}
    for row in rows:
        rid = row.get("recording_id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", rid) or rid in seen_ids:
            raise ValueError(f"Invalid or duplicate recording_id: {rid!r}")
        seen_ids.add(rid)
        if row["label"] not in LABELS:
            raise ValueError(f"Unknown label for {rid}; expected {LABELS}")
        if int(row["rpm"]) <= 0 or float(row["distance_cm"]) <= 0 or int(row["repeat"]) <= 0:
            raise ValueError(f"Invalid condition: {rid}")
        has_radar = bool(row.get("radar_path") or row.get("radar_parts"))
        has_contact = bool(row.get("contact_path"))
        if not has_radar and not has_contact:
            unused.append(rid)
            continue
        if not has_radar or not has_contact:
            raise ValueError(f"{rid}: both radar and contact paths are required")
        if row.get("radar_path") and row.get("radar_parts"):
            raise ValueError(f"{rid}: specify radar_path OR radar_parts, not both")
        parts = [row["radar_path"]] if row.get("radar_path") else json_list(row["radar_parts"], "radar_parts")
        radars = [resolve_input(root, p) for p in parts]
        contact = resolve_input(root, row["contact_path"])
        for p in radars:
            # Same physical source cannot be repackaged as several independent recordings.
            if str(p) in source_owner:
                raise ValueError(f"Source reused: {p}; {source_owner[str(p)]} and {rid}")
            source_owner[str(p)] = rid
        cfg = resolve_input(root, row["cfg_path"]) if row.get("cfg_path") else Path(args.cfg).expanduser().resolve()
        config = parse_cfg(cfg)
        fs = float(row.get("contact_fs_hz") or 4000)
        if not fs.is_integer() or fs <= 0:
            raise ValueError(f"{rid}: contact_fs_hz must be a positive integer")
        cid = row.get("contact_id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cid):
            raise ValueError(f"{rid}: contact_id must identify the original DAT session")
        contact_spec = (str(contact), fs, row["label"], row["rpm"])
        if cid in contacts and contacts[cid] != contact_spec:
            raise ValueError(f"{rid}: shared contact_id {cid} has inconsistent source/condition/sample rate")
        if str(contact) in contact_names and contact_names[str(contact)] != cid:
            raise ValueError(f"{rid}: one DAT must use one contact_id across all radar recordings")
        contacts[cid] = contact_spec
        contact_names[str(contact)] = cid
        if row.get("pairing_level") not in {"continuous_session", "condition_reference", "recording", "unknown"}:
            raise ValueError(f"{rid}: pairing_level must describe the actual acquisition relationship")
        fmt = row.get("radar_format") or args.radar_format
        if fmt != "headerless_reordered_adc":
            raise ValueError(f"{rid}: explicitly set --radar-format headerless_reordered_adc only after confirming capture output format")
        order = row.get("iq_order") or "iiqq_neg"
        if order not in {"iiqq_neg", "iiqq_pos"}:
            raise ValueError(f"{rid}: unsupported iq_order {order}")
        aux = [resolve_input(root, p) for p in json_list(row.get("aux_paths", ""), "aux_paths")]
        jobs.append(dict(row=row, radars=radars, contact=contact, cfg=cfg, config=config,
                         contact_fs_hz=int(fs), iq_order=order, aux=aux))
    if args.recording:
        requested = set(args.recording)
        unknown = requested - {j["row"]["recording_id"] for j in jobs}
        if unknown:
            raise ValueError(f"Requested IDs have no complete file mapping: {sorted(unknown)}")
        jobs = [j for j in jobs if j["row"]["recording_id"] in requested]
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        raise ValueError("No mapped recordings. Fill radar_path/contact_path in the manifest first.")
    return jobs, unused


def fingerprint(job, args):
    sources = job["radars"] + [job["contact"]] + job["aux"]
    specs = [dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sources]
    code = {name: hash_file(ROOT / name) for name in ["preprocess_local.py", "radar_r2.py", "contact_packets.py"]}
    payload = dict(version=VERSION, metadata=job["row"], sources=specs, cfg_sha256=hash_file(job["cfg"]),
                   fs=job["contact_fs_hz"], iq_order=job["iq_order"], roi_half_width_m=args.roi_half_width_m,
                   compact_bins=args.compact_bins, archive_wide=not args.no_wide_cache,
                   max_frames=args.max_frames, code_sha256=code)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def compress_copy(source, dest):
    digest = hashlib.sha256()
    with source.open("rb") as src, gzip.open(dest, "wb", compresslevel=3) as out:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
            out.write(block)
    return digest.hexdigest()


def export_contact_once(job, out):
    """Export one long DAT once even when twelve radar recordings refer to it."""
    cid = job["row"]["contact_id"]
    dest = out / "upload" / "contacts" / cid
    source = job["contact"]
    spec = dict(source=str(source), size=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns,
                sample_rate_hz=job["contact_fs_hz"], parser_sha256=hash_file(ROOT / "contact_packets.py"))
    sig = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if (dest / "DONE.json").is_file():
        previous = json.loads((dest / "DONE.json").read_text(encoding="utf-8"))
        if previous["fingerprint"] != sig or any(not (dest / n).is_file() for n in previous["files"]):
            raise ValueError(f"Contact session {cid} changed or incomplete; use a new output directory")
        return json.loads((dest / "contact_qc.json").read_text(encoding="utf-8"))
    work = out / "work_contacts_local" / cid
    if dest.exists() or work.exists():
        raise ValueError(f"Partial contact export exists: {cid}; use a new output directory")
    work.mkdir(parents=True)
    qc = process_contact(source, work / "contact.h5", sample_rate_hz=job["contact_fs_hz"])
    copied_sha = compress_copy(source, work / "contact_original.DAT.gz")
    if copied_sha != qc["source_sha256"] or source.stat().st_size != spec["size"] or source.stat().st_mtime_ns != spec["mtime_ns"]:
        raise ValueError(f"Contact source changed between parsing and archival: {source}")
    json_write(work / "contact_qc.json", qc)
    files = {p.name: dict(bytes=p.stat().st_size, sha256=hash_file(p)) for p in work.iterdir() if p.is_file()}
    json_write(work / "DONE.json", dict(fingerprint=sig, source=spec, contact_id=cid, files=files))
    dest.parent.mkdir(parents=True, exist_ok=True)
    work.replace(dest)
    return qc


def process(args):
    jobs, unused = read_jobs(args)
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    for j in jobs:
        checks.append(dict(recording_id=j["row"]["recording_id"], radar_bytes=sum(p.stat().st_size for p in j["radars"]),
                           contact_bytes=j["contact"].stat().st_size, cfg=j["config"],
                           pair_confirmed=j["row"].get("paired_confirmed", "")))
    json_write(out / "preflight.json", dict(version=VERSION, mapped_recordings=len(jobs),
               unmapped_recordings=unused, checks=checks, raw_format="headerless_reordered_adc",
               split="unassigned; repeat labels alone do not establish independent acquisitions"))
    print(f"Preflight OK: {len(jobs)} mapped recordings; {len(unused)} blank template rows. Details: {out / 'preflight.json'}", flush=True)
    if args.dry_run:
        return
    catalog_path = out / "export_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"expected_radar": {}}
    for j in jobs:
        catalog["expected_radar"][j["row"]["recording_id"]] = j["row"]["contact_id"]
    json_write(catalog_path, catalog)
    runtime = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, h5py=h5py.__version__)
    json_write(out / "runtime.json", runtime)
    summaries, failures = [], []
    for index, job in enumerate(jobs, 1):
        rid = job["row"]["recording_id"]
        dest = out / "upload" / "radar" / rid
        work = out / "work_local" / rid
        sig, provenance = fingerprint(job, args)
        if (dest / "DONE.json").is_file():
            old = json.loads((dest / "DONE.json").read_text(encoding="utf-8"))
            if old.get("fingerprint") != sig:
                raise ValueError(f"{rid}: previous export uses different inputs/options/code. Use a NEW output directory.")
            missing = [p for p in old["files"] if not (dest / p).is_file()]
            if missing:
                raise ValueError(f"{rid}: completed output missing files: {missing}; use a new output directory")
            export_contact_once(job, out)
            if not args.no_wide_cache and not (out / "wide_cache_local" / f"{rid}.h5").is_file():
                raise ValueError(f"{rid}: local wide ROI cache is missing; preserve current upload and recover the cache or use a new output directory")
            print(f"[{index}/{len(jobs)}] {rid}: already complete (source size/mtime match; use verify for output hashes)", flush=True)
            summaries.append(json.loads((dest / "recording.json").read_text(encoding="utf-8")))
            continue
        try:
            if dest.exists() or work.exists():
                raise ValueError(f"Partial output exists for {rid}. Preserve it for diagnosis and use a new output directory.")
            work.mkdir(parents=True)
            cq = export_contact_once(job, out)
            print(f"[{index}/{len(jobs)}] {rid}: decoding radar ...", flush=True)
            cache = out / "wide_cache_local" / f"{rid}.h5"
            if not args.no_wide_cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
            rq = process_radar(job["radars"] if len(job["radars"]) > 1 else job["radars"][0], job["cfg"],
                               work / "radar.h5", float(job["row"]["distance_cm"]),
                               iq_order=job["iq_order"], roi_half_width_m=args.roi_half_width_m,
                               compact_bins=args.compact_bins, batch_frames=args.batch_frames,
                               archive_h5=None if args.no_wide_cache else cache, max_frames=args.max_frames)
            shutil.copyfile(job["cfg"], work / "actual_radar.cfg")
            aux_sources = []
            for i, p in enumerate(job["aux"]):
                name = f"aux_{i:02d}_{p.name}"
                shutil.copyfile(p, work / name)
                aux_sources.append(dict(source=str(p), exported=name, sha256=hash_file(p)))
            pair_ok = job["row"].get("paired_confirmed", "").lower() in {"yes", "true", "1", "是"}
            rec = dict(recording_id=rid, metadata=job["row"], schema_version=VERSION,
                       source_provenance=provenance, radar=rq, contact=cq, auxiliary=aux_sources,
                       contact_id=job["row"]["contact_id"], contact_file=f"../../contacts/{job['row']['contact_id']}/contact.h5",
                       pairing_level=job["row"]["pairing_level"],
                       recording_pair_confirmed=pair_ok, precise_time_alignment="not established",
                       association_confirmed=pair_ok, synchronized_window_pair=False,
                       eligible_for_session_reference=(pair_ok and args.max_frames is None
                           and cq["ready_for_contact_model"] and rq["complete_two_second_windows"] > 0),
                       is_pilot=args.max_frames is not None, split="unassigned", runtime=runtime,
                       created_utc=datetime.now(timezone.utc).isoformat())
            json_write(work / "recording.json", rec)
            files = {p.name: dict(bytes=p.stat().st_size, sha256=hash_file(p)) for p in work.iterdir() if p.is_file()}
            json_write(work / "DONE.json", dict(fingerprint=sig, files=files))
            dest.parent.mkdir(parents=True, exist_ok=True)
            work.replace(dest)
            summaries.append(rec)
            print(f"[{index}/{len(jobs)}] {rid}: complete; upload {sum(x['bytes'] for x in files.values()) / 1e6:.1f} MB", flush=True)
        except Exception as e:
            failure = dict(recording_id=rid, error=f"{type(e).__name__}: {e}")
            failures.append(failure)
            print(f"ERROR {rid}: {failure['error']}", file=sys.stderr, flush=True)
        json_write(out / "summary.json", dict(completed=summaries, failed=failures, unmapped=unused))
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "upload" / "radar").glob("*/recording.json"))]
    json_write(out / "summary.json", dict(completed=summaries, failed_in_latest_run=failures, unmapped_in_latest_manifest=unused))
    csv_write(out / "qc_index.csv", [dict(recording_id=r["recording_id"], label=r["metadata"]["label"],
               rpm=r["metadata"]["rpm"], distance_cm=r["metadata"]["distance_cm"], repeat=r["metadata"]["repeat"],
               contact_id=r["contact_id"], frames=r["radar"]["frames_exported"],
               duration_s=r["radar"]["nominal_export_duration_s"], selected_range_m=r["radar"]["selected_center_range_m"],
               zero_chirps=r["radar"]["all_zero_ADC_chirps"], adc_near_limit_fraction=r["radar"]["ADC_near_limit_fraction"],
               tail_bytes=r["radar"]["trailing_incomplete_frame_bytes"],
               contact_complete_packets=r["contact"]["complete_packets"],
               radar_flags=";".join(r["radar"]["quality_flags"]),
               paired_confirmed=r["recording_pair_confirmed"], is_pilot=r["is_pilot"],
               pairing_level=r["pairing_level"]) for r in summaries],
              ["recording_id", "label", "rpm", "distance_cm", "repeat", "contact_id", "frames", "duration_s",
               "selected_range_m", "zero_chirps", "adc_near_limit_fraction", "tail_bytes", "contact_complete_packets",
               "radar_flags", "paired_confirmed", "pairing_level", "is_pilot"])
    print(f"Finished: {len(summaries)} successful, {len(failures)} failed. Upload directory: {out / 'upload'}", flush=True)
    if failures:
        raise SystemExit(2)


def verify(args):
    root = Path(args.output).expanduser().resolve()
    manifests = sorted((root / "upload").rglob("DONE.json"))
    if not manifests:
        raise ValueError("No completed recordings found under output/upload/")
    failures, missing_groups, dangling = [], [], []
    catalog_path = root / "export_catalog.json"
    if not catalog_path.exists():
        raise ValueError("export_catalog.json is required; transfer it alongside upload/ to verify whole-group completeness")
    expected = json.loads(catalog_path.read_text(encoding="utf-8"))["expected_radar"]
    for rid, cid in expected.items():
        for rel in [Path("radar") / rid / "DONE.json", Path("contacts") / cid / "DONE.json"]:
            if not (root / "upload" / rel).is_file():
                missing_groups.append(str(rel))
    checked = 0
    for p in manifests:
        m = json.loads(p.read_text(encoding="utf-8"))
        for name, item in m["files"].items():
            f = p.parent / name
            if not f.is_file() or f.stat().st_size != item["bytes"] or hash_file(f) != item["sha256"]:
                failures.append(str(f))
            checked += 1
    for rec_path in (root / "upload" / "radar").glob("*/recording.json"):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        target = (rec_path.parent / rec["contact_file"]).resolve()
        if not target.is_file() or not (target.parent / "DONE.json").is_file():
            dangling.append(rec["recording_id"])
    result = dict(completed_export_groups=len(manifests), expected_radar_recordings=len(expected),
                  files_checked=checked, failed_files=failures, missing_export_groups=sorted(set(missing_groups)),
                  radar_with_missing_contact=dangling,
                  verifies="export file integrity only; not hardware decoding or scientific validity")
    json_write(root / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures or missing_groups or dangling:
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    t = s.add_parser("template", help="Generate the 144-recording mapping table")
    t.add_argument("--output", default="manifest_template_144.csv")
    t.set_defaults(func=template)
    i = s.add_parser("inventory", help="List source files; never guess synchronized pairs")
    i.add_argument("--data-root", required=True)
    i.add_argument("--output", default="file_inventory.csv")
    i.set_defaults(func=inventory)
    r = s.add_parser("process", help="Validate manifest then export recordings")
    r.add_argument("--data-root", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--cfg", default=str(ROOT / "R2_reference.cfg"))
    r.add_argument("--output", required=True)
    r.add_argument("--radar-format", choices=["headerless_reordered_adc"])
    r.add_argument("--recording", action="append", help="Process only this ID; may be repeated")
    r.add_argument("--limit", type=int)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--roi-half-width-m", type=float, default=.12)
    r.add_argument("--compact-bins", type=int, default=5)
    r.add_argument("--batch-frames", type=int, default=4)
    r.add_argument("--no-wide-cache", action="store_true", help="Not recommended: omit local wide ROI fallback")
    r.add_argument("--max-frames", type=int, help="Pilot only: truncate radar, mark output ineligible for full training")
    r.set_defaults(func=process)
    v = s.add_parser("verify", help="Verify export SHA256 checksums before/after transfer")
    v.add_argument("--output", required=True)
    v.set_defaults(func=verify)
    args = p.parse_args()
    if hasattr(args, "limit") and args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if hasattr(args, "max_frames") and args.max_frames is not None and args.max_frames < 1:
        p.error("--max-frames must be positive")
    try:
        args.func(args)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        p.exit(2, f"ERROR: {e}\n")


if __name__ == "__main__":
    main()
