"""Read-only capture inventory; write evidence only beside this script.

No raw capture is substituted, downloaded, relabelled, or overwritten. Session
confirmations are an explicit overlay, not edits of archived metadata.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
OUT = Path(_c2r_resolve_path(__file__)).resolve().parent
ROOT = OUT.parent.parent
SOURCE = ROOT / 'four_class_preprocessing_20260921/results'
CHAIN = ROOT / 'four_class_chain_20260922'
SEARCH_ROOTS = [ROOT, Path(_c2r_resolve_path('/media/nas_users/huangyating'))]

def read(path):
    return json.loads(path.read_text())

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    manifest = read(SOURCE / 'input_manifest.json')
    expected = {row['file']: row for row in manifest}
    command = ['rg', '--files', '--hidden', '--no-ignore', '-g', '*.bin', '-g', '*.DAT', '-g', '*.dat', '-g', '*.npz', '-g', '*.zip', *map(str, SEARCH_ROOTS)]
    p = subprocess.run(command, capture_output=True, text=True)
    if p.returncode not in (0, 1):
        raise RuntimeError(p.stderr)
    paths = [Path(_c2r_resolve_path(line)) for line in p.stdout.splitlines() if line]
    located = {name: [str(path) for path in paths if path.name == name] for name in expected}
    missing = [dict(file=name, expected_bytes=row['bytes'], sha256=row['sha256'], modality='radar' if name.endswith('.bin') else 'contact', found_paths=located[name]) for name, row in expected.items()]
    write('original_capture_availability.json', missing)
    other_dat = [str(path) for path in paths if path.suffix.lower() == '.dat' and path.name not in expected]
    possible_other_capture = [str(path) for path in paths if path.suffix == '.bin' and path.name not in expected and any((s in path.name.lower() for s in ('normal', 'broken', 'roll', 'keep', 'room', 'narrow', 'adc')))]
    archive_evidence = []
    for path in paths:
        if path.suffix != '.zip' or ROOT not in path.parents:
            continue
        with zipfile.ZipFile(path) as z:
            matches = [n for n in z.namelist() if Path(_c2r_resolve_path(n)).name in expected]
        archive_evidence.append(dict(archive=str(path), original_capture_members=matches))
    bags = read(SOURCE / 'recording_bags.json')
    range_step = 3430000 / 512 * 299792458 / (2 * 49990000000000.0)
    rows = []
    for bag in bags:
        r = read(SOURCE / Path(_c2r_resolve_path(bag['radar'])).with_suffix('.json'))
        c = read(SOURCE / Path(_c2r_resolve_path(bag['contact'])).with_suffix('.json'))
        with np.load(SOURCE / bag['radar'], allow_pickle=False) as z:
            bins = z['range_bins'].astype(int)
        state = r['metadata']['state']
        main = int(bag['label']) >= 0
        rows.append(dict(bag_id=bag['candidate_bag_id'], state=state, label4_archived=bag['label'], label4_confirmed=0 if state == 'bigNormal' else bag['label'], group='main_four_class' if main else 'external_normal' if state == 'bigNormal' else 'unknown_fault_keep', contact_windows=len(c['windows']), radar_windows=len(r['windows']), contact_sampling_hz_user_confirmed=4000, baud_rate_user_confirmed=r['metadata']['baud_candidate'], simultaneous_recording_user_confirmed=True, sample_level_synchronization_available=False, distance_user_confirmed_m='0.4x (approximately forty-something cm)', same_cfg_user_confirmed=True, actual_rpm_ground_truth='unknown_not_inferred_from_33Hz_peak', bearing_entity_user_confirmed='same physical bearing across two baud captures within each main class' if main else 'not individually identified', independent_restarts_or_remounts_confirmed=False, severity_label_available=False, saved_range_bins=','.join(map(str, bins)), saved_range_min_m=float(bins.min() * range_step), saved_range_max_m=float(bins.max() * range_step), range_center_m=r['quality']['range_center_m'], fixed_geometry_roi_029_061m_overlap=bool(np.any((bins * range_step >= 0.29) & (bins * range_step <= 0.61))), archived_condition_confirmed=bag['condition_confirmed'], confirmation_overlay_source='user replies in current task history; archived files unchanged'))
    write_csv('recording_inventory.csv', rows)
    write('recording_inventory.json', rows)
    proposed = OUT / 'corrected_geometry_if_raw_available'
    command = [sys.executable, str(CHAIN / 'scripts/reprocess_geometry.py'), '--raw-dir', str(ROOT), '--output', str(proposed), '--distance-m', '.45', '--check-only']
    check = subprocess.run(command, text=True, capture_output=True)
    (OUT / 'geometry_check.stdout.json').write_text(check.stdout)
    (OUT / 'geometry_check.stderr.txt').write_text(check.stderr)
    guard = json.loads(check.stdout)
    if not any((located[name] for name in expected if name.endswith('.bin'))):
        assert check.returncode == 2 and (not proposed.exists())
        assert len(guard['missing_raw_files']) == 12
    folds = read(ROOT / 'retention_alignment_20260922/radar/stage_b_v0/splits.json')
    by_id = {row['bag_id']: row for row in rows}
    fold_rows = []
    for fold in folds:
        counts = Counter((by_id[k]['label4_archived'] for k in fold['train_bags']))
        fold_rows.append(dict(fold=fold['fold'], train_count_by_class=dict(counts), has_heldout_bags=bool(fold['test_bags']), distinct_training_same_class_donor_available=all((v >= 2 for v in counts.values())), class_prototype_equals_recording_mean=all((v == 1 for v in counts.values())), train_test_overlap=sorted(set(fold['train_bags']) & set(fold['test_bags']))))
    write('same_class_shuffle_feasibility.json', fold_rows)
    inputs = [SOURCE / 'input_manifest.json', SOURCE / 'recording_bags.json', SOURCE / 'summary.json', CHAIN / 'scripts/reprocess_geometry.py', CHAIN / 'radar/check_geometry_reprocessing.py', ROOT / 'retention_alignment_20260922/radar/stage_b_v0/splits.json']
    summary = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable, cpu_threads=2, search_roots=list(map(str, SEARCH_ROOTS)), search_includes_hidden_and_ignored=True, search_follows_directory_symlinks=False, search_file_extensions=['.bin', '.DAT', '.dat', '.npz', '.zip'], matched_files_count=len(paths), original_radar_found=sum((bool(located[n]) for n in expected if n.endswith('.bin'))), original_contact_found=sum((bool(located[n]) for n in expected if n.endswith('.DAT'))), other_contact_raw_files=other_dat, other_possible_capture_files=possible_other_capture, original_captures_in_local_archives=archive_evidence, main_paired_recordings=8, main_contact_windows=sum((r['contact_windows'] for r in rows if r['group'] == 'main_four_class')), main_radar_windows=sum((r['radar_windows'] for r in rows if r['group'] == 'main_four_class')), new_four_class_recordings_identified=0, roi_correction_status='not_executed_missing_original_ADC_bins', cannot_recover_target_range_from_saved_outer_ring_IQ=True, geometry_guard_exit_code=check.returncode, geometry_guard_output_created=proposed.exists(), meaningful_training_only_same_class_shuffle_on_existing_heldout_folds=False, novel_diagnostic_task_with_sufficient_train_test_recordings_available=False, known_unseen_fault_type_keep_has_two_recordings=True, severity_labels_available=False, timing_offset_drift_ground_truth_available=False, static_earlier_speed_contact_records_are_not_paired_new_four_class_tests=True, source_files_unchanged=True, provenance=[dict(path=str(p), sha256=sha(p)) for p in inputs], script_sha256=sha(Path(_c2r_resolve_path(__file__))), audit_limits='Capture search only within stated roots, filename/path identities. No claim that missing files do not exist elsewhere. Existing archived NPZ are derivatives, not new recordings.')
    write('audit_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('original_radar_found', 'original_contact_found', 'new_four_class_recordings_identified', 'main_paired_recordings', 'main_contact_windows', 'main_radar_windows', 'roi_correction_status', 'geometry_guard_exit_code')}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
