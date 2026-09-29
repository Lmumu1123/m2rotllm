"""Read-only integrity/quality audit of uploaded R2 exports; no model fitting."""
from c2r_paths import resolve_path as _c2r_resolve_path
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(_c2r_resolve_path('/media/nas_users/huangyating/data'))
OUT = Path(_c2r_resolve_path(__file__)).resolve().parent
LABELS = ['normal', 'inBroken', 'outBroken', 'roll']

def sha256(path, compressed=False):
    h = hashlib.sha256()
    op = gzip.open if compressed else open
    with op(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys((k for r in rows for k in r)))
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def verify_done():
    rows = []
    for i, done in enumerate(sorted((ROOT / 'upload').glob('*/*/DONE.json'))):
        j = json.loads(done.read_text())
        for name, expected in j['files'].items():
            path = done.parent / name
            actual = sha256(path) if path.exists() else None
            rows.append(dict(group=done.parent.parent.name, id=done.parent.name, file=name, expected_sha256=expected['sha256'], actual_sha256=actual, expected_bytes=expected['bytes'], actual_bytes=path.stat().st_size if path.exists() else None, ok=actual == expected['sha256'] and path.stat().st_size == expected['bytes']))
        if (i + 1) % 24 == 0:
            print(f'Integrity checked {i + 1}/156 export groups', flush=True)
    save_csv('integrity_checks.csv', rows)
    return rows

def radar_audit():
    rows, inventory, matches, profiles = ([], [], [], [])
    for path in sorted((ROOT / 'upload/radar').glob('*/recording.json')):
        j = json.loads(path.read_text())
        m = j['metadata']
        r = j['radar']
        rid = j['recording_id']
        row = {k: m[k] for k in ['label', 'rpm', 'distance_cm', 'repeat', 'contact_id']}
        row.update(recording_id=rid, nominal_duration_s=r['nominal_export_duration_s'], frames=r['frames_exported'], windows=r['complete_two_second_windows'], source_sha256=r['source_sha256'], source_bytes=r['source_bytes'], radar_source_name=Path(_c2r_resolve_path(m['radar_path'])).name, edge=any(('ROI_edge' in f for f in r['quality_flags'])), mirror=any(('mirror' in f for f in r['quality_flags'])), near_limit_flag=any(('saturation_candidate' in f for f in r['quality_flags'])), adc_near_limit_fraction=r['ADC_near_limit_fraction'], pilot_negative_positive_ratio=r['pilot_negative_to_positive_dynamic_ratio'], selected_range_m=r['selected_center_range_m'], selected_bin=r['selected_center_bin'], source_tail_bytes=r['trailing_incomplete_frame_bytes'], physical_roi_low_m=r['physical_roi_bounds_m'][0], physical_roi_high_m=r['physical_roi_bounds_m'][1], compact_range_low_m=r['compact_range_bounds_m'][0], compact_range_high_m=r['compact_range_bounds_m'][1], valid_chirp_fraction_reported=r['valid_chirp_fraction'], contact_association='continuous_session_only', environment=m.get('environment', ''), bearing_id=m.get('bearing_id', ''))
        row['edge_side'] = ('lower' if abs(row['selected_range_m'] - row['physical_roi_low_m']) < 0.021 else 'upper') if row['edge'] else 'interior'
        with h5py.File(path.parent / 'radar.h5', 'r') as f:
            cfg = json.loads(f.attrs['cfg_json'])
            n = len(f['iq'])
            t = f['frame_start_s'][:]
            co = f['chirp_offset_s'][:]
            row['time_grid_ok'] = bool(np.allclose(t, np.arange(n) * 0.04, atol=1e-12) and np.allclose(co, np.arange(192) * 0.0002, atol=1e-12))
            row['valid_chirp_fraction_recomputed'] = float(np.mean(f['valid_chirp'][:]))
            row['shape_ok'] = list(f['iq'].shape[1:]) == [192, 4, 5] and n == row['frames']
            row['cfg_sha256'] = cfg['cfg_sha256']
            ranges = f['range_profile_m'][:]
            prof = np.mean(f['range_profile_dynamic_power'][:], axis=0, dtype=np.float64)
            nonzero = ranges >= 0.06
            pk = np.flatnonzero(nonzero)[np.argmax(prof[nonzero])]
            row['global_dynamic_peak_m'] = float(ranges[pk])
            row['nominal_center_minus_declared_m'] = float(row['selected_range_m'] - int(m['distance_cm']) / 100)
            profiles.append((row.copy(), ranges, prof))
            wp = ROOT / 'wide_cache_local' / (rid + '.h5')
            frames = sorted(set([0, min(1, n - 1), n // 2, max(0, n - 2), n - 1]))
            with h5py.File(wp, 'r') as wide:
                bins = wide['range_bin_indices'][:]
                ix = [int(np.where(bins == b)[0][0]) for b in f['range_bin_indices'][:]]
                compact = f['iq'][frames]
                wider = wide['iq'][frames]
                sampled_equal = np.array_equal(compact, wider[:, :, :, ix])
                matches.append(dict(recording_id=rid, wide_exists=True, sampled_frames=json.dumps(frames), sampled_complex_elements=compact.size, exact_match=sampled_equal, finite_sample=bool(np.isfinite(compact).all() and np.isfinite(wider).all()), wide_frame_count_ok=wide['iq'].shape[0] == n))
        inv = dict(m)
        inv.update(inventory_source='derived_from_uploaded_recording.json_not_original_manifest', uploaded_radar_h5=str(path.parent / 'radar.h5'), uploaded_contact_h5=str(ROOT / 'upload/contacts' / m['contact_id'] / 'contact.h5'), uploaded_wide_h5=str(wp), source_sha256=r['source_sha256'], split='UNASSIGNED', repeat_independence='not_established_timestamp_rank_only')
        inventory.append(inv)
        rows.append(row)
    save_csv('radar_qc.csv', rows)
    save_csv('derived_inventory.csv', inventory)
    save_csv('compact_wide_sample_checks.csv', matches)
    grouped = []
    for cols in [('distance_cm',), ('label',), ('rpm',), ('label', 'distance_cm'), ('label', 'rpm', 'distance_cm')]:
        groups = defaultdict(list)
        for r in rows:
            groups[tuple((r[c] for c in cols))].append(r)
        for keys, rs in groups.items():
            gr = dict(grouping='+'.join(cols), n=len(rs))
            gr.update(dict(zip(cols, keys)))
            gr.update(edge=sum((r['edge'] for r in rs)), lower_edge=sum((r['edge_side'] == 'lower' for r in rs)), upper_edge=sum((r['edge_side'] == 'upper' for r in rs)), mirror=sum((r['mirror'] for r in rs)), adc_near_limit_flag=sum((r['near_limit_flag'] for r in rs)), median_adc_near_limit_fraction=float(np.median([r['adc_near_limit_fraction'] for r in rs])), median_duration_s=float(np.median([r['nominal_duration_s'] for r in rs])), min_duration_s=min((r['nominal_duration_s'] for r in rs)), max_duration_s=max((r['nominal_duration_s'] for r in rs)))
            grouped.append(gr)
    save_csv('radar_qc_grouped.csv', grouped)
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=True)
    for i, label in enumerate(LABELS):
        for k, distance in enumerate(['20', '40', '80']):
            ax = axes[i, k]
            sel = [p for p in profiles if p[0]['label'] == label and p[0]['distance_cm'] == distance]
            for r, x, p in sel:
                ax.plot(x, 10 * np.log10(np.maximum(p, 1e-12) / max(p)), alpha=0.35, linewidth=0.8)
            ax.axvline(int(distance) / 100, color='black', ls='--', lw=1)
            ax.axvspan(int(distance) / 100 - 0.12, int(distance) / 100 + 0.12, color='gold', alpha=0.15)
            ax.set_title(f'{label}, {distance} cm (12 recordings)')
            ax.set_xlim(0, 1.5)
            ax.set_ylim(-60, 3)
            ax.grid(alpha=0.2)
            if i == 3:
                ax.set_xlabel('Range (m), nominal CFG scale')
            if k == 0:
                ax.set_ylabel('Dynamic power (dB / own maximum)')
    fig.suptitle('Full stored range profiles; dashed = declared distance; shaded = retained wide ROI')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / 'radar_range_profiles.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for li, label in enumerate(LABELS):
        rs = [r for r in rows if r['label'] == label]
        xx = np.array([int(r['distance_cm']) for r in rs]) + (li - 1.5) * 1.2
        axes[0].scatter(xx, [r['selected_range_m'] * 100 for r in rs], s=20, label=label, alpha=0.7)
        axes[1].scatter(xx, [r['adc_near_limit_fraction'] * 100 for r in rs], s=20, alpha=0.7)
        axes[2].scatter(xx, [r['pilot_negative_positive_ratio'] for r in rs], s=20, alpha=0.7)
    axes[0].plot([15, 85], [15, 85], ls='--', color='gray')
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel('Selected bin center (cm)')
    axes[1].set_ylabel('ADC near-limit words (%)')
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Negative/positive dynamic power ratio')
    for ax in axes:
        ax.set_xlabel('Declared distance (cm), horizontal jitter by class')
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'radar_quality_by_distance.png', dpi=170)
    plt.close(fig)
    return (rows, matches, grouped)

def contact_audit():
    sessions, packets, archive = ([], [], [])
    clip_examples = []
    for path in sorted((ROOT / 'upload/contacts').glob('*/contact.h5')):
        cid = path.parent.name
        q = json.loads((path.parent / 'contact_qc.json').read_text())
        with h5py.File(path, 'r') as f:
            x = f['raw_xyz'][:]
            ords = f['packet_ordinal_observed'][:]
            endpoints = np.isclose(np.abs(x), 1.998604, rtol=0, atol=1e-09)
            near = np.abs(x) >= 1.99
            label = cid.split('_')[0]
            rpm = int(cid.split('_')[1][:-1])
            row = dict(contact_id=cid, label=label, rpm=rpm, complete_packets=len(x), incomplete_candidates=q['incomplete_candidates'], packet_starts_observed=q['packet_starts_observed'], complete_fraction_of_candidate_starts=len(x) / q['packet_starts_observed'], observed_complete_sample_seconds=len(x) * 1.024, syntax_rows=q['valid_syntax_rows'], orphan_valid_rows=q['orphan_valid_rows'], source_sha256=q['source_sha256'], finite=bool(np.isfinite(x).all()), endpoint_abs_1p998604_fraction=float(endpoints.mean()), near_abs_1p99_fraction=float(near.mean()), endpoint_z_fraction=float(endpoints[:, :, 2].mean()), near_z_fraction=float(near[:, :, 2].mean()), min_value=float(x.min()), max_value=float(x.max()), units=q['units'], sample_rate_hz=q['sample_rate_hz'], timestamp_mapping='unavailable', issues_truncated=q['issues_truncated'])
            row.update({'reason_' + k: v for k, v in q['issue_reason_counts'].items()})
            sessions.append(row)
            for i in range(len(x)):
                p = dict(contact_id=cid, packet_index=i, packet_ordinal_observed=int(ords[i]), label=label, rpm=rpm)
                for a, axis in enumerate('xyz'):
                    p['endpoint_fraction_' + axis] = float(endpoints[i, :, a].mean())
                    p['near_1p99_fraction_' + axis] = float(near[i, :, a].mean())
                    p['std_' + axis] = float(np.std(x[i, :, a]))
                    p['kurtosis_pearson_' + axis] = float(f['packet_stats/kurtosis_pearson_xyz'][i, a])
                packets.append(p)
            if rpm == 2000:
                clip_examples.append((label, x[0]))
        actual = sha256(path.parent / 'contact_original.DAT.gz', compressed=True)
        archive.append(dict(contact_id=cid, expected_original_DAT_sha256=q['source_sha256'], decompressed_sha256=actual, ok=actual == q['source_sha256']))
    save_csv('contact_sessions_qc.csv', sessions)
    save_csv('contact_packet_qc.csv', packets)
    save_csv('contact_archive_checks.csv', archive)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    xx = np.arange(len(sessions))
    colors = ['#1177BB' if s['label'] == 'normal' else '#CC3311' if s['label'] == 'inBroken' else '#229966' if s['label'] == 'outBroken' else '#AA77AA' for s in sessions]
    ax.bar(xx, [100 * s['endpoint_z_fraction'] for s in sessions], color=colors)
    ax.set_xticks(xx, [s['contact_id'].replace('_session', '') for s in sessions], rotation=45, ha='right')
    ax.set_ylabel('Z samples exactly at +/- 1.998604 (%)')
    ax.set_title('Repeated numeric endpoints: clipping candidate; physical units/range not verified')
    fig.tight_layout()
    fig.savefig(OUT / 'contact_endpoint_fraction.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True)
    for ax, label in zip(axes, LABELS):
        x = next((x for la, x in clip_examples if la == label))
        t = np.arange(800) / 4000
        ax.plot(t, x[:800, 2], lw=0.7, label='z axis')
        ax.axhline(-1.998604, color='red', ls='--', lw=0.6)
        ax.axhline(1.998604, color='red', ls='--', lw=0.6)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel('Within-packet time (s)')
    fig.suptitle('First complete packet, 2000 rpm, first 0.2 s; native numeric units')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / 'contact_waveform_examples.png', dpi=170)
    plt.close(fig)
    return (sessions, packets, archive)

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    integrity = verify_done()
    radar, matches, grouped = radar_audit()
    contact, packets, archive = contact_audit()
    hashes = Counter((r['source_sha256'] for r in radar))
    contact_hashes = Counter((r['source_sha256'] for r in contact))
    config_hashes = sorted(set((r['cfg_sha256'] for r in radar)))
    factor_counts = Counter(((r['label'], r['rpm'], r['distance_cm']) for r in radar))
    shared_counts = Counter((r['contact_id'] for r in radar))
    endpoint_by_session = {r['contact_id']: dict(endpoint_z_fraction=r['endpoint_z_fraction'], near_1p99_all_axes_fraction=r['near_abs_1p99_fraction']) for r in contact}
    summary = dict(created_utc=datetime.utcnow().isoformat() + 'Z', data_root=str(ROOT), mode='read_only_no_training_no_data_exclusion', integrity=dict(groups=len(set(((r['group'], r['id']) for r in integrity))), files=len(integrity), ok=all((r['ok'] for r in integrity)), bytes_hashed=sum((r['actual_bytes'] or 0 for r in integrity)), scope='against_each_uploaded_DONE.json;not_original_raw_BIN_verification'), radar=dict(recordings=len(radar), frames=sum((r['frames'] for r in radar)), two_second_windows=sum((r['windows'] for r in radar)), nominal_seconds=sum((r['nominal_duration_s'] for r in radar)), duration_minmax_s=[min((r['nominal_duration_s'] for r in radar)), max((r['nominal_duration_s'] for r in radar))], edge=sum((r['edge'] for r in radar)), negative_mirror=sum((r['mirror'] for r in radar)), adc_near_limit_candidates=sum((r['near_limit_flag'] for r in radar)), source_hashes_unique=len(hashes), duplicated_source_hashes={k: v for k, v in hashes.items() if v > 1}, factorial_cells=len(factor_counts), all_factorial_cells_have_four_recordings=all((v == 4 for v in factor_counts.values())), radar_recordings_per_contact_session=dict(shared_counts), config_hashes=config_hashes, all_shapes_valid=all((r['shape_ok'] for r in radar)), all_nominal_time_grids_valid=all((r['time_grid_ok'] for r in radar)), valid_chirp_fraction_min=min((r['valid_chirp_fraction_recomputed'] for r in radar)), compact_wide_sample_matches=all((r['exact_match'] for r in matches)), compact_wide_sampled_elements=sum((r['sampled_complex_elements'] for r in matches))), contact=dict(sessions=len(contact), complete_packets=sum((r['complete_packets'] for r in contact)), incomplete_candidates=sum((r['incomplete_candidates'] for r in contact)), complete_fraction=sum((r['complete_packets'] for r in contact)) / sum((r['packet_starts_observed'] for r in contact)), source_hashes_unique=len(contact_hashes), original_archives_verify=all((r['ok'] for r in archive)), endpoint_by_session=endpoint_by_session, all_raw_finite=all((r['finite'] for r in contact))), missing_original_top_level_metadata=['manifest.csv', 'export_catalog.json', 'source_checksums.json', 'wide_cache_checksums.json', 'pilot_decision.json'], subsequent_user_confirmation=dict(contact_range='2 g', source='user confirmation in current task, relayed by root', interpretation='fixed +/-1.998604 plateaus consistent with clipping at +/-2g; over-range amplitudes unrecoverable'), identifiability_limits=['one_contact_session_per_class_rpm', 'same_session_contact_cannot_be_split_as_independent_teachers', 'repeat_rank_is_filename_time_order_not_verified_independent_acquisition', 'bearing_id_environment_and_hardware_frame_timestamps_not_supplied', 'no_raw_BIN_or_capture_logs_uploaded;cannot_verify_decoder_layout_sign_frame_origin_or_packet_loss', 'valid_chirp_only_means_not_all_ADC_words_zero', 'export_metadata_had_unconfirmed_units;user_subsequently_confirmed_2g_range;cannot_recover_clipped_amplitudes'], recommendations=['main_grouped_transfer_protocol_leave_one_rpm_out_holds_out_four_whole_DAT_sessions', 'distance_transfer_protocol_is_within_contact_session_condition_transfer', 'recording_holdout_cannot_prove_independent_contact_generalization', 'retain_all_QC_flags_and_compare_quality_nuisance_baseline', 'report_contact_clipping_sensitivity_without_class_specific_deletion'])
    (OUT / 'audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
