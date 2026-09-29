"""Read manifests and stat only project-scoped files; never read large raw binaries."""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import csv
import json
from collections import Counter
BASE = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection'))
OUT = Path(_c2r_resolve_path(__file__)).parent
ENV = BASE / 'environment_validation_20260919/results'
V2 = BASE / 'encoder_validation_20260923_v2'
RAW = Path(_c2r_resolve_path('/media/nas_users/huangyating/data'))

def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def save_csv(name, rows):
    keys = list(dict.fromkeys((k for row in rows for k in row)))
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
env = json.loads((ENV / 'input_manifest.json').read_text())
window_counts = Counter((row['file'] for row in read_csv(ENV / 'windows.csv')))
rows = []
for row in env:
    path = BASE / 'data9.18' / row['file']
    rows.append(dict(dataset='environment_20260918', file=row['file'], raw_path=str(path), raw_currently_exists=path.exists(), derived_features_currently_exist=(ENV / 'features.npz').exists(), environment=row['environment'], rpm=row['rpm'], distance_cm=row['distance_cm'], state=row['bearing_state'], condition='off' if row['rpm'] == 0 else 'rotating_normal', windows_2s=window_counts[row['file']], paired_contact='not_available_in_archived_protocol', independent_repeat='unconfirmed', physical_bearing_count='unconfirmed'))
save_csv('environment_recordings.csv', rows)
counts = Counter(((r['environment'], r['rpm'], r['distance_cm'], r['condition']) for r in rows))
save_csv('environment_condition_matrix.csv', [dict(environment=k[0], rpm=k[1], distance_cm=k[2], condition=k[3], recordings=v) for k, v in sorted(counts.items())])
contacts = {r['file']: r for r in read_csv(V2 / 'raw/contact_raw_identity_and_quality.csv')}
radar = read_csv(V2 / 'geometry_corrected/radar/metadata.csv')
radar_counts = Counter((r['bag_id'] for r in radar))
unique = {r['bag_id']: r for r in radar}
pairs = []
for bag_id, r in unique.items():
    contact_name = Path(_c2r_resolve_path(r['contact_candidate'])).with_suffix('.DAT').name
    c = contacts[contact_name]
    radar_path = RAW / (bag_id + '.bin')
    contact_path = RAW / contact_name
    pairs.append(dict(dataset='paired_20260920', radar_file=radar_path.name, radar_path=str(radar_path), radar_currently_exists=radar_path.exists(), contact_file=contact_name, contact_path=str(contact_path), contact_currently_exists=contact_path.exists(), state=r['state'], four_class_label=r['label'], baud=r['baud_candidate'], distance_user_description_cm='40+', environment='not_recorded', rpm_truth='not_recorded', windows_radar_2s=radar_counts[bag_id], windows_contact_1s=c['saved_1s_windows'], simultaneous_recording='user_confirmed', window_synchronization='not_established', physical_bearing_identity='same_per_class_across_baud_groups_for_four_classes'))
save_csv('paired_recordings.csv', pairs)
speed = json.loads((BASE / 'contact_validation_20260919/results/quality.json').read_text())
speed_rows = []
for r in speed:
    path = BASE / '接触式' / r['file']
    speed_rows.append(dict(file=r['file'], raw_path=str(path), raw_currently_exists=path.exists(), rpm=r['rpm'], state='normal_per_previous_report', max_contiguous_samples=r['max_contiguous_samples'], sampling_rate_hz=r['sampling_rate_hz'], max_contiguous_seconds=r['max_contiguous_samples'] / r['sampling_rate_hz'], strict_1s_windows=r['strict_one_second_windows'], paired_radar='not_established', environment='not_recorded', independent_recordings_per_rpm=1))
save_csv('contact_speed_recordings.csv', speed_rows)
summary = dict(audit='2026-09-24 read-only raw stat and existing manifests', environment_expected_raw_directory_exists=(BASE / 'data9.18').exists(), environment_archived_recordings=len(rows), environment_rotating_normal=sum((r['rpm'] > 0 for r in rows)), environment_off=sum((r['rpm'] == 0 for r in rows)), environment_windows=sum(window_counts.values()), environment_derived_available=(ENV / 'features.npz').exists(), paired_recordings=len(pairs), paired_raw_files_present=sum((r['radar_currently_exists'] + r['contact_currently_exists'] for r in pairs)), paired_contact_windows=sum((int(r['windows_contact_1s']) for r in pairs)), paired_radar_windows=sum((int(r['windows_radar_2s']) for r in pairs)), standalone_contact_speed_files=len(speed_rows), standalone_contact_speed_strict_1s_windows=sum((r['strict_1s_windows'] for r in speed_rows)), limitations=['No raw large-bin reads or retraining performed.', 'Missing expected raw directory is not a global absence claim.', 'Archived duplication checks do not establish independent physical repeats.', 'No cross-fault, multi-speed paired matrix is documented in the inspected project files.'])
(OUT / 'data_inventory_verification.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(summary, indent=2, ensure_ascii=False))
