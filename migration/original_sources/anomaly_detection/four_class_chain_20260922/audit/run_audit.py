"""Read-only audit of uploaded four-class exports; writes only beside this script."""
from pathlib import Path
import sys, json, hashlib, importlib.util, io, unittest, tempfile
from collections import Counter
from itertools import product
import numpy as np
import scipy

OUT = Path(__file__).resolve().parent
SRC = Path('/home/huangyating/anomaly_detection/four_class_preprocessing_20260921')
DATA = SRC/'results'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC))
tempfile.tempdir = str(OUT)
import preprocess

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(name): return json.loads((DATA/name).read_text())
def save(name, data): (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def main():
    summary=read('summary.json'); verified={r['file']:r for r in read('verification.json')['files']}
    spec=importlib.util.spec_from_file_location('mbhm_dcn', '/media/nas_users/huangyating/bearllm-assets/mbhm_dataset/src/dcn.py')
    official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    results=[]; sample_hashes={}; max_dcn_delta=0.; max_official_delta=0.
    for row in summary:
        p=DATA/row['output']; side=json.loads(p.with_suffix('.json').read_text()); n=row['windows']
        sha=digest(p)
        rec=dict(file=p.name, sha256=sha, hash_matches_uploaded_manifest=sha==verified[p.name]['sha256'],
                 modality=row['modality'], label=row['label'], windows=n, baud_candidate=row['baud_candidate'])
        assert rec['hash_matches_uploaded_manifest']
        assert len(side['windows'])==n
        with np.load(p, allow_pickle=False) as z:
            for key in z.files:
                v=z[key]
                if np.issubdtype(v.dtype, np.number): assert np.isfinite(v).all(), (p.name,key)
            assert (z['labels']==row['label']).all()
            if row['modality']=='contact':
                raw=z['raw_xyz']; d=z['dcn']
                assert raw.shape==(n,4000,3) and d.shape==(n,3,24000)
                assert (d[:,:,4000:]==0).all()
                assert np.array_equal(z['available_dcn_mask'], np.arange(24000)<4000)
                np.testing.assert_allclose(np.mean(d*d,axis=-1),.0001,rtol=1e-5)
                actual=np.stack([preprocess.dcn(x) for x in raw])
                delta=float(np.max(np.abs(actual-d)))
                max_dcn_delta=max(max_dcn_delta,delta)
                np.testing.assert_allclose(actual,d,atol=2e-7,rtol=1e-4)
                native=np.stack([np.stack([official.dcn(x[:,a].astype(np.float64)) for a in range(3)]) for x in raw])
                official_delta=float(np.max(np.abs(native-actual))); max_official_delta=max(max_official_delta,official_delta)
                np.testing.assert_allclose(native,actual,atol=1e-7)
                hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in raw]
                for i,h in enumerate(hashes): sample_hashes.setdefault(h,[]).append(dict(file=p.name,row=i))
                q=side['quality']
                rec.update(complete_packets=q['complete_packets'], packet_starts_observed=q['packet_starts_observed'],
                    issue_count=q['issue_count'], issue_reasons=dict(Counter(x['reason'] for x in q['issues'])),
                    dcn_from_float32_raw_max_abs_error=delta, dcn_matches_mbmh_dataset_max_abs_error=official_delta,
                    raw_min_xyz=np.min(raw,axis=(0,1)).tolist(),raw_max_xyz=np.max(raw,axis=(0,1)).tolist(),
                    values_near_1_998604_fraction=float(np.mean(np.isclose(abs(raw),1.998604,atol=5e-7,rtol=0))),
                    absolute_packet_timestamps_available=any(w['absolute_window_start'] is not None for w in side['windows']))
            else:
                iq=z['iq_frames']; m=z['valid_chirp_mask']; phase=z['valid_phase_mask']; t=z['observed_time_s']
                assert iq.shape==(n,20,192,12,2)
                np.testing.assert_array_equal(phase,m[:,:,1:] & m[:,:,:-1])
                np.testing.assert_allclose(t[1:,0]-t[:-1,-1],.0045)
                np.testing.assert_allclose(np.diff(t,axis=1),.0005)
                np.testing.assert_allclose(z['frame_log_shape'],z['frame_log_power']-z['frame_log_power'].mean(-1,keepdims=True),atol=1e-6)
                # Independently recompute a representative frame spectrum from exported IQ.
                x=iq[0,...,0]+1j*iq[0,...,1]
                fft=np.fft.fft(x[m[0].all(axis=1)]*np.hanning(192)[None,:,None],n=4000,axis=1)
                k=np.rint(z['frequency_hz']*2).astype(int)
                pwr=np.median(np.mean(abs(fft[:,k])**2+abs(fft[:,-k])**2,axis=0),axis=1)/(2000*np.sum(np.hanning(192)**2))
                log=np.log10(pwr+1e-14)
                np.testing.assert_allclose(log,z['frame_log_power'][0],atol=2e-6)
                rec.update(quality=side['quality'], frame_log_power_recompute_max_abs_error=float(np.max(abs(log-z['frame_log_power'][0]))),
                           missing_observed_chirps=int(np.count_nonzero(~m)), frame_gaps_s=.0045,
                           frame_spectrum_resolution_hz=2000/192)
        results.append(rec)
    bags=read('recording_bags.json'); byfile={r['file']:r for r in results}
    assert len({b['contact'] for b in bags})==len(bags)==12
    assert len({b['radar'] for b in bags})==12
    assert all(byfile[b['contact']]['label']==byfile[b['radar']]['label']==b['label'] for b in bags)
    assert all(not b['synchronized_window_pairs'] for b in bags)
    four=[b for b in bags if b['label']>=0]
    protocols=[]
    for trainbaud,testbaud in [(115200,460800),(460800,115200)]:
        train=[b for b in four if byfile[b['contact']]['baud_candidate']==trainbaud]
        test=[b for b in four if byfile[b['contact']]['baud_candidate']==testbaud]
        assert len(train)==len(test)==4 and set(b['label'] for b in train)==set(range(4))
        protocols.append(dict(name=f'baud_{trainbaud}_to_{testbaud}', status='exploratory_file_groups_not_verified_independent_runs',
            train_bags=[b['candidate_bag_id'] for b in train],test_bags=[b['candidate_bag_id'] for b in test],
            train_contact_files=[b['contact'] for b in train],test_contact_files=[b['contact'] for b in test],
            train_radar_files=[b['radar'] for b in train],test_radar_files=[b['radar'] for b in test],
            normal_training_query_with_no_independent_in_training_normal_reference=[b['contact'] for b in train if b['label']==0]))
    exhaustive=[]
    grouped={label:sorted([b for b in four if b['label']==label],key=lambda b:byfile[b['contact']]['baud_candidate']) for label in range(4)}
    for choice in product(range(2),repeat=4):
        train=[grouped[label][choice[label]] for label in range(4)]
        test=[grouped[label][1-choice[label]] for label in range(4)]
        exhaustive.append(dict(split_id=''.join(map(str,choice)),class_order=['normal','inBroken','outBroken','roll'],
            train_bags=[b['candidate_bag_id'] for b in train],test_bags=[b['candidate_bag_id'] for b in test],
            complement_split_id=''.join(str(1-v) for v in choice),
            interpretation='overlapping exploratory partition; not an independent experiment'))
    stream=io.StringIO(); suite=unittest.defaultTestLoader.loadTestsFromName('test_preprocess')
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'unit_tests.log').write_text(stream.getvalue()); assert tests.wasSuccessful()
    runtime=read('runtime.json')
    source_checks=dict(preprocess_sha256=digest(SRC/'preprocess.py'), preprocessing_hash_matches_export=digest(SRC/'preprocess.py')==runtime['preprocessing_sha256'],
        config_sha256=digest(SRC/'雷达原始配置.cfg'), config_hash_matches_export=digest(SRC/'雷达原始配置.cfg')==runtime['config_sha256'])
    assert source_checks['preprocessing_hash_matches_export'] and source_checks['config_hash_matches_export']
    result=dict(status='numeric_export_audit_passed_metadata_not_verified', python=sys.executable, python_version=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,source_checks=source_checks, exported_npz_count=len(results), files=results,
        all_contact_windows=sum(r['windows'] for r in results if r['modality']=='contact'),
        all_radar_windows=sum(r['windows'] for r in results if r['modality']=='radar'),
        four_class_contact_windows=sum(r['windows'] for r in results if r['modality']=='contact' and r['label']>=0),
        four_class_radar_windows=sum(r['windows'] for r in results if r['modality']=='radar' and r['label']>=0),
        duplicate_contact_raw_windows=[v for v in sample_hashes.values() if len(v)>1],
        dcn_max_abs_error_from_exported_float32_raw=max_dcn_delta, dcn_max_abs_error_vs_dataset_definition=max_official_delta,
        unit_tests_run=tests.testsRun, unit_tests_passed=True, raw_inputs_rehashed=False,
        raw_input_search=dict(roots=['/home/huangyating (rg default traversal)','/media/nas_users/huangyating/bearllm-assets'],
                             matching_20260920_DAT_or_bin_found=False, original_export_host='macOS /Users/anthea/Desktop/C2RLLM'),
        assumptions=dict(contact_fs_hz=4000, contact_fs_evidence='user explicitly confirmed 4000 Hz on 2026-09-22; no independent hardware clock validation',
            radar_configuration='archived R0 hash verified; user confirmed same cfg for this batch',
            recording_bags='user confirmed corresponding DAT/bin simultaneously recorded; no per-window synchronized timing proof',
            independent_runs_confirmed=False, geometry_confirmed=True, four_class_same_operating_condition_confirmed=False),
        user_confirmation_20260922=dict(contact_sampling_rate_hz=4000, radar_distance='all four classes forty-something cm',
            radar_configuration_same=True, filename_115200_and_460800='serial baud rate only; two recordings per class',
            dat_bin_same_time=True, keep='bearing housing/base fault, outside current four classes',
            bigNormal='different motor normal, same radar configuration; external normal-only check',
            still_unconfirmed=['precise rpm and load','independent stop-start runs','bearing entity identities'],
            impact='outBroken retained 0.884 m ROI conflicts with reported target distance; raw re-extraction required'),
        exploratory_protocols=protocols,
        exhaustive_balanced_splits=exhaustive,
        exhaustive_split_interpretation=dict(partitions=16,complementary_partition_pairs=8,unique_recording_bags=8,
            held_out_occurrences_per_bag=8, independent_experiments=0,
            independent_experiments_note='not established; 0 verified independent experiments does not assert recordings are identical',
            aggregation='Report paired split-wise differences descriptively; 64 held-out appearances are only 8 unique bags. No window or split-independent confidence intervals.'),
        protocol_rules=['Keep all windows, axes and cross-modal members of each bag in one fold.',
          'Fit scalers, PCA, normalization learned statistics and teachers only on train bags.',
          'Predeclare fixed hyperparameters and epochs: four training bags leave no class-complete independent validation set.',
          'Student inference on held-out bags must not use held-out contact features or labels.',
          'Training-normal self reference or held-out-normal reference invalidates faithful independent-reference BearLLM comparison.',
          'Zero reference/query-only variants may be explicitly labeled adaptations, never original BearLLM reproduction.',
          'Report bag-level macro-F1, accuracy and all eight bag predictions. Window metrics are auxiliary.',
          'Baud is transport metadata, not rpm. No cross-environment, cross-structure or entity-generalization claim.',
          'Repeated seeds quantify optimizer variation, not independent experimental sample size.',
          'keep is an out-of-four-class base fault; bigNormal is normal on another motor. Neither enters main four-class training or its matched normal reference pool.'])
    save('audit.json',result); save('exploratory_group_splits.json',protocols); save('exhaustive_16_split_protocol.json',exhaustive)
    print(json.dumps({k:result[k] for k in ['status','python','exported_npz_count','four_class_contact_windows','four_class_radar_windows','dcn_max_abs_error_from_exported_float32_raw','dcn_max_abs_error_vs_dataset_definition','unit_tests_run','source_checks']},indent=2))

if __name__=='__main__': main()
