"""Read-only audit of split exposure, references, exact duplicates and scalers.

No training and no old result mutation. Exact-byte duplication checks do not
establish original-recording or physical-entity independence.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, sqlite3, time
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
ROOT = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/retention_alignment_20260922'))
OLD = ROOT.parent / 'four_class_chain_20260922'
RAW = ROOT.parent / 'four_class_preprocessing_20260921/results'
ASSET = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-assets/mbhm_dataset'))
PRE = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain'))
OUT = Path(_c2r_resolve_path(__file__)).resolve().parent

def writej(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def filehash(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def equal_file_weights(meta):
    sizes = meta.groupby('bag_id').size()
    return np.array([1 / sizes[b] for b in meta.bag_id]) * len(meta) / len(sizes)

def main():
    assert 'envs/m2vllm' in __import__('sys').executable
    torch.set_num_threads(2)
    started = time.monotonic()
    with sqlite3.connect(f'file:{ASSET}/metadata.sqlite?mode=ro', uri=True) as db:
        schema = [r[0] for r in db.execute("SELECT sql FROM sqlite_master WHERE type='table'")]
        source = pd.read_sql_query('SELECT f.*,c.dataset AS source,c.code,c.channel,c.rpm,c.load FROM file_info f JOIN "condition" c USING(condition_id) ORDER BY file_id', db)
    old = json.loads((PRE / 'dataset.json').read_text())
    pairs = []
    qsplit = {}
    for split, values in old.items():
        for q, r, y in values:
            if q in qsplit:
                assert qsplit[q] == split
            qsplit[q] = split
            pairs.append(dict(query_id=q, reference_id=r, label=y, query_split=split))
    source = source.set_index('file_id', drop=False)
    p = pd.DataFrame(pairs)
    p['reference_query_split'] = p.reference_id.map(qsplit)
    p['self_reference'] = p.query_id == p.reference_id
    p['same_condition'] = source.loc[p.query_id, 'condition_id'].to_numpy() == source.loc[p.reference_id, 'condition_id'].to_numpy()
    p['reference_healthy'] = source.loc[p.reference_id, 'label'].to_numpy() == 0
    assert p.same_condition.all() and p.reference_healthy.all()
    src_ref = p.groupby(['query_split', 'reference_query_split']).agg(pair_count=('query_id', 'size'), unique_reference_count=('reference_id', 'nunique')).reset_index()
    src_ref.to_csv(OUT / 'source_reference_cross_split.csv', index=False)
    self_ref = p[p.self_reference].groupby('query_split').size().to_dict()
    splitstats = []
    for split in ['train', 'val', 'test']:
        ids = {q for q, s in qsplit.items() if s == split}
        part = p[p.query_split == split]
        trainids = {q for q, s in qsplit.items() if s == 'train'}
        conditions = set(source.loc[list(ids), 'condition_id'])
        traincond = set(source.loc[list(trainids), 'condition_id'])
        splitstats.append(dict(split=split, n_queries=len(ids), n_pairs=len(part), self_reference_pairs=int(part.self_reference.sum()), n_conditions=len(conditions), conditions_shared_with_train=len(conditions & traincond), query_id_overlap_with_train=len(ids & trainids) if split != 'train' else None))
    pd.DataFrame(splitstats).to_csv(OUT / 'source_split_statistics.csv', index=False)
    cache = np.load(ROOT / 'source/cache.npz')
    md = pd.read_csv(ROOT / 'source/metadata.csv')
    assert np.array_equal(cache['query_id'], md.file_id) and np.array_equal(cache['reference_ids'], md[['ref_id_0', 'ref_id_1', 'ref_id_2']])
    replay = md[md.split == 'train']
    rr = []
    for row in replay.itertuples():
        assert qsplit[row.file_id] == 'train'
        for j in range(3):
            rid = getattr(row, f'ref_id_{j}')
            rr.append(dict(query_id=row.file_id, reference_id=rid, reference_query_split=qsplit[rid], self_reference=row.file_id == rid))
    replayrefs = pd.DataFrame(rr)
    replayrefs.to_csv(OUT / 'source_replay_references.csv', index=False)
    replayrefstats = replayrefs.groupby('reference_query_split').agg(pairs=('query_id', 'size'), unique_references=('reference_id', 'nunique'), unique_replay_queries=('query_id', 'nunique')).reset_index()
    replayrefstats.to_csv(OUT / 'source_replay_reference_summary.csv', index=False)
    localrefs = json.loads((ROOT / 'preprocessing/reference_metadata.json').read_text())
    for ref in localrefs:
        ref['independently_verified_query_split'] = qsplit[ref['file_id']]
        ref['independently_verified_healthy'] = int(source.loc[ref['file_id'], 'label']) == 0
    pd.DataFrame(localrefs).to_csv(OUT / 'main_local_external_references.csv', index=False)
    baseline = json.loads((ROOT / 'results/retained_head_v1/source_baseline.json').read_text())['test']
    ret = pd.read_csv(ROOT / 'results/retained_head_v1/source_retention.csv')
    ret = ret[(ret.method == 'retained_head') & ret.variant.isin(['v0', 'query_no_demean'])].copy()
    ret['version'] = 'v1'
    ret = ret.rename(columns={'test_acc10': 'acc10', 'test_acc4': 'acc4', 'test_n': 'n'})
    cons = pd.read_csv(ROOT / 'results/retained_head_conservative/source_group_metrics.csv')
    cons = cons[(cons.split == 'test') & (cons.source == 'ALL')].copy()
    cons['version'] = 'conservative'
    cons['fold'] = 'all_known'
    result = pd.concat([ret, cons], ignore_index=True)[['version', 'variant', 'fold', 'n', 'acc10', 'acc4', 'delta_acc10_pp', 'delta_acc4_pp']]
    for k in ['10', '4']:
        before = baseline['ten' if k == '10' else 'four']['accuracy']
        result[f'original_acc{k}'] = before
        assert np.max(abs(result[f'delta_acc{k}_pp'] - 100 * (result[f'acc{k}'] - before))) < 1e-09
        result[f'correct{k}'] = np.rint(result[f'acc{k}'] * result.n).astype(int)
    result.to_csv(OUT / 'source_accuracy_retention.csv', index=False)
    cm = pd.read_csv(ROOT / 'preprocessing/variants/v0/retrained_fcn/metadata.csv')
    h = np.load(ROOT / 'preprocessing/variants/v0/retrained_fcn/contact_features.npz')['hidden_mean']
    rm = pd.read_csv(OLD / 'radar/metadata.csv')
    rz = np.load(OLD / 'radar/features.npz')
    x = rz['frame_shape']
    splits = []
    scalers = []
    source_h = cache['hidden_refs'][cache['split'] == 'train'].reshape(-1, 128)
    source_mean = source_h.mean(0)
    source_scale = np.maximum(source_h.std(0), 0.01)
    for fold, baud in [('115200_to_460800', 115200), ('460800_to_115200', 460800)]:
        head = np.load(ROOT / 'results/retained_head_v1/models/v0' / fold / 'retained_head.npz')
        ck = torch.load(ROOT / 'radar/stage_b_v0/models/four_class_provisional_roi' / fold / 'ce_feat_1_kd.pt', map_location='cpu', weights_only=False)
        ctrain = (cm.label >= 0) & (cm.baud == baud)
        ctest = (cm.label >= 0) & (cm.baud != baud)
        rtrain = (rm.label >= 0) & (rm.baud_candidate == baud)
        rtest = (rm.label >= 0) & (rm.baud_candidate != baud)
        ctr = set(cm.loc[ctrain, 'bag_id'])
        cte = set(cm.loc[ctest, 'bag_id'])
        rtr = set(rm.loc[rtrain, 'bag_id'])
        rte = set(rm.loc[rtest, 'bag_id'])
        assert ctr == rtr == set(head['local_training_bags']) == set(ck['train_bags'])
        assert cte == rte
        assert not ctr & cte
        cs = StandardScaler().fit(h[ctrain], sample_weight=equal_file_weights(cm[ctrain]))
        rs = StandardScaler().fit(x[rtrain], sample_weight=equal_file_weights(rm[rtrain]))
        rec = dict(fold=fold, contact_train_windows=int(ctrain.sum()), contact_test_windows=int(ctest.sum()), radar_train_windows=int(rtrain.sum()), radar_test_windows=int(rtest.sum()), train_bags=sorted(ctr), test_bags=sorted(cte), train_test_bag_intersection=[], known_same_bearing_entity_cross_split=True, known_same_main_motor_rig_cross_split=True, external_training_rows=0)
        splits.append(rec)
        sc = dict(fold=fold, source_head_mean_error=float(abs(head['source_scaler_mean'] - source_mean).max()), source_head_scale_error=float(abs(head['source_scaler_scale'] - source_scale).max()), contact_student_mean_error=float(abs(ck['contact_mean'] - cs.mean_).max()), contact_student_scale_error=float(abs(ck['contact_scale'] - cs.scale_).max()), radar_mean_error=float(abs(ck['radar_mean'] - rs.mean_).max()), radar_scale_error=float(abs(ck['radar_scale'] - rs.scale_).max()))
        assert max((v for k, v in sc.items() if k != 'fold')) < 1e-09
        scalers.append(sc)
    writej(OUT / 'local_split_membership.json', splits)
    pd.DataFrame(scalers).to_csv(OUT / 'scaler_reconstruction.csv', index=False)
    summary = json.loads((RAW / 'summary.json').read_text())
    file_records = []
    window_records = []
    frames = defaultdict(list)
    allhash = defaultdict(list)
    for item in summary:
        path = RAW / item['output']
        side = json.loads(path.with_suffix('.json').read_text())
        z = np.load(path)
        fh = filehash(path)
        allhash[fh].append(path.name)
        array = 'raw_xyz' if item['modality'] == 'contact' else 'iq_frames'
        samples = z[array]
        seen_frames = set()
        overlap = 0
        for i, value in enumerate(samples):
            wh = digest(value)
            window_records.append(dict(modality=item['modality'], file=path.name, row=i, sha256=wh))
            if item['modality'] == 'radar':
                first = int(side['windows'][i]['first_frame'])
                ids = set(range(first, first + 20))
                overlap += len(ids & seen_frames)
                seen_frames |= ids
                for j, fr in enumerate(value):
                    frames[digest(fr)].append((path.name, i, first + j))
        file_records.append(dict(file=path.name, modality=item['modality'], sha256=fh, n_windows=len(samples), same_file_frame_overlap=overlap, source_raw_filename=side['metadata']['file']))
        print('local hashed', path.name, len(samples), flush=True)
    pd.DataFrame(file_records).to_csv(OUT / 'local_file_hashes.csv', index=False)
    ww = pd.DataFrame(window_records)
    ww.to_csv(OUT / 'local_window_hashes.csv', index=False)
    duplicates = []
    for (mod, sha), a in ww.groupby(['modality', 'sha256']):
        if len(a) > 1:
            duplicates.append(dict(kind='whole_window', modality=mod, hash=sha, locations=a[['file', 'row']].to_dict('records')))
    for sha, positions in frames.items():
        if len(positions) > 1:
            duplicates.append(dict(kind='exported_iq_frame', modality='radar', hash=sha, locations=positions))
    writej(OUT / 'local_exact_duplicates.json', duplicates)
    writej(OUT / 'preliminary_summary.json', dict(read_only=True, source_schema=schema, source_self_reference_pairs=self_ref, replay_queries=len(replay), replay_self_reference_pairs=int(replayrefs.self_reference.sum()), local_npz_duplicate_groups=[v for v in allhash.values() if len(v) > 1], local_window_or_iq_frame_duplicate_groups=len(duplicates), local_within_file_overlapping_frame_occurrences=sum((x['same_file_frame_overlap'] for x in file_records)), all_source_query_ids_disjoint_between_splits=True, scalers_reconstructed_train_only=True, elapsed_seconds=time.monotonic() - started))
    hashes = defaultdict(list)
    h5manifest = []
    with h5py.File(ASSET / 'data.hdf5', 'r') as hf:
        d = hf['vibration']
        attributes = dict(hf.attrs)
        dattributes = dict(d.attrs)
        for begin in range(0, len(d), 256):
            block = d[begin:begin + 256]
            for k, row in enumerate(block):
                hashes[digest(row)].append(begin + k)
            if begin % 16384 == 0:
                print('MBHM DCN rows hashed', min(begin + 256, len(d)), 'elapsed', round(time.monotonic() - started, 1), flush=True)
    duplicated = []
    cross = []
    for sha, ids in hashes.items():
        if len(ids) < 2:
            continue
        rec = dict(sha256=sha, n_rows=len(ids), file_ids=ids, query_splits=[qsplit.get(i, 'excluded') for i in ids], sources=source.loc[ids, 'source'].tolist(), labels=source.loc[ids, 'label'].astype(int).tolist())
        duplicated.append(rec)
        if len(set(rec['query_splits']) & {'train', 'val', 'test'}) > 1:
            cross.append(rec)
    writej(OUT / 'source_exact_duplicate_rows.json', duplicated)
    writej(OUT / 'source_cross_split_exact_duplicate_rows.json', cross)
    sourcerowhash = pd.DataFrame([dict(sha256=sha, file_id=i, query_split=qsplit.get(i, 'excluded')) for sha, ids in hashes.items() for i in ids])
    sourcerowhash.to_csv(OUT / 'source_dcn_row_hashes.csv', index=False)
    final = dict(source_dcn_rows_hashed=int(len(source)), source_dcn_row_bytes=24000 * 4, source_exact_duplicate_groups=len(duplicated), source_exact_duplicate_rows=sum((r['n_rows'] for r in duplicated)), source_cross_split_exact_duplicate_groups=len(cross), source_cross_split_exact_duplicate_rows=sum((r['n_rows'] for r in cross)), hdf5_file_attributes=attributes, hdf5_dataset_attributes=dattributes, physical_origin_available=False, limitation='Byte-distinct derived DCN windows may still overlap in original raw recording or share physical bearing; no original time/start/recording fields available.', elapsed_seconds=time.monotonic() - started)
    writej(OUT / 'source_duplicate_summary.json', final)
    print(json.dumps(final), flush=True)
if __name__ == '__main__':
    main()
