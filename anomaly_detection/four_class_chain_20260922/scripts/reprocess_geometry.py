"""Re-export all known radar recordings with one physical ROI; preserve contacts.

No radar hardware is accessed. Missing/duplicate/changed raw files stop the job.
Check-only is read-only and does not create output directories.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

CHAIN=Path(__file__).resolve().parents[1]
PRE=CHAIN.parent/'four_class_preprocessing_20260921'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stat(path):
    s=path.stat();return (s.st_size,s.st_mtime_ns)

def load_preprocessor():
    spec=importlib.util.spec_from_file_location('original_four_class_preprocess',PRE/'preprocess.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def inventory(args,pre):
    summary=read(args.existing_results/'summary.json')
    bags=read(args.existing_results/'recording_bags.json')
    original_manifest=read(args.existing_results/'input_manifest.json')
    reference={r['file']:r for r in original_manifest}
    radar=[r for r in summary if r['modality']=='radar']
    contact=[r for r in summary if r['modality']=='contact']
    if len(radar)!=12 or len(contact)!=12:
        raise ValueError('Expected exactly 12 archived radar and 12 contact recordings; source is not this dataset')
    names={r['file'] for r in radar}
    located={name:[] for name in names}
    if args.raw_dir.is_dir():
        for p in args.raw_dir.rglob('*.bin'):
            if p.name in located:located[p.name].append(p.resolve())
    missing=sorted(name for name,paths in located.items() if not paths)
    duplicates={name:list(map(str,paths)) for name,paths in located.items() if len(paths)>1}
    sizes=[];missing_reference=[]
    found={name:paths[0] for name,paths in located.items() if len(paths)==1}
    for name,p in sorted(found.items()):
        if name not in reference:missing_reference.append(name);continue
        expected=reference[name]['bytes'];actual=p.stat().st_size
        if expected!=actual:sizes.append(dict(file=name,expected_bytes=expected,actual_bytes=actual))
    missing_contact=[]
    for item in contact:
        for filename in [item['output'],str(Path(item['output']).with_suffix('.json'))]:
            if not (args.existing_results/filename).is_file():missing_contact.append(filename)
    ids={r['output'] for r in radar};cids={r['output'] for r in contact}
    if len(bags)!=12 or {b['radar'] for b in bags}!=ids or {b['contact'] for b in bags}!=cids:
        raise ValueError('Archived recording_bags does not match the 12 radar/contact identities')
    pre.check_config(args.config)
    fail=bool(missing or duplicates or sizes or missing_reference or missing_contact)
    audit=dict(status='blocked_missing_or_invalid_inputs' if fail else 'inventory_ok_hashes_not_yet_checked',
        check_only=args.check_only,raw_dir=str(args.raw_dir),raw_dir_exists=args.raw_dir.is_dir(),
        output=str(args.output),existing_results=str(args.existing_results),required_radar_files=12,
        found_unique_radar_files=len(found),missing_raw_files=missing,duplicate_raw_files=duplicates,
        size_mismatches=sizes,missing_archived_raw_manifest=missing_reference,
        missing_contact_exports=missing_contact,contact_exports_reused=12,
        radar_config_checked='exact supported R0 fields',config=str(args.config),
        shared_distance_m=args.distance_m,shared_roi_m=[max(.04,args.distance_m-.16),args.distance_m+.16],
        sha256_policy='Full raw SHA256 checked against archived manifest before any output is created',
        synchronized_window_pairs=False)
    return audit,radar,contact,bags,reference,found

def run(args,pre,audit,radar,contact,bags,reference,found):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be a new or empty directory; existing outputs are never overwritten')
    # Validate complete source identity before allocating result output.
    inputs=[];raw_stats={}
    for name,path in sorted(found.items()):
        print(f'Checking raw SHA256: {name}',flush=True)
        before=stat(path);digest=sha(path)
        if before!=stat(path):raise RuntimeError(f'Raw changed while hashing: {path}')
        if digest!=reference[name]['sha256']:
            raise ValueError(f'Raw SHA256 differs from archived capture: {name}; stopping before output creation')
        raw_stats[name]=before
        inputs.append(dict(file=name,path=str(path),kind='raw_radar',bytes=before[0],mtime_ns=before[1],sha256=digest))
    archived_verification=read(args.existing_results/'verification.json')
    expected_exports={v['file']:v['sha256'] for v in archived_verification['files']}
    contact_sources=[]
    for item in contact:
        npz=args.existing_results/item['output']
        if sha(npz)!=expected_exports.get(npz.name):
            raise ValueError(f'Contact export changed from archived verification: {npz.name}')
        for path in [npz,npz.with_suffix('.json')]:
            before=stat(path);digest=sha(path)
            if before!=stat(path):raise RuntimeError(f'Contact export changed while hashing: {path}')
            contact_sources.append((path,before,digest))
    args.output.mkdir(parents=True,exist_ok=True)
    args._initialized_output=True
    results=args.output/'results';results.mkdir()
    provenance=args.output/'provenance';provenance.mkdir()
    for filename in ['summary.json','recording_bags.json','input_manifest.json','runtime.json','verification.json']:
        shutil.copy2(args.existing_results/filename,provenance/f'original_{filename}')
    shutil.copy2(args.config,provenance/'radar_config.cfg')
    shutil.copy2(PRE/'preprocess.py',provenance/'preprocess.py')
    shutil.copy2(Path(__file__),provenance/'reprocess_geometry.py')
    shutil.copy2(CHAIN/'radar/extract_features.py',provenance/'extract_features.py')
    write(args.output/'inventory.json',audit)
    write(args.output/'status.json',dict(status='running',stage='copy_contact_exports'))
    for source,before,digest in contact_sources:
        dest=results/source.name;shutil.copy2(source,dest)
        if before!=stat(source) or sha(dest)!=digest:
            raise RuntimeError(f'Contact changed or copy verification failed: {source.name}')
        inputs.append(dict(file=source.name,path=str(source),kind='reused_contact_export',
            bytes=before[0],mtime_ns=before[1],sha256=digest))
    output_summary=list(contact)
    for item in radar:
        source=found[item['file']]
        if stat(source)!=raw_stats[source.name]:raise RuntimeError(f'Raw changed before export: {source}')
        print(f'Reprocessing shared ROI: {source.name}',flush=True)
        write(args.output/'status.json',dict(status='running',stage='radar_export',file=source.name))
        exported=pre.radar_export(source,results/item['output'],args.distance_m)
        if stat(source)!=raw_stats[source.name]:raise RuntimeError(f'Raw changed during export: {source}')
        output_summary.append(exported)
        write(results/'summary.json',output_summary)
    write(results/'recording_bags.json',bags)
    write(results/'input_manifest.json',inputs)
    runtime=dict(python=sys.version,executable=sys.executable,command=sys.argv,
        distance_m=args.distance_m,shared_roi_m=audit['shared_roi_m'],
        preprocessing_sha256=sha(PRE/'preprocess.py'),config_sha256=sha(args.config),
        wrapper_sha256=sha(Path(__file__)),feature_script_sha256=sha(CHAIN/'radar/extract_features.py'),
        contacts='byte-identical copied NPZ/JSON; raw DAT not required',
        pairing='original candidate recording bags preserved; no window alignment inferred',
        raw_radar_sha256='all 12 match original manifest before export',threads=2)
    write(results/'runtime.json',runtime)
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='2'
    write(args.output/'status.json',dict(status='running',stage='verify_exports'))
    subprocess.run([sys.executable,str(PRE/'verify_exports.py'),str(results)],check=True,env=env)
    write(args.output/'status.json',dict(status='running',stage='radar_features'))
    subprocess.run([sys.executable,str(CHAIN/'radar/extract_features.py'),'--source',str(results),
        '--output',str(args.output/'radar'),'--distance-m',str(args.distance_m)],check=True,env=env)
    for name,path in found.items():
        if stat(path)!=raw_stats[name]:raise RuntimeError(f'Input changed before completion: {name}')
    write(args.output/'status.json',dict(status='complete',radar_recordings=12,contact_recordings_reused=12,
        raw_inputs_unmodified=True,radar_features=str(args.output/'radar/features.npz'),
        next_step='Review selected range, saturation and plots before retraining; models were not rerun'))
    print(f'Complete: {args.output}',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,required=True,help='Folder with all 12 original .bin, recursively searched')
    parser.add_argument('--output',type=Path,required=True,help='New/empty output directory, untouched in check-only mode')
    parser.add_argument('--distance-m',type=float,default=.45)
    parser.add_argument('--config',type=Path,default=PRE/'雷达原始配置.cfg')
    parser.add_argument('--existing-results',type=Path,default=PRE/'results')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    args._initialized_output=False
    for key in ['raw_dir','output','config','existing_results']:setattr(args,key,getattr(args,key).resolve())
    if not math.isfinite(args.distance_m) or args.distance_m<=.04:
        parser.error('--distance-m must be finite and >0.04 m')
    try:
        pre=load_preprocessor();items=inventory(args,pre);audit=items[0]
        print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
        if audit['status'].startswith('blocked'):return 2
        if args.check_only:return 0
        run(args,pre,*items)
        return 0
    except Exception as exc:
        if args._initialized_output:
            write(args.output/'status.json',dict(status='failed',error_type=type(exc).__name__,message=str(exc)))
        print(json.dumps(dict(status='failed',error_type=type(exc).__name__,message=str(exc)),ensure_ascii=False),file=sys.stderr)
        return 1

if __name__=='__main__':raise SystemExit(main())
