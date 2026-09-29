"""Exercise missing-input safeguards and feature-API reuse; never fabricate a capture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from extract_features import digest

ROOT=Path(__file__).resolve().parent
SCRIPT=ROOT.parent/'scripts/reprocess_geometry.py'
SOURCE=ROOT.parent.parent/'four_class_preprocessing_20260921/results'
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='2'
before=digest(ROOT/'features.npz')
checks={}
with tempfile.TemporaryDirectory(prefix='radar_geometry_api_check_') as temp:
    base=Path(temp);absent=base/'must_not_be_created'
    common=[sys.executable,str(SCRIPT),'--raw-dir',str(SOURCE),'--output',str(absent)]
    p=subprocess.run(common+['--check-only'],text=True,capture_output=True,env=env)
    doc=json.loads(p.stdout)
    assert p.returncode==2 and len(doc['missing_raw_files'])==12 and not absent.exists()
    checks['real_missing_check_only']=dict(exit_code=p.returncode,missing_files=doc['missing_raw_files'],output_created=False)
    p=subprocess.run(common,text=True,capture_output=True,env=env)
    assert p.returncode==2 and not absent.exists()
    checks['real_missing_run']=dict(exit_code=p.returncode,output_created=False)
    p=subprocess.run([sys.executable,str(ROOT/'extract_features.py'),'--source',str(SOURCE),
        '--output',str(ROOT)],text=True,capture_output=True,env=env)
    assert p.returncode==2 and digest(ROOT/'features.npz')==before
    checks['existing_feature_output_refused']=True
    out=base/'new_features'
    p=subprocess.run([sys.executable,str(ROOT/'extract_features.py'),'--source',str(SOURCE),
        '--output',str(out),'--distance-m','.45'],text=True,capture_output=True,env=env)
    if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    with np.load(ROOT/'features.npz',allow_pickle=False) as old, np.load(out/'features.npz',allow_pickle=False) as new:
        assert set(old.files)==set(new.files)
        for key in old.files:np.testing.assert_array_equal(old[key],new[key],err_msg=key)
    checks['new_source_output_api']=dict(status='pass',all_feature_arrays_identical=True,old_output_untouched=True)
assert digest(ROOT/'features.npz')==before
checks['status']='passed'
checks['raw_reprocessing_execution']='not executed: all 12 original ADC bin are absent; no substitute/fabricated signals used'
(ROOT/'geometry_reprocessing_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
