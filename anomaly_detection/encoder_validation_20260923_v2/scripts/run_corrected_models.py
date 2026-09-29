"""Replay old fixed algorithms with corrected radar inputs in separate outputs."""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import hashlib, importlib.util, json, subprocess, sys
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parents[1]
OLD = ROOT.parent / 'encoder_validation_20260923'

def sha(p):
    return hashlib.sha256(Path(_c2r_resolve_path(p)).read_bytes()).hexdigest()

def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main():
    assert sys.version_info >= (3, 10)
    state = json.loads((ROOT / 'geometry_corrected/status.json').read_text())
    assert state['status'] == 'complete'
    before = {str(p): sha(p) for p in [OLD / 'controls/run_controls.py', OLD / 'rotllm/run_radar_transfer.py', OLD / 'scripts/contact_only_stitch.py', ROOT / 'protocol.json']}
    (ROOT / 'controls').mkdir(exist_ok=True)
    if not (ROOT / 'controls/results/verification.json').exists():
        mod = load(OLD / 'controls/run_controls.py', 'corrected_controls')
        mod.ROOT = ROOT / 'controls'
        mod.RADAR = ROOT / 'geometry_corrected/radar'
        mod.main()
    rot_src = (OLD / 'rotllm/run_radar_transfer.py').read_text()
    rot_src = rot_src.replace('B=Path(__file__).resolve().parent', f"B=Path({str(OLD / 'rotllm')!r})\nV2=Path({str(ROOT)!r})")
    rot_src = rot_src.replace("OUT=B/'radar'", "OUT=V2/'rotllm_radar'")
    rot_src = rot_src.replace("B.parent/'protocol.json'", "V2/'protocol.json'")
    rot_script = ROOT / 'scripts/run_corrected_rotllm.py'
    rot_script.write_text(rot_src)
    if not (ROOT / 'rotllm_radar/verification.json').exists():
        subprocess.run([sys.executable, str(rot_script)], check=True)
    stitch_src = (OLD / 'scripts/contact_only_stitch.py').read_text()
    stitch_src = stitch_src.replace("ROT = ROOT/'rotllm'", f"ROT = Path({str(OLD / 'rotllm')!r})")
    stitch_script = ROOT / 'scripts/run_corrected_stitch.py'
    stitch_script.write_text(stitch_src)
    if not (ROOT / 'results/contact_only_stitch/verification.json').exists():
        subprocess.run([sys.executable, str(stitch_script)], check=True)
    assert before == {p: sha(p) for p in before}
    provenance = dict(original_sources=before, original_files_unchanged=True, changes='Only radar source paths and new output paths; architecture, head checkpoints, preprocessing target variant, losses, steps and seeds unchanged.', generated_script_sha={str(p): sha(p) for p in [rot_script, stitch_script]}, controls_invocation='import old module, override ROOT output and RADAR input, run unchanged main()', caveat='Old imported protocol text about invalid outer ROI is historical; see v2 protocol and actual corrected metadata for current inputs.')
    (ROOT / 'results/replay_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
if __name__ == '__main__':
    main()
