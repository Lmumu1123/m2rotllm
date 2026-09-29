"""Checks source independence, internal dedup, Qwen chunks, exclusions, tamper rejection."""
from pathlib import Path
import importlib.util, json, tempfile, gzip, hashlib, subprocess, sys, os
SCRIPT = Path(__file__).with_name('export_model_vault.py')
def sha(data): return hashlib.sha256(data).hexdigest()
with tempfile.TemporaryDirectory(prefix='c2r-vault-test-') as td:
    base=Path(td); repo=base/'repo'; source=base/'original'; repo.mkdir(); source.mkdir()
    (repo/'BearLLM/weights').mkdir(parents=True)
    a=b'unique-checkpoint-content'*7
    entries=[]
    for name in ['a.pt','b.pt']:
        s=source/name;s.write_bytes(a)
        (repo/'BearLLM/weights'/name).write_bytes(a)
        entries.append(dict(group='BearLLM',relative_path='weights/'+name,source_path=str(s),
                            role='project_trained_checkpoint_or_learned_mapping_keep',size_bytes=len(a),sha256=sha(a)))
    (repo/'BearLLM/model.py').write_text('class Encoder: pass\n')
    (repo/'.env').write_text('DO_NOT_COPY=fake-secret\n')
    (repo/'large_features.npz').write_bytes(b'data-do-not-copy')
    qwen=b'qwen-checkpoint-test-contents'*101
    rel='external/bearllm-assets/qwen_weights/model.safetensors'
    parts=[]
    for index,content in enumerate([qwen[:1300],qwen[1300:]]):
        p=repo/f'migration/model_parts/qwen_model/part-{index:03d}.gz';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(gzip.compress(content,mtime=0))
        parts.append(dict(path=p.relative_to(repo).as_posix(),bytes=len(content),sha256=sha(content),archive_sha256=sha(p.read_bytes())))
    mp=repo/'migration/manifests/qwen_parts.json';mp.parent.mkdir(exist_ok=True)
    mp.write_text(json.dumps(dict(target=rel,sha256=sha(qwen),parts=parts)))
    entries.append(dict(group='bearllm-assets',relative_path='qwen_weights/model.safetensors',
        source_path=str(source/'missing-qwen'),role='public_base_llm_pinned_download',size_bytes=len(qwen),sha256=sha(qwen)))
    inventory=base/'inventory.json';inventory.write_text(json.dumps(dict(entries=entries,associated_support_files=[])))
    vault=base/'vault'
    result=subprocess.run([sys.executable,str(SCRIPT),'--repo',str(repo),'--inventory',str(inventory),
        '--target',str(vault),'--no-original-fallback'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr+result.stdout
    m=json.loads((vault/'MODEL_VAULT_MANIFEST.json').read_text())
    assert m['status']=='complete_verified' and m['stats']['model_paths']==3
    assert m['stats']['unique_model_contents']==2
    assert len(m['files'])==4, m
    aa=vault/'files/BearLLM/weights/a.pt';bb=vault/'files/BearLLM/weights/b.pt'
    assert aa.stat().st_ino==bb.stat().st_ino
    assert aa.stat().st_ino!=(repo/'BearLLM/weights/a.pt').stat().st_ino
    assert aa.stat().st_ino!=(source/'a.pt').stat().st_ino
    assert (vault/'files'/rel).read_bytes()==qwen
    verify=subprocess.run([sys.executable,str(SCRIPT),'--target',str(vault),'--verify-only'],capture_output=True,text=True)
    assert verify.returncode==0,verify.stderr
    # Deliberate modification is limited to temporary test output, never original input.
    aa.chmod(0o644);aa.write_bytes(b'tamper')
    bad=subprocess.run([sys.executable,str(SCRIPT),'--target',str(vault),'--verify-only'],capture_output=True,text=True)
    assert bad.returncode!=0
    assert (source/'a.pt').read_bytes()==a and (repo/'BearLLM/weights/a.pt').read_bytes()==a
    # Even a syntactically valid partial manifest must never verify successfully.
    m['status']='failed_incomplete_do_not_delete_sources';(vault/'MODEL_VAULT_MANIFEST.json').write_text(json.dumps(m))
    incomplete=subprocess.run([sys.executable,str(SCRIPT),'--target',str(vault),'--verify-only'],capture_output=True,text=True)
    assert incomplete.returncode!=0 and 'Vault is incomplete' in incomplete.stderr
print('passed: source inode isolation, internal SHA dedup, chunk reconstruction, exclusions, content tamper, incomplete vault rejection')
