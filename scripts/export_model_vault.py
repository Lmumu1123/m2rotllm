#!/usr/bin/env python3
"""Copy and verify C2R model files into an independent, deduplicated vault.

No source file is changed or removed. Hard links are made ONLY inside the new
vault, after copying original bytes to a new vault-owned inode. Python stdlib.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import uuid

ORIGINAL_ROOTS = {
    'BearLLM': '/home/huangyating/BearLLM',
    'anomaly_detection': '/home/huangyating/anomaly_detection',
    'bearllm-runs': '/media/nas_users/huangyating/bearllm-runs',
    'bearllm-assets': '/media/nas_users/huangyating/bearllm-assets',
}
DEST_PREFIX = {
    'BearLLM': 'BearLLM', 'anomaly_detection': 'anomaly_detection',
    'bearllm-runs': 'external/bearllm-runs',
    'bearllm-assets': 'external/bearllm-assets',
}
SKIP_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache',
             '.ruff_cache', '.cache', 'node_modules', '.browser_check',
             '.render_dependencies', 'model_parts'}
CODE_EXTENSIONS = {'.py', '.sh', '.cfg', '.yaml', '.yml', '.toml', '.ini'}
SMALL_PROTOCOL_NAMES = {
    'protocol.json', 'protocol_before_results.json', 'splits.json', 'config.json',
    'schema.json', 'feature_schema.json', 'input_schema.json',
    'teacher_interfaces.json', 'provenance.json', 'external_reference_protocol.json',
    'adapter_config.json', 'trainer_state.json', 'dataset.json',
    'verification.json',
}
# Kept inventory support files include the nine-reference bank and l3.npy.
# Pure numeric evaluation output is not needed to retain model parameters.
EXCLUDED_SUPPORT_NAMES = {'baseline_metrics_detail.json'}


def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def writej(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(t in ('..', '') for t in p.parts):
        raise ValueError(f'Unsafe destination path: {value!r}')
    return p.as_posix()


def original_to_relative(path):
    p = Path(path)
    for group, base in ORIGINAL_ROOTS.items():
        try:
            suffix = p.relative_to(base)
        except ValueError:
            continue
        return safe_relative((PurePosixPath(DEST_PREFIX[group]) / suffix.as_posix()).as_posix())
    raise ValueError(f'Unknown original path prefix: {path}')


def find_inventory(repo, explicit):
    if explicit:
        return Path(explicit).resolve()
    for name in ['migration/manifests/model_inventory.json',
                 'migration/model_inventory.json', 'model_inventory.json']:
        p = repo / name
        if p.is_file():
            return p
    raise FileNotFoundError('Pass --inventory with the migration model_inventory.json path')


def build_plan(repo, inventory, allow_original=True):
    planned = {}
    skipped = []
    for entry in inventory['entries']:
        rel = safe_relative(f"{DEST_PREFIX[entry['group']]}/{entry['relative_path']}")
        candidates = [repo / rel]
        if allow_original:
            candidates.append(Path(entry['source_path']))
        source = next((p for p in candidates if p.is_file()), None)
        parts_manifest = None
        if source is None and entry['role'] == 'public_base_llm_pinned_download':
            mp = repo / 'migration/manifests/qwen_parts.json'
            if mp.is_file():
                part_info = readj(mp)
                if part_info['target'] != rel or part_info['sha256'] != entry['sha256']:
                    raise ValueError('Qwen parts manifest does not match model inventory')
                for part in part_info['parts']:
                    pp = repo / safe_relative(part['path'])
                    if not pp.is_file():
                        raise FileNotFoundError(pp)
                parts_manifest = str(mp)
        if source is None and parts_manifest is None:
            raise FileNotFoundError(f'Model is missing from repository and original server: {rel}')
        if source is not None and source.stat().st_size != entry['size_bytes']:
            raise ValueError(f'Model byte count changed: {source}')
        planned[rel] = dict(relative_restore_path=rel, source_path=str(source) if source else None,
                            original_source_path=entry['source_path'], category='model',
                            role=entry['role'], expected_sha256=entry['sha256'],
                            size_bytes=entry['size_bytes'], parts_manifest=parts_manifest)
    for original in inventory.get('associated_support_files', []):
        rel = original_to_relative(original)
        if Path(original).name in EXCLUDED_SUPPORT_NAMES:
            skipped.append({'path':original, 'reason':'evaluation metrics, not a model dependency'})
            continue
        candidates = [repo / rel]
        if allow_original:
            candidates.append(Path(original))
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(f'Model support file is missing: {original}')
        planned.setdefault(rel, dict(relative_restore_path=rel, source_path=str(source),
                                     original_source_path=original, category='support',
                                     role='paired_model_configuration_reference_or_training_state',
                                     expected_sha256=None, size_bytes=source.stat().st_size,
                                     parts_manifest=None))
    # Use the migrated, portable code, not the original hard-coded-path version.
    for base, dirs, files in os.walk(repo, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            p = Path(base) / name
            rel = p.relative_to(repo).as_posix()
            if p.is_symlink():
                # Data/model symlinks never silently enter the support-code snapshot.
                continue
            is_code = p.suffix in CODE_EXTENSIONS
            is_environment = rel.startswith('migration/environment/') or (
                name.startswith(('requirements', 'environment')) and p.suffix in {'.txt','.json','.yaml','.yml'})
            is_notice = name in {'LICENSE', 'LICENSE.md', 'NOTICE', '.env.example'}
            is_protocol = name in SMALL_PROTOCOL_NAMES and p.stat().st_size <= 1024 * 1024
            if not any([is_code, is_environment, is_notice, is_protocol]):
                continue
            category = 'code' if is_code else 'environment' if is_environment else 'support'
            planned.setdefault(rel, dict(relative_restore_path=rel, source_path=str(p),
                                         original_source_path=None, category=category,
                                         role='portable_source_environment_or_protocol',
                                         expected_sha256=None, size_bytes=p.stat().st_size,
                                         parts_manifest=None))
    # Keep the exact inventory used to select all trained heads and learned mappings.
    return sorted(planned.values(), key=lambda x: (x['category'] != 'model', x['relative_restore_path'])), skipped


def copy_model_to_object(plan, repo, object_dir, object_cache):
    expected = plan['expected_sha256']
    if plan['source_path']:
        source = Path(plan['source_path'])
        actual = sha256(source)
        if expected is not None and actual != expected:
            raise ValueError(f'Model hash changed: {source}')
        destination = object_dir / actual[:2] / actual
        if actual not in object_cache:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(destination.name + '.copying-' + uuid.uuid4().hex)
            with source.open('rb') as inp, temp.open('xb') as out:
                shutil.copyfileobj(inp, out, length=8 * 1024 * 1024)
            if sha256(temp) != actual:
                raise ValueError(f'Copied hash mismatch: {source}')
            # A fresh copy is essential: never hard-link the original checkpoint.
            if os.path.samestat(source.stat(), temp.stat()):
                raise RuntimeError('Unsafe source hard-link detected')
            os.replace(temp, destination)
            destination.chmod(0o444)
            object_cache[actual] = destination
        else:
            destination = object_cache[actual]
            if os.path.samestat(source.stat(), destination.stat()):
                raise RuntimeError('Vault unexpectedly shares an inode with the source')
    else:
        info = readj(plan['parts_manifest'])
        actual = expected
        destination = object_dir / actual[:2] / actual
        if actual not in object_cache:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(destination.name + '.copying-' + uuid.uuid4().hex)
            full_hash = hashlib.sha256()
            total_bytes = 0
            with temp.open('xb') as out:
                for part in info['parts']:
                    source = repo / safe_relative(part['path'])
                    if sha256(source) != part['archive_sha256']:
                        raise ValueError(f'Qwen compressed part hash mismatch: {source}')
                    part_hash, part_bytes = hashlib.sha256(), 0
                    with gzip.open(source, 'rb') as inp:
                        for data in iter(lambda: inp.read(8 * 1024 * 1024), b''):
                            full_hash.update(data); part_hash.update(data)
                            part_bytes += len(data); total_bytes += len(data)
                            out.write(data)
                    if part_hash.hexdigest() != part['sha256'] or part_bytes != part['bytes']:
                        raise ValueError(f'Qwen uncompressed part mismatch: {source}')
            if full_hash.hexdigest() != expected or total_bytes != plan['size_bytes']:
                raise ValueError('Assembled Qwen checkpoint does not match inventory')
            if sha256(temp) != expected:
                raise ValueError('Assembled Qwen file failed disk verification')
            os.replace(temp, destination)
            destination.chmod(0o444)
            object_cache[actual] = destination
    if destination.stat().st_size != plan['size_bytes']:
        raise ValueError(f'Object size mismatch: {destination}')
    return actual, destination


def verify_vault(target, allow_pending=False):
    manifest = readj(target / 'MODEL_VAULT_MANIFEST.json')
    allowed = {'complete_verified'}
    if allow_pending:
        allowed.add('copied_pending_final_verification')
    if manifest.get('status') not in allowed:
        raise ValueError('Vault is incomplete; do not delete original files')
    if len(manifest['files']) != manifest['expected_paths']:
        raise ValueError('Vault file count is incomplete')
    if sha256(target / 'SOURCE_MODEL_INVENTORY.json') != manifest['inventory_sha256']:
        raise ValueError('Source model inventory copy has changed')
    verified_objects = {}
    available_sources = 0
    total = 0
    for item in manifest['files']:
        p = target / safe_relative(item['vault_path'])
        if not p.is_file() or p.is_symlink():
            raise FileNotFoundError(f'Missing or unexpected symlink: {p}')
        stat = p.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        # Duplicate paths are verified against the same freshly hashed inode.
        actual = verified_objects.get(key)
        if actual is None:
            actual = sha256(p)
            verified_objects[key] = actual
        if actual != item['sha256'] or stat.st_size != item['size_bytes']:
            raise ValueError(f'Vault verification failed: {p}')
        original = item.get('source_path')
        if original and Path(original).is_file():
            available_sources += 1
            if os.path.samestat(stat, Path(original).stat()):
                raise ValueError(f'Unsafe link to source file: {p}')
        total += 1
    return {'status':'passed', 'verified_file_paths':total,
            'verified_unique_inodes':len(verified_objects),
            'checked_available_sources_have_independent_inodes':True,
            'original_source_paths_checked':available_sources}


def make_readme(target, stats):
    text = f'''# C2R 模型保留包

生成时间：{datetime.now(timezone.utc).isoformat()}。

本目录是模型、参数和必要配置/源码的保留包，不是完整结果库或完整训练数据集。模型文件 {stats['model_paths']} 份（{stats['unique_model_contents']} 个不同内容），辅助文件 {stats['auxiliary_paths']} 份；全部内容按 SHA-256 去重后 {stats['unique_stored_bytes']:,} 字节。

- `MODEL_VAULT_MANIFEST.json` 逐项记录原路径、Git 仓库恢复相对路径、角色、SHA-256、文件大小和 vault 路径。
- `files/` 保留与迁移仓库相同的目录结构，包括分类头、scaler、物理读取器、模型间转换、雷达编码器和可继续训练的状态。
- `objects/` 中每个 SHA-256 内容只存一份；`files/` 中的相同内容通过 **vault 内部硬链接**去重。所有内容首先从源文件复制到新 inode，原目录文件与此包没有硬链接。
- Qwen `model.safetensors` 是完整文件，基座配置、tokenizer 和 Bear 的 `l3.npy` 一并保留。
- 9 条固定公开健康参考也在 `files/anomaly_detection/four_class_chain_20260922/contact/` 内，基本接触推理不必重新读取 13 GB MBHM。
- `dataset.json` 若出现只保存训练样本/参考 ID 划分；未收集原始波形、完整数据集、大型特征缓存和实验预测结果。

## 在另一台主机继续实验

先克隆 `https://github.com/Lmumu1123/m2rotllm` 的 C2R 分支，按仓库迁移说明恢复环境和所需数据。完整程序、实验说明及结果以 Git 仓库为准；本包是服务器上的模型备份与恢复来源，不替代 Git 项目。

需要从本包恢复某个权重时，按 manifest 中 `relative_restore_path` 拷贝到新仓库对应位置。例如将 `files/external/bearllm-runs/...` 拷贝至新仓库的 `external/bearllm-runs/...`。**用普通复制（如 `cp --reflink=auto` 或 Python `shutil.copy2`），不要建立指向 vault 的硬链接。**不要覆盖已有不同模型，先比较 SHA。

这个包的内容设为只读，避免后续微调覆盖归档。开始新训练应把所需权重复制到工作目录并另设输出路径。若修改 vault 中某个共享 inode，会同时影响它的重复别名；因此不应在此目录直接训练写 checkpoint。

## 校验与清理

运行仓库的 `scripts/export_model_vault.py --verify-only --target "{target}"`，可逐项重新校验。验证相同 inode 的多个恢复路径时，只读一次内容哈希，但每条路径都会检查其大小、SHA 对应关系以及是否错误链接到原始文件。

本工具没有删除源文件。只有在 Git 上传成功、另一主机恢复验证通过、数据另行迁走之后，才可按清理说明删除旧工作副本。不要删除本目录、唯一原始数据或尚未迁出的 `data/upload` / `wide_cache_local`。

此包保留实验模型，不代表所有模型已达到验收指标。R2 诊断头未通过旧源域准确率下降≤0.5百分点的验证；最新跨转速雷达结果也尚未达到95%，这些标注随 manifest 角色保留。
'''
    (target / 'MODEL_VAULT_README.md').write_text(text, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, help='Migrated C2R repository root')
    ap.add_argument('--inventory', type=Path, help='model_inventory.json; auto-detected under migration/manifests')
    ap.add_argument('--target', type=Path, required=True, help='NEW vault directory; no existing directory is overwritten')
    ap.add_argument('--dry-run', action='store_true', help='Validate the plan and print counts without copying')
    ap.add_argument('--verify-only', action='store_true', help='Verify an already exported vault; requires only --target')
    ap.add_argument('--no-original-fallback', action='store_true', help='Use repository files/chunks only, never original server paths')
    args = ap.parse_args()
    target = args.target.expanduser().resolve()
    if args.verify_only:
        print(json.dumps(verify_vault(target),ensure_ascii=False,indent=2)); return
    if args.repo is None:
        ap.error('--repo is required unless --verify-only is used')
    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    if target == repo or repo in target.parents:
        raise ValueError('Place the vault outside the Git repository')
    inventory_path = find_inventory(repo, args.inventory)
    inventory = readj(inventory_path)
    plan, skipped = build_plan(repo, inventory, not args.no_original_fallback)
    summary = {'planned_paths':len(plan), 'models':sum(x['category']=='model' for x in plan),
               'auxiliary':sum(x['category']!='model' for x in plan),
               'logical_bytes':sum(x['size_bytes'] for x in plan),
               'model_inventory':str(inventory_path), 'repo':str(repo), 'target':str(target),
               'skipped_support_files':skipped, 'changes_to_original_files':False}
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:
        return
    if target.exists():
        raise FileExistsError('Target already exists; verify it or choose another NEW target. Nothing was overwritten.')
    target.mkdir(parents=True)
    object_dir = target / 'objects'
    object_dir.mkdir()
    manifest = {'format_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
                'status':'building','expected_paths':len(plan),'repo':str(repo),'inventory_sha256':sha256(inventory_path),
                'source_files_modified_or_deleted':False,
                'hardlink_policy':'copy to independent vault inode first; hardlink duplicates only within vault',
                'skipped_support_files':skipped,'files':[]}
    writej(target/'MODEL_VAULT_MANIFEST.json',manifest)
    shutil.copy2(inventory_path,target/'SOURCE_MODEL_INVENTORY.json')
    object_cache = {}
    try:
        for index, item in enumerate(plan):
            digest,obj = copy_model_to_object(item,repo,object_dir,object_cache)
            alias = target/'files'/item['relative_restore_path']
            alias.parent.mkdir(parents=True,exist_ok=True)
            os.link(obj,alias)
            if not os.path.samestat(alias.stat(),obj.stat()):
                raise RuntimeError('Vault internal deduplication link failed')
            record = dict(item)
            record.pop('expected_sha256')
            record.update(sha256=digest,vault_path=alias.relative_to(target).as_posix(),
                          object_path=obj.relative_to(target).as_posix())
            manifest['files'].append(record)
            if (index+1)%50==0:
                writej(target/'MODEL_VAULT_MANIFEST.json',manifest)
                print(f'copied and verified {index+1}/{len(plan)} paths',flush=True)
        models=[x for x in manifest['files'] if x['category']=='model']
        stats={'model_paths':len(models),'unique_model_contents':len({x['sha256'] for x in models}),
               'model_logical_bytes':sum(x['size_bytes'] for x in models),
               'auxiliary_paths':len(manifest['files'])-len(models),
               'auxiliary_logical_bytes':sum(x['size_bytes'] for x in manifest['files'] if x['category']!='model'),
               'unique_content_objects':len(object_cache),
               'unique_stored_bytes':sum(p.stat().st_size for p in object_cache.values()),
               'categories':dict(Counter(x['category'] for x in manifest['files']))}
        manifest['stats']=stats
        manifest['status']='copied_pending_final_verification'
        writej(target/'MODEL_VAULT_MANIFEST.json',manifest)
        verification=verify_vault(target,allow_pending=True)
        manifest['verification']=verification
        manifest['status']='complete_verified'
        writej(target/'MODEL_VAULT_MANIFEST.json',manifest)
        make_readme(target,stats)
        print(json.dumps({'stats':stats,'verification':verification,'status':manifest['status']},ensure_ascii=False,indent=2))
    except Exception as exc:
        manifest['status']='failed_incomplete_do_not_delete_sources'
        manifest['failure']=f'{type(exc).__name__}: {exc}'
        writej(target/'MODEL_VAULT_MANIFEST.json',manifest)
        raise


if __name__ == '__main__':
    main()
