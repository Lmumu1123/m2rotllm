#!/usr/bin/env python3
"""Adapt a staging COPY of historical Python sources, preserving result files.

Requires --apply. Emits before/after hashes. Never run on the original projects.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

PREFIXES = ('/home/huangyating', '/media/nas_users/huangyating')
ALIAS = '_c2r_resolve_path'


def is_old(value):
    return isinstance(value, str) and any(value == p or value.startswith(p + '/') for p in PREFIXES)


class Relocate(ast.NodeTransformer):
    def __init__(self):
        self.changes = 0

    def wrapped(self, value):
        self.changes += 1
        return ast.Call(func=ast.Name(id=ALIAS, ctx=ast.Load()), args=[value], keywords=[])

    def visit_JoinedStr(self, node):
        # An f-string constant cannot be replaced with an ordinary Call node.
        values = []
        for v in node.values:
            if isinstance(v, ast.Constant) and is_old(v.value):
                values.append(ast.FormattedValue(value=self.wrapped(v), conversion=-1))
            elif isinstance(v, ast.FormattedValue):
                values.append(self.generic_visit(v))
            else:
                values.append(v)
        node.values = values
        return node

    def visit_Constant(self, node):
        return ast.copy_location(self.wrapped(node), node) if is_old(node.value) else node

    def visit_Call(self, node):
        # Runtime paths read from archived JSON retain their historical values in
        # the file, then are explicitly resolved at the filesystem boundary.
        if isinstance(node.func, ast.Name) and node.func.id in ('Path', 'open') and node.args:
            arg = node.args[0]
            node.args[0] = self.wrapped(arg)
            # Don't recursively wrap the newly introduced resolver.
            for i in range(1, len(node.args)):
                node.args[i] = self.visit(node.args[i])
            node.keywords = [self.visit(k) for k in node.keywords]
            return node
        return self.generic_visit(node)

    def visit_Assert(self, node):
        if ast.unparse(node.test) == "'envs/m2vllm' in sys.executable":
            self.changes += 1
            return ast.copy_location(ast.parse('assert sys.version_info >= (3, 10)').body[0], node)
        return self.generic_visit(node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if root in (Path('/home/huangyating'), Path('/media/nas_users/huangyating')):
        raise SystemExit('Refusing to alter original workspace')
    if not (root / 'c2r_paths.py').exists():
        raise SystemExit('Place c2r_paths.py at staging root first')
    output = root / 'migration_source_adaptations.json'
    prior = json.loads(output.read_text())['files'] if output.exists() else []
    report = []
    for base in ('anomaly_detection', 'BearLLM', 'research_proposals'):
        for p in sorted((root / base).rglob('*.py')):
            if p.is_symlink() or any(x in p.parts for x in ('.git', '__pycache__', '.venv', 'node_modules')):
                continue
            text = p.read_text(encoding='utf-8')
            if 'from c2r_paths import resolve_path as ' + ALIAS in text:
                continue
            relative = p.relative_to(root).as_posix()
            # Leave portable sources and third-party numerical code untouched.
            # One verifier reads original absolute paths from result JSON.
            needs_dynamic_resolution = relative == 'anomaly_detection/r2_validation_20260928/contact/verify_contact_results.py'
            if not any(prefix in text for prefix in PREFIXES) and 'envs/m2vllm' not in text and not needs_dynamic_resolution:
                continue
            tree = ast.parse(text, filename=str(p))
            trans = Relocate()
            tree = trans.visit(tree)
            # Cached fixed references are exactly the original nine training
            # queries. This avoids requiring the complete MBHM HDF5 for inference.
            reference_fallback = False
            if p.relative_to(root).as_posix() == 'anomaly_detection/four_class_chain_20260922/contact/evaluate_contact.py':
                fallback = ast.parse('''
cached = Path(__file__).resolve().parent / 'external_references.npz'
protocol = cached.with_name('external_reference_protocol.json')
if cached.exists() and protocol.exists():
    with np.load(cached, allow_pickle=False) as saved:
        refs = saved['dcn'].astype(np.float32)
        ids = saved['file_id'].tolist()
    chosen = json.loads(protocol.read_text())['references']
    assert refs.shape == (9, 24000) and np.isfinite(refs).all()
    assert ids == [row['file_id'] for row in chosen]
    return refs, chosen
''').body
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == 'external_references':
                        node.body = fallback + node.body
                        reference_fallback = True
            if not trans.changes and not reference_fallback:
                continue
            imp = ast.ImportFrom(module='c2r_paths', names=[ast.alias(name='resolve_path', asname=ALIAS)], level=0)
            pos = 0
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
                pos = 1
            while pos < len(tree.body) and isinstance(tree.body[pos], ast.ImportFrom) and tree.body[pos].module == '__future__':
                pos += 1
            tree.body.insert(pos, imp)
            ast.fix_missing_locations(tree)
            new_text = ast.unparse(tree) + '\n'
            compile(new_text, str(p), 'exec')
            record = dict(path=p.relative_to(root).as_posix(), original_sha256=hashlib.sha256(text.encode()).hexdigest(),
                          relocated_sha256=hashlib.sha256(new_text.encode()).hexdigest(), changes=trans.changes,
                          cached_reference_fallback=reference_fallback)
            report.append(record)
            if args.apply:
                # AST rendering changes formatting/comments. Retain exact original
                # sources as an auditable snapshot, including scientific comments.
                original = root / 'migration/original_sources' / p.relative_to(root)
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, original)
                p.write_text(new_text, encoding='utf-8')
    if args.apply:
        merged = {item['path']: item for item in prior}
        merged.update({item['path']: item for item in report})
        output.write_text(json.dumps(dict(policy='source-only explicit relocation; no result JSON/CSV changes', files=[merged[k] for k in sorted(merged)]), indent=2) + '\n')
    print(json.dumps(dict(apply=args.apply, changed_files_this_invocation=len(report), report=str(output)), indent=2))


if __name__ == '__main__':
    main()
