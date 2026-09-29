#!/usr/bin/env python3
"""Recompute all archived R2 checkpoint metrics into a NEW output directory."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['C2R_ROOT'] = str(ROOT)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = ROOT / 'anomaly_detection/r2_validation_20260928/independent_check/audit_alignment.py'
    spec = importlib.util.spec_from_file_location('c2r_full_checkpoint_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = output
    module.ROOT = path.parent.parent
    module.audit()

if __name__ == '__main__':
    main()
