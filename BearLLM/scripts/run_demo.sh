#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
exec "$repo_dir/scripts/run_m2vllm.sh" "$repo_dir/run_demo.py" "$@"
