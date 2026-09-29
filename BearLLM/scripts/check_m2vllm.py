#!/usr/bin/env python3
"""Gate full runs on an independent Conda environment and real CUDA checks."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/full/environment_ready.json")
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    if prefix.name != "m2vllm":
        raise RuntimeError(f"Expected the m2vllm Conda environment, got {prefix}")
    packages = ["torch", "transformers", "peft", "accelerate", "numpy", "scipy", "python-dotenv",
                "h5pickle", "h5py", "huggingface-hub", "tokenizers", "safetensors", "pandas", "pyarrow",
                "scikit-learn", "matplotlib", "rouge-score", "nltk", "tqdm"]
    versions = {}
    for name in packages:
        distribution = importlib.metadata.distribution(name)
        location = Path(distribution.locate_file("")).resolve()
        if not location.is_relative_to(prefix):
            raise RuntimeError(f"External Python dependency: {name} at {location}")
        versions[name] = distribution.version
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if pip_check.returncode:
        raise RuntimeError(pip_check.stdout + pip_check.stderr)
    import torch
    from transformers import Trainer, TrainingArguments
    from peft import PeftConfig
    from models.FCN import FaultClassificationNetwork
    from dotenv import dotenv_values
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
    cuda_checks = []
    for device_id in range(torch.cuda.device_count()):
        device = f"cuda:{device_id}"
        x = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
        y = x @ x.T
        assert y.isfinite().all().item()
        cuda_checks.append({"index": device_id, "name": torch.cuda.get_device_name(device_id),
                            "capability": torch.cuda.get_device_capability(device_id), "bf16_matmul": "passed"})
    model = FaultClassificationNetwork().to("cuda:0").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    logits = model(torch.randn(2, 2, 24000, device="cuda:0"))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1], device="cuda:0"))
    loss.backward()
    assert torch.isfinite(loss).item()
    assert all(parameter.grad is None or parameter.grad.isfinite().all().item() for parameter in model.parameters())
    optimizer.step()
    env = dotenv_values(ROOT / ".env")
    config = PeftConfig.from_pretrained(env["BEARLLM_WEIGHTS"])
    assert config.r == 4 and config.lora_alpha == 32 and len(config.target_modules) == 10
    result = {"status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "executable": sys.executable, "prefix": str(prefix), "python": sys.version,
              "packages": versions, "cuda": torch.version.cuda, "cuda_architectures": torch.cuda.get_arch_list(),
              "gpus": cuda_checks, "pip_check": pip_check.stdout.strip(),
              "trainer_import": "passed", "official_lora_config": "passed",
              "synthetic_fcn_forward_backward_optimizer_step": "passed (not a dataset accuracy result)",
              "apex_available": importlib.util.find_spec("apex") is not None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
