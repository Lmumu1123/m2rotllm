"""Smoke-test RotLLM multimodal embedding injection (not paper metrics).

Requires:
  - .env configured
  - qwen_weights/ = Qwen2.5-3B-Instruct
  - data/data.hdf5 with indices covering used file_ids
  - weights/encoder_weights.pth + proj_weights.pth
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from dotenv import dotenv_values, load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))
_CFG = dotenv_values(os.path.join(ROOT, ".env"))

# Scripts call dotenv_values() with no path; pin to project .env.
import dotenv as _dotenv

_orig = _dotenv.dotenv_values


def _dotenv_values(*args, **kwargs):
    if not args and "dotenv_path" not in kwargs and "dirname" not in kwargs:
        return dict(_CFG)
    return _orig(*args, **kwargs)


_dotenv.dotenv_values = _dotenv_values

import torch
from transformers import AutoTokenizer

from code.fine_tune.rotllm import get_mod_qwen


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    tokenizer = AutoTokenizer.from_pretrained(
        _CFG["QWEN_WEIGHTS"], padding_side="left", trust_remote_code=True
    )
    system_prompt = (
        "You are a mechanical expert with extensive expertise in rotating parts"
        "such as bearings and gears. Please answer my question based on the reference signal status."
    )
    # 9-digit ids; decode uses positions [4:10] of the 11-token object span -> file_id
    # 000000100 -> 100, 000000500 -> 500 (must exist in data/data.hdf5)
    test_inputs = [
        "What is the reference signal status? reference signal is <|object_ref_start|>000000100<|object_ref_end|>",
        "How does bearing damage affect equipment operation?",
    ]
    texts = []
    for user in test_inputs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    qwen = get_mod_qwen()
    tokens = tokenizer(texts, return_tensors="pt", padding=True).to(qwen.device)
    with torch.no_grad():
        out = qwen.generate(**tokens, max_new_tokens=128)
    responses = tokenizer.batch_decode(out, skip_special_tokens=True)
    for i, res in enumerate(responses):
        print("=" * 60)
        print(f"CASE {i}")
        print(res[-800:])
    print("INFERENCE_SMOKE_OK")


if __name__ == "__main__":
    main()
