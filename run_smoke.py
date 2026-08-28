"""Minimal smoke test for RotLLM environment setup.

Uses synthetic DCN features in data/data.hdf5 (file_id < 2000) and
provided encoder/proj weights. Does not reproduce paper metrics.
"""
from __future__ import annotations

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(".env")

import torch
from torch.utils.data import DataLoader

from src.fine_tune.convert_weights import VibrationProjection
from src.models.SFN import SpecFoldNet, VibrationEncoder
from src.pre_train.dataloader import (
    VibrationDataset,
    get_filtered_sample_list,
    split_file_list,
)
from src.pre_train.main import LtModel


def main() -> None:
    from dotenv import dotenv_values

    cfg = dotenv_values(".env")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = cfg["RotLLM_WEIGHTS"]

    enc = VibrationEncoder(3, feature_channels=180, conv_layers=4)
    enc.load_state_dict(
        torch.load(f"{weights}/encoder_weights.pth", map_location="cpu", weights_only=True)
    )
    proj = VibrationProjection()
    proj.load_state_dict(
        torch.load(f"{weights}/proj_weights.pth", map_location="cpu", weights_only=True)
    )
    x = torch.randn(2, 3, 8000)
    with torch.no_grad():
        emb = proj(enc(x))
    print(f"encoder+proj OK -> {tuple(emb.shape)}")

    samples = get_filtered_sample_list()
    samples = samples[samples[:, 0] < 2000]
    train, _, _ = split_file_list(samples)
    loader = DataLoader(VibrationDataset(train[:64]), batch_size=16, num_workers=0)
    bx, by = next(iter(loader))
    model = LtModel().to(device)
    loss = torch.nn.functional.cross_entropy(model.model(bx.to(device)), by.to(device))
    print(f"pretrain step OK -> loss={float(loss):.4f} device={device}")

    qwen_dir = cfg["QWEN_WEIGHTS"]
    qwen_cfg = os.path.join(qwen_dir, "config.json")
    if os.path.exists(qwen_cfg):
        from transformers import AutoConfig

        conf = AutoConfig.from_pretrained(qwen_dir)
        print(f"qwen config OK -> hidden_size={conf.hidden_size} model={conf.model_type}")
        if conf.hidden_size != 2048:
            print("WARNING: RotLLM projection expects hidden_size=2048 (Qwen2.5-3B)")
    else:
        print(f"qwen weights not ready at {qwen_dir} (download Qwen2.5-3B-Instruct)")

    print("SMOKE_OK")


if __name__ == "__main__":
    main()
