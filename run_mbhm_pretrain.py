"""Pre-train / evaluate SFN encoder on MBHM (bearing-only proxy for RotLLM LMR).

Usage (after data.hdf5 is downloaded):
  conda activate rotllm
  cd /home/huangyating/RotLLM
  CUDA_VISIBLE_DEVICES=0 python run_mbhm_pretrain.py --epochs 5
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from dotenv import dotenv_values, load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

# Point RotLLM data paths at MBHM (prefer /data1/datasets)
_CFG = dict(dotenv_values(os.path.join(ROOT, ".env")))
_MBHM = _CFG.get("MBHM_DIR") or (
    "/data1/datasets/RotLLM/mbhm_dataset"
    if os.path.isdir("/data1/datasets/RotLLM/mbhm_dataset")
    else os.path.join(ROOT, "mbhm_dataset")
)
_CFG["DATASET_METADATA_PATH"] = _CFG.get("MBHM_METADATA_PATH") or os.path.join(
    _MBHM, "metadata.sqlite"
)
_wrapper = os.path.join(_MBHM, "data_as_rotllm.hdf5")
_raw = _CFG.get("MBHM_DCN_DATASET_PATH") or os.path.join(_MBHM, "data.hdf5")
_CFG["DCN_DATASET_PATH"] = _wrapper if os.path.exists(_wrapper) else _raw
_CFG["LOG_PATH"] = os.path.join(
    _CFG.get("LOG_PATH", os.path.join(ROOT, "logs")), "mbhm"
)
os.makedirs(_CFG["LOG_PATH"], exist_ok=True)

import dotenv as _dotenv

_dotenv.dotenv_values = lambda *a, **k: dict(_CFG)

import h5py
import h5pickle
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.models.SFN import SpecFoldNet
from src.pre_train.dataloader import get_filtered_sample_list, split_file_list


class MBHMVibrationDataset(Dataset):
    """Like RotLLM VibrationDataset but accepts HDF5 key `data` or `vibration`."""

    def __init__(self, subset_info, snr=None):
        self.info = subset_info
        path = _CFG["DCN_DATASET_PATH"]
        if snr is not None:
            path = path.replace("data.hdf5", f"data_noise_{snr}.hdf5")
        f = h5pickle.File(path, "r")
        if "data" in f:
            self.data = f["data"]
        elif "vibration" in f:
            self.data = f["vibration"]
        else:
            raise KeyError(f"No data/vibration in {path}: {list(f.keys())}")

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        file_id, label = self.info[idx]
        data = torch.from_numpy(self.data[int(file_id)]).float().unsqueeze(0)
        return data, torch.tensor(int(label), dtype=torch.long)


class LtSFN(L.LightningModule):
    def __init__(self, num_classes: int = 10, fold_num: int = 3):
        super().__init__()
        self.model = SpecFoldNet(fold_num=fold_num)
        # MBHM has 10 bearing classes; replace final classifier
        in_dim = self.model.fc[0].in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        acc = (self.model(x).argmax(1) == y).float().mean()
        self.log("test_acc", acc, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=5, factor=0.5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }


def load_pretrained_encoder(model: LtSFN, weights_path: str) -> None:
    ew = torch.load(weights_path, map_location="cpu", weights_only=True)
    miss, unexp = model.model.encoder.load_state_dict(ew, strict=False)
    print(f"loaded encoder from {weights_path}; missing={miss} unexpected={unexp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument(
        "--init-encoder",
        default=os.path.join(ROOT, "weights", "encoder_weights.pth"),
        help="Load RotLLM encoder weights before training/eval",
    )
    ap.add_argument("--no-init-encoder", action="store_true")
    args = ap.parse_args()

    data_path = _CFG["DCN_DATASET_PATH"]
    if not os.path.exists(data_path):
        raise SystemExit(
            f"Missing {data_path}. Wait for MBHM data.hdf5 download, then:\n"
            f"  python adapt_mbhm.py"
        )

    # Ensure wrapper exists when raw MBHM file is present
    if data_path.endswith("data.hdf5") and "mbhm" in data_path:
        with h5py.File(data_path, "r") as f:
            print("hdf5 keys:", list(f.keys()), "shape", f[list(f.keys())[0]].shape)

    samples = get_filtered_sample_list()
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    train, val, test = split_file_list(samples)
    print(f"MBHM split train/val/test = {len(train)}/{len(val)}/{len(test)}")

    train_loader = DataLoader(
        MBHMVibrationDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        MBHMVibrationDataset(val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        MBHMVibrationDataset(test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = LtSFN(num_classes=10)
    if not args.no_init_encoder and os.path.exists(args.init_encoder):
        load_pretrained_encoder(model, args.init_encoder)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=_CFG["LOG_PATH"],
        callbacks=[
            ModelCheckpoint(monitor="val_acc", mode="max", filename="mbhm_best_acc"),
            ModelCheckpoint(monitor="val_loss", mode="min", filename="mbhm_best_loss"),
            EarlyStopping(monitor="val_loss", patience=8, mode="min"),
        ],
        log_every_n_steps=20,
    )

    if not args.eval_only:
        trainer.fit(model, train_loader, val_loader)
    res = trainer.test(model, test_loader)
    print("TEST", res)


if __name__ == "__main__":
    main()
