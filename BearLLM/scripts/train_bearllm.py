#!/usr/bin/env python3
"""Reproduce released BearLLM training without modifying official checkpoints.

The default protocol intentionally retains the release's modulo split, three
unrestricted healthy references, batch-loss LR scheduler, and all-corpus tuning.
Those choices differ from the paper and do not imply a leakage-free benchmark.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from dotenv import dotenv_values
from torch.utils.data import DataLoader, Dataset


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def save_checkpoint(path, obj):
    temp = Path(str(path) + ".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def log_event(output, event):
    line = json.dumps(event, ensure_ascii=False)
    print(line, flush=True)
    with (output / "epochs.jsonl").open("a") as file:
        file.write(line + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_splits(data_dir, seed):
    """Same SQL iteration/modulo/reference draws as upstream; local RNG seeded."""
    conn = sqlite3.connect(f"file:{data_dir / 'metadata.sqlite'}?mode=ro", uri=True)
    rows = conn.execute("SELECT condition_id, file_id, label FROM file_info").fetchall()
    refs = collections.defaultdict(list)
    for condition, file_id in conn.execute(
            "SELECT condition_id, file_id FROM file_info WHERE label = 0"):
        refs[condition].append(file_id)
    datasets = dict(conn.execute("SELECT condition_id, dataset FROM condition"))
    conn.close()
    source = {}
    labels = {}
    splits = {name: [] for name in ("train", "val", "test")}
    rand = random.Random(seed)
    eligible = 0
    excluded = collections.Counter()
    for condition, query, label in rows:
        source[query] = datasets[condition]
        labels[query] = label
        if condition not in refs:
            excluded[datasets[condition]] += 1
            continue
        subset = "train" if eligible % 10 < 7 else "val" if eligible % 10 < 9 else "test"
        for _ in range(3):
            splits[subset].append([query, rand.choice(refs[condition]), label])
        eligible += 1
    summary = {"seed": seed, "total_queries": len(rows), "eligible_queries": eligible,
               "excluded_queries_without_reference": dict(excluded),
               "pair_counts": {k: len(v) for k, v in splits.items()},
               "query_counts": {k: len({r[0] for r in v}) for k, v in splits.items()},
               "reference_policy": "all healthy rows of the same condition, across splits",
               "split_policy": "released SQL order; eligible index modulo 10: train 0..6, val 7..8, test 9"}
    train_queries = {r[0] for r in splits["train"]}
    for split, pairs in splits.items():
        summary[f"{split}_references_outside_training_queries"] = sum(r[1] not in train_queries for r in pairs)
    return splits, source, labels, summary


class SignalStore:
    def __init__(self, path, preload):
        self.path = str(path)
        self.file = None
        self.data = None
        if preload:
            started = time.monotonic()
            with h5py.File(self.path, "r") as file:
                self.data = file["vibration"][:]
            print(json.dumps({"event": "ram_preload", "shape": self.data.shape,
                              "bytes": self.data.nbytes, "seconds": time.monotonic() - started}), flush=True)

    def __getitem__(self, index):
        if self.data is not None:
            return self.data[index]
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        return self.file["vibration"][index]


class PairDataset(Dataset):
    def __init__(self, store, pairs):
        self.store, self.pairs = store, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        query, ref, label = self.pairs[index]
        return np.stack((self.store[query], self.store[ref])).astype(np.float32), label, query


def metric(cm):
    cm = np.asarray(cm, dtype=np.int64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    total = int(cm.sum())
    tp = cm.diagonal()
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(10, float), where=(support + predicted) > 0)
    recall = np.divide(tp, support, out=np.zeros(10, float), where=support > 0)
    return {"n": total, "correct": int(tp.sum()), "accuracy": float(tp.sum() / total) if total else None,
            "macro_f1_all_10_classes": float(f1.mean()),
            "balanced_accuracy_present_classes": float(recall[support > 0].mean()) if total else None,
            "false_alarm_rate": float(cm[0, 1:].sum() / support[0]) if support[0] else None,
            "missed_alarm_rate": float(cm[1:, 0].sum() / support[1:].sum()) if support[1:].sum() else None,
            "confusion_matrix": cm.tolist()}


@torch.inference_mode()
def evaluate_fcn(model, loader, device, sources):
    model.eval()
    cms = collections.defaultdict(lambda: np.zeros((10, 10), dtype=np.int64))
    loss_sum = 0.0
    batch_loss_sum = 0.0
    count = 0
    for data, labels, queries in loader:
        logits = model(data.to(device, non_blocking=True))
        truth = labels.to(device, non_blocking=True)
        batch_loss = torch.nn.functional.cross_entropy(logits, truth).item()
        batch_loss_sum += batch_loss
        loss_sum += batch_loss * len(labels)
        pred = logits.argmax(-1).cpu().numpy()
        truth = labels.numpy()
        count += len(truth)
        np.add.at(cms["MBHM"], (truth, pred), 1)
        names = np.array([sources[int(q)] for q in queries])
        for name in np.unique(names):
            mask = names == name
            np.add.at(cms[str(name)], (truth[mask], pred[mask]), 1)
    return {"loss": batch_loss_sum / len(loader), "sample_weighted_loss": loss_sum / count,
            "loss_reduction": "mean of batch means, matching released eval_epoch",
            "counting_unit": "query-reference pair (three per eligible query)",
            "by_dataset": {k: metric(v) for k, v in sorted(cms.items())}}


def runtime_config(args):
    result = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result.update(torch_version=torch.__version__, python_version=sys.version,
                  cuda_version=torch.version.cuda,
                  trainer_sha256=sha256(__file__),
                  metadata_sha256=sha256(args.data_dir / "metadata.sqlite"),
                  corpus_sha256=sha256(args.data_dir / "corpus.json"))
    result["effective_batch_size"] = args.batch_size * (args.accumulation_steps if args.stage == "finetune" else 1)
    result["loader_protocol"] = {"train_shuffle": True, "validation_shuffle": True,
                                 "test_shuffle": True, "persistent_workers": False} if args.stage == "pretrain" else {
                                     "engine": "transformers.Trainer", "dataloader_num_workers": 0,
                                     "loss_normalization": "native Trainer model loss kwargs"}
    result["package_versions"] = {name: importlib.metadata.version(name)
                                  for name in ("transformers", "peft", "numpy", "h5py", "scipy")}
    manifest_path = args.data_dir / "mbhm_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        result["dataset_revision"] = manifest.get("revision")
        result["dataset_download_verified"] = manifest.get("verified")
        for record in manifest.get("files", []):
            if record["rfilename"] == "data.hdf5":
                result["dataset_expected_sha256"] = record.get("lfs", {}).get("sha256")
    if args.fcn_dir:
        result["initial_fcn_sha256"] = {name: sha256(args.fcn_dir / name)
                                       for name in ("feature_encoder.pth", "classifier.pth")}
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        result["gpu"] = torch.cuda.get_device_name(args.device)
    return result


def pretrain(args):
    from models.FCN import FaultClassificationNetwork

    output = args.output_dir
    splits, sources, _, summary = source_splits(args.data_dir, args.seed)
    write_json(output / "split_manifest.json", summary)
    split_path = output / "dataset.json"
    if split_path.exists() and json.loads(split_path.read_text()) != splits:
        raise ValueError("Existing split manifest differs; choose a fresh output directory.")
    write_json(split_path, splits)
    store = SignalStore(args.data_dir / "data.hdf5", args.preload)
    loaders = {}
    for name, pairs in splits.items():
        loaders[name] = DataLoader(PairDataset(store, pairs), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers,
                                  pin_memory=args.device.startswith("cuda"),
                                  persistent_workers=False)
    model = FaultClassificationNetwork().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=150, factor=0.5)
    start = 0
    best_acc, best_loss = -1.0, float("inf")
    selected_epoch = 0
    if args.resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best_acc, best_loss = checkpoint["epoch"], checkpoint["best_acc"], checkpoint["best_loss"]
        selected_epoch = checkpoint.get("selected_epoch", start)
        if checkpoint.get("selection_protocol") != "source-loss-or-accuracy":
            raise ValueError("Checkpoint uses a different training protocol; use a fresh output directory")
        restore_rng(checkpoint["rng"])
    overall_start = time.monotonic()
    for epoch in range(start, args.epochs):
        started = time.monotonic()
        model.train()
        loss_sum, seen = 0.0, 0
        for batch_id, (data, labels, _) in enumerate(loaders["train"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(data.to(args.device, non_blocking=True))
            loss = torch.nn.functional.cross_entropy(logits, labels.to(args.device, non_blocking=True))
            loss.backward()
            optimizer.step()
            if args.scheduler == "source-batch-loss":
                scheduler.step(loss.detach())
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
            if (batch_id + 1) % args.log_steps == 0:
                print(json.dumps({"stage": "pretrain", "epoch": epoch + 1, "batch": batch_id + 1,
                                  "batches": len(loaders["train"]), "loss": loss.item(),
                                  "lr": optimizer.param_groups[0]["lr"]}), flush=True)
        validation = evaluate_fcn(model, loaders["val"], args.device, sources)
        acc = validation["by_dataset"]["MBHM"]["accuracy"]
        if args.scheduler == "validation-loss":
            scheduler.step(validation["loss"])
        improved = acc > best_acc
        source_selected = improved or validation["loss"] < best_loss
        if source_selected:
            selected_epoch = epoch + 1
        best_acc = max(best_acc, acc)
        best_loss = min(best_loss, validation["loss"])
        checkpoint = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "best_acc": best_acc, "best_loss": best_loss,
                      "rng": rng_state(), "config": runtime_config(args),
                      "selected_epoch": selected_epoch, "selection_protocol": "source-loss-or-accuracy"}
        save_checkpoint(output / "last.pt", checkpoint)
        if source_selected:
            save_checkpoint(output / "selected.pt", checkpoint)
            model.save_weights(str(output / "fcn"))
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            model.save_weights(str(output / "fcn_best_accuracy"))
        log_event(output, {"stage": "pretrain", "epoch": epoch + 1, "train_loss": loss_sum / seen,
                           "train_pairs": seen, "val": validation,
                           "best_val_accuracy": best_acc, "lr": optimizer.param_groups[0]["lr"],
                           "source_checkpoint_selected": source_selected, "source_selected_epoch": selected_epoch,
                           "seconds": time.monotonic() - started})
        if optimizer.param_groups[0]["lr"] < 1e-7 and not args.disable_early_stop:
            print("Source early stopping: learning rate below 1e-7", flush=True)
            break
    last = evaluate_fcn(model, loaders["test"], args.device, sources)
    best_checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    best = evaluate_fcn(model, loaders["test"], args.device, sources)
    selected_checkpoint = torch.load(output / "selected.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected_checkpoint["model"])
    selected = evaluate_fcn(model, loaders["test"], args.device, sources)
    report = {"stage": "pretrain", "epochs_completed": epoch + 1 if start < args.epochs else start,
              "requested_epochs": args.epochs, "best_epoch": best_checkpoint["epoch"],
              "best_validation_accuracy": best_acc, "last_checkpoint_test": last,
              "best_checkpoint_test": best, "elapsed_this_invocation_seconds": time.monotonic() - overall_start,
              "source_selected_epoch": selected_checkpoint["epoch"], "source_selected_checkpoint_test": selected,
              "fine_tuning_initialization": "source selected checkpoint (validation loss OR accuracy improves)",
              "exported_fcn_sha256": {name: sha256(output / "fcn" / name)
                                       for name in ("feature_encoder.pth", "classifier.pth")},
              "protocol": summary, "config": runtime_config(args)}
    write_json(output / "results.json", report)
    print(json.dumps({"event": "pretrain_complete", "results": str(output / "results.json")}), flush=True)


class CorpusSignals:
    def __init__(self, corpus, store, device, decode):
        self.rows = {row["id"]: row for row in corpus}
        self.store, self.device, self.decode = store, device, decode
        self.seen_ids = []

    @torch.no_grad()
    def get_signal(self, ids, train_mode=True):
        pairs = []
        for sample_id in self.decode(ids).cpu().tolist():
            self.seen_ids.append(int(sample_id))
            row = self.rows[int(sample_id)]
            pairs.append(np.stack((self.store[row["vib_id"]], self.store[row["ref_id"]])))
        return torch.from_numpy(np.stack(pairs).astype(np.float32)).to(self.device)


def prepare_corpus(corpus, tokenizer, module):
    examples = []
    for row in corpus:
        part1, part2 = module.mod_xt_for_qwen(row["instruction"])
        a = tokenizer.encode(part1, add_special_tokens=False)
        b = tokenizer.encode(part2, add_special_tokens=False)
        signal = (module.encode_sample_id(row["id"]) + module.signal_token_id).tolist()
        prompt = a + signal + b
        target = tokenizer.encode(row["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
        examples.append({"input_ids": torch.tensor(prompt + target),
                         "attention_mask": torch.ones(len(prompt) + len(target), dtype=torch.long),
                         "labels": torch.tensor([-100] * len(prompt) + target)})
    return examples


def save_lora_artifacts(model, output):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    # PEFT wraps original linears in base_layer. The released inference adapter
    # expects unwrapped base weights; its LoRA deltas live in adapter_model.
    state = model.get_input_embeddings().adapter.state_dict()
    base_state = {key.replace(".base_layer.", "."): value.detach().cpu()
                  for key, value in state.items() if ".lora_" not in key}
    save_checkpoint(output / "vibration_adapter.pth", base_state)


def finetune(args):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint
    from src import fine_tuning as module

    output = args.output_dir
    if not args.fcn_dir:
        raise ValueError("--fcn-dir must point to a trained FCN directory containing classifier.pth and feature_encoder.pth")
    for name in ("classifier.pth", "feature_encoder.pth"):
        if not (args.fcn_dir / name).is_file():
            raise FileNotFoundError(args.fcn_dir / name)
    module.qwen_weights = str(args.qwen_dir)
    module.fcn_weights = str(args.fcn_dir)
    module.l3_weights = str(output / "l3.npy")
    module.align_weights = str(output / "align.pth")
    module.adapter_weights = str(output / "initial_vibration_adapter.pth")
    checkpoint_dir = get_last_checkpoint(str(output / "trainer")) if (output / "trainer").exists() else None
    if args.resume:
        if checkpoint_dir is None:
            raise ValueError("No native Trainer checkpoint is available to resume")
        module.adapter_weights = str(Path(checkpoint_dir) / "vibration_adapter.pth")
        if not Path(module.adapter_weights).is_file():
            raise FileNotFoundError("Resume requires checkpoint vibration weights including BatchNorm buffers")
    original_hyperparameters = module.HyperParameters

    class DeviceHyperParameters(original_hyperparameters):
        def __init__(self):
            super().__init__()
            self.device = torch.device(args.device)

    module.HyperParameters = DeviceHyperParameters
    # Use the inference constructor to avoid opening another HDF5 handle and
    # then explicitly provide the corpus resolver for every train/eval batch.
    model = module.get_bearllm(train_mode=False)
    model.to(args.device)
    corpus = json.loads((args.data_dir / "corpus.json").read_text())
    if len({row["id"] for row in corpus}) != len(corpus):
        raise ValueError("Corpus IDs must be unique")
    store = SignalStore(args.data_dir / "data.hdf5", args.preload)
    signals = CorpusSignals(corpus, store, args.device, module.decode_sample_id)
    model.get_input_embeddings().signal_converter = signals
    model = get_peft_model(model, LoraConfig(target_modules="all-linear", task_type=TaskType.CAUSAL_LM,
                                           r=4, lora_alpha=32, lora_dropout=0.1))
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    examples = prepare_corpus(corpus, tokenizer, module)
    updates_per_epoch = math.ceil(math.ceil(len(examples) / args.batch_size) / args.accumulation_steps)
    total_updates = args.epochs * updates_per_epoch
    resume_progress = None
    if args.resume and (Path(checkpoint_dir) / "epoch_progress.json").exists():
        resume_progress = json.loads((Path(checkpoint_dir) / "epoch_progress.json").read_text())
        checkpoint_state = json.loads((Path(checkpoint_dir) / "trainer_state.json").read_text())
        history_path = output / "epochs.jsonl"
        if history_path.exists():
            records = [json.loads(line) for line in history_path.read_text().splitlines()]
            records = [row for row in records if row["epoch"] <= math.floor(checkpoint_state["epoch"])]
            history_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))

    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.epoch = 0
            self.started = 0.0
            self.loss_sum = 0.0
            self.loss_steps = 0
            self.last_logged_step = 0

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch = int(state.epoch or 0) + 1
            self.started = time.monotonic()
            if resume_progress and resume_progress["epoch"] == self.epoch:
                signals.seen_ids = resume_progress["seen_ids"][:]
                self.loss_sum = resume_progress["loss_sum"]
                self.loss_steps = resume_progress["loss_steps"]
                self.last_logged_step = resume_progress["last_logged_step"]
            else:
                signals.seen_ids = []
                self.loss_sum = 0.0
                self.loss_steps = 0
                self.last_logged_step = state.global_step

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                steps = state.global_step - self.last_logged_step
                self.loss_sum += float(logs["loss"]) * steps
                self.loss_steps += steps
                self.last_logged_step = state.global_step

        def on_epoch_end(self, args, state, control, optimizer=None, **kwargs):
            if len(signals.seen_ids) != len(corpus) or len(set(signals.seen_ids)) != len(corpus):
                raise RuntimeError(f"Epoch {self.epoch} did not consume every corpus ID exactly once: "
                                   f"{len(signals.seen_ids)} forwards/{len(set(signals.seen_ids))} unique")
            log_event(output, {"stage": "finetune", "epoch": self.epoch, "global_step": state.global_step,
                               "train_loss": self.loss_sum / max(1, self.loss_steps),
                               "loss_source": "native Trainer step-weighted logged loss",
                               "corpus_rows": len(signals.seen_ids), "unique_corpus_ids": len(set(signals.seen_ids)),
                               "lr": trainer._get_learning_rate(), "seconds": time.monotonic() - self.started})
            # Additional epoch-boundary saves improve recoverability; they do
            # not alter the native optimizer, scheduler, or accumulated loss.
            control.should_save = True
            return control

        def on_save(self, args, state, control, model=None, **kwargs):
            directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            adapter = model.get_input_embeddings().adapter.state_dict()
            base = {key.replace(".base_layer.", "."): value.detach().cpu()
                    for key, value in adapter.items() if ".lora_" not in key}
            save_checkpoint(directory / "vibration_adapter.pth", base)
            write_json(directory / "epoch_progress.json", {
                "epoch": self.epoch, "seen_ids": signals.seen_ids,
                "loss_sum": self.loss_sum, "loss_steps": self.loss_steps,
                "last_logged_step": self.last_logged_step})

    train_args = TrainingArguments(
        output_dir=str(output / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        logging_steps=args.log_steps,
        num_train_epochs=args.epochs,
        save_steps=200,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        # Operational differences from the release: retain two resumable
        # checkpoints, write plain logs, and avoid external reporting services.
        save_total_limit=2,
        disable_tqdm=True,
        report_to=[],
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=examples,
                      eval_dataset=None, data_collator=module.collate_fn,
                      callbacks=[ProgressCallback()])
    args.finetune_engine = "transformers.Trainer"
    args.model_accepts_loss_kwargs = trainer.model_accepts_loss_kwargs
    args.loss_normalization = "native Trainer model loss kwargs"
    write_json(output / "config.json", runtime_config(args))
    write_json(output / "trainer_arguments.json", train_args.to_dict())
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    write_json(output / "corpus_protocol.json", {
        "rows": len(corpus), "unique_vibration_ids": len({row["vib_id"] for row in corpus}),
        "task_counts": dict(collections.Counter(row["task_id"] for row in corpus)),
        "label_counts": dict(collections.Counter(row["label_id"] for row in corpus)),
        "training_scope": "all released corpus rows; no held-out language split, as upstream",
        "feature_encoder_mode": "train-mode BatchNorm buffers update, preserving released model.train() behavior",
        "max_sequence_tokens": max(len(item["input_ids"]) for item in examples),
        "trainable_parameters": trainable_names,
        "total_optimizer_updates": total_updates,
        "finetune_engine": args.finetune_engine,
        "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
        "loss_normalization": args.loss_normalization,
        "fcn_dir": str(args.fcn_dir)})
    overall_start = time.monotonic()
    train_result = trainer.train(resume_from_checkpoint=checkpoint_dir if args.resume else None)
    trainer.save_state()
    save_lora_artifacts(trainer.model, output / "weights")
    write_json(output / "results.json", {"stage": "finetune", "requested_epochs": args.epochs,
               "epochs_completed": int(trainer.state.epoch),
               "global_step": trainer.state.global_step, "corpus_rows": len(corpus), "weights": str(output / "weights"),
               "trainer_metrics": train_result.metrics,
               "finetune_engine": args.finetune_engine,
               "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
               "loss_normalization": args.loss_normalization,
               "elapsed_this_invocation_seconds": time.monotonic() - overall_start,
               "evaluation_scope_note": "All-corpus generation is in-sample; no held-out language accuracy is claimed.",
               "exported_weights_sha256": {name: sha256(output / "weights" / name)
                                            for name in ("vibration_adapter.pth", "adapter_model.safetensors")},
               "config": runtime_config(args)})
    print(json.dumps({"event": "finetune_complete", "results": str(output / "results.json")}), flush=True)


def main():
    env = dotenv_values(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pretrain", "finetune"))
    parser.add_argument("--data-dir", type=Path, default=Path(env["MBHM_DATASET"]))
    parser.add_argument("--qwen-dir", type=Path, default=Path(env["QWEN_WEIGHTS"]))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fcn-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scheduler", choices=("source-batch-loss", "validation-loss"), default="source-batch-loss")
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    args.batch_size = args.batch_size or (1024 if args.stage == "pretrain" else 1)
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
        parser.error("epochs, batch-size, accumulation-steps must be positive")
    args.output_dir = args.output_dir.resolve()
    official = Path(env["BEARLLM_WEIGHTS"]).resolve()
    if args.output_dir == official or official in args.output_dir.parents:
        parser.error("Output must not overwrite official pretrained weights")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if ((args.output_dir / "last.pt").exists() or
        any((args.output_dir / "trainer").glob("checkpoint-*"))) and not args.resume:
        parser.error("Existing training checkpoint: use --resume or a new --output-dir")
    old_config_path = args.output_dir / "config.json"
    if args.resume:
        old = json.loads(old_config_path.read_text())
        for key in ("stage", "seed", "epochs", "batch_size", "accumulation_steps", "scheduler", "learning_rate"):
            if old[key] != getattr(args, key):
                parser.error(f"Resume config differs for {key}; exact resume requires the original setting")
        for key, file in (("metadata_sha256", "metadata.sqlite"), ("corpus_sha256", "corpus.json")):
            if old[key] != sha256(args.data_dir / file):
                parser.error(f"Resume dataset changed: {file}")
    torch.set_num_threads(args.torch_threads)
    seed_all(args.seed)
    write_json(old_config_path, runtime_config(args))
    pretrain(args) if args.stage == "pretrain" else finetune(args)


if __name__ == "__main__":
    main()
