#!/usr/bin/env python3
"""Reproducible, resumable evaluation of the released MBHM/BearLLM checkpoints.

The released checkpoint's training membership is unknown. Results are descriptive
all-data checkpoint evaluations, not independent held-out/generalization claims.
MBHM HDF5 signals are already DCN transformed: never apply DCN a second time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS = [
    "Fault-Free", "Minor Inner Ring Fault", "Moderate Inner Ring Fault",
    "Severe Inner Ring Fault", "Minor Ball Fault", "Moderate Ball Fault",
    "Severe Ball Fault", "Minor Outer Ring Fault", "Moderate Outer Ring Fault",
    "Severe Outer Ring Fault",
]
TASKS = {0: "fault_detection", 1: "fault_classification", 2: "maintenance", 3: "risk_analysis"}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(dataset):
    with sqlite3.connect(f"file:{dataset / 'metadata.sqlite'}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        # An INTEGER PRIMARY KEY's natural scan is file_id order, as in upstream.
        return [dict(row) for row in conn.execute(
            'SELECT f.file_id,f.condition_id,f.label,c.dataset AS source '
            'FROM file_info f JOIN "condition" c ON f.condition_id=c.condition_id '
            'ORDER BY f.file_id')]


def make_protocol(rows, seed, references):
    """Match upstream 7/2/1 enumeration and three replacement reference draws."""
    healthy = defaultdict(list)
    for row in rows:
        if row["label"] == 0:
            healthy[row["condition_id"]].append(row["file_id"])
    generator = random.Random(seed)
    pairs, exclusions = [], []
    query_index = 0
    for row in rows:
        candidates = healthy.get(row["condition_id"])
        if not candidates:
            exclusions.append({**row, "reason": "no_fault_free_reference_in_same_condition"})
            continue
        split = "train" if query_index % 10 < 7 else "val" if query_index % 10 < 9 else "test"
        for reference_index in range(references):
            pairs.append({**row, "ref_id": generator.choice(candidates), "split": split,
                          "reference_index": reference_index, "pair_index": len(pairs)})
        query_index += 1
    return pairs, exclusions


def make_fallback_protocol(rows, seed, references):
    """Auxiliary predictions only for upstream-excluded queries; never combine scores."""
    healthy_conditions = {row["condition_id"] for row in rows if row["label"] == 0}
    healthy_sources = defaultdict(list)
    for row in rows:
        if row["label"] == 0:
            healthy_sources[row["source"]].append(row["file_id"])
    missing = [row for row in rows if row["condition_id"] not in healthy_conditions]
    generator = random.Random(seed)
    pairs = []
    for row in missing:
        if not healthy_sources[row["source"]]:
            raise ValueError(f"No healthy reference even in source {row['source']}")
        for reference_index in range(references):
            pairs.append({**row, "ref_id": generator.choice(healthy_sources[row["source"]]),
                          "split": "excluded_from_upstream", "reference_index": reference_index,
                          "pair_index": len(pairs)})
    return pairs, [{**row, "reason": "no_fault_free_reference_in_same_condition"} for row in missing]


def classification_metrics(records, classes=10):
    """Last confusion column records unparseable predictions as errors."""
    matrix = [[0] * (classes + 1) for _ in range(classes)]
    for truth, prediction in records:
        matrix[int(truth)][int(prediction) if 0 <= int(prediction) < classes else classes] += 1
    count = sum(map(sum, matrix))
    class_stats = []
    for label in range(classes):
        true_positive = matrix[label][label]
        support = sum(matrix[label])
        predicted = sum(row[label] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_stats.append({"label": label, "name": LABELS[label] if classes == 10 else ["healthy", "faulty"][label],
                            "support": support, "precision": precision, "recall": recall, "f1": f1})
    present = [row["f1"] for row in class_stats if row["support"]]
    return {"count": count, "accuracy": sum(matrix[i][i] for i in range(classes)) / count if count else None,
            "macro_f1_fixed_classes": sum(row["f1"] for row in class_stats) / classes,
            "macro_f1_present_classes": sum(present) / len(present) if present else None,
            "invalid_predictions": sum(row[-1] for row in matrix), "per_class": class_stats,
            "confusion_matrix": matrix, "confusion_columns": list(range(classes)) + ["unparseable"]}


def normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def parse_class(text):
    """Earliest explicit canonical class mention; ambiguous prose is not adjudicated."""
    text = normalize_text(text)
    matches = [(match.start(), label) for label, name in enumerate(LABELS)
               for match in [re.search(r"\b" + re.escape(normalize_text(name)) + r"\b", text)] if match]
    return min(matches)[1] if matches else -1


def parse_binary(text):
    match = re.match(r"^[\s*\"'`]*(yes|no)\b", text, re.IGNORECASE)
    return int(match.group(1).lower() == "yes") if match else -1


def text_metrics(prediction, reference):
    predicted, target = normalize_text(prediction).split(), normalize_text(reference).split()
    overlap = sum((Counter(predicted) & Counter(target)).values())
    unigram_f1 = 2 * overlap / (len(predicted) + len(target)) if predicted or target else 1.0
    # Exact LCS length with bit-parallel dynamic programming, independent of BLEU.
    masks = defaultdict(int)
    for index, token in enumerate(target):
        masks[token] |= 1 << index
    state = 0
    for token in predicted:
        union = state | masks[token]
        state = union & ~(union - ((state << 1) | 1))
    lcs = state.bit_count()
    rouge_l_f1 = 2 * lcs / (len(predicted) + len(target)) if predicted or target else 1.0
    return {"normalized_exact_match": float(predicted == target), "unigram_f1": unigram_f1,
            "rouge_l_f1": rouge_l_f1}


def read_resume(path, key):
    """Discard only an interrupted final JSONL line; reject duplicates/corruption."""
    records = {}
    if not path.exists():
        return records
    with path.open("rb+") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not stream.read(1):
                    stream.truncate(position)
                    break
                raise ValueError(f"Corrupt non-final JSONL line: {path}")
            identifier = record[key]
            if identifier in records:
                raise ValueError(f"Duplicate {key}={identifier} in {path}")
            records[identifier] = record
            if not line.endswith(b"\n"):
                stream.write(b"\n")
    return records


def load_adapter(checkpoint, device, include_lora):
    import torch
    from src.fine_tuning import AlignmentAdapter
    adapter = AlignmentAdapter()
    adapter.load_state_dict(torch.load(checkpoint / "vibration_adapter.pth", map_location="cpu", weights_only=True))
    if include_lora:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        config = json.loads((checkpoint / "adapter_config.json").read_text())
        if config.get("use_rslora") or config.get("use_dora") or config.get("rank_pattern") or config.get("alpha_pattern"):
            raise ValueError("This adapter loader expects the released uniform standard LoRA configuration.")
        wrapped = get_peft_model(adapter, LoraConfig(r=config["r"], lora_alpha=config["lora_alpha"],
                                  lora_dropout=0.0, target_modules=["linear1", "linear2", "linear3"]))
        prefix = "base_model.model.model.embed_tokens.adapter."
        weights = {"base_model.model." + name[len(prefix):]: value
                   for name, value in load_file(checkpoint / "adapter_model.safetensors").items()
                   if name.startswith(prefix)}
        if len(weights) != 6:
            raise ValueError(f"Expected six released adapter LoRA tensors, found {len(weights)}")
        loaded = set_peft_model_state_dict(wrapped, weights)
        if loaded.unexpected_keys or any("lora_" in key for key in loaded.missing_keys):
            raise ValueError(f"Adapter LoRA load failed: {loaded}")
        adapter = wrapped.get_base_model()
    return adapter.to(device).eval()


def adapter_logits(adapter, signals):
    features = adapter.feature_encoder(signals)
    return classifier_logits(adapter, features)


def classifier_logits(adapter, features):
    import torch
    layer = adapter.alignment_layer
    logits = layer.linear2(layer.relu(layer.linear1(features.reshape(len(features), -1))))
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite adapter classifier logits; refusing to report accuracy")
    return logits


def signal_store(hdf5, storage):
    data = hdf5["vibration"]
    if data.shape != (135516, 24000) or str(data.dtype) != "float32":
        raise ValueError(f"Unexpected released MBHM signal shape/dtype: {data.shape}, {data.dtype}")
    if storage == "memory":
        started = time.monotonic()
        loaded = data[:]
        print(json.dumps({"signal_storage": "memory", "bytes": loaded.nbytes,
                          "load_seconds": time.monotonic() - started}), flush=True)
        return loaded
    return data


def read_signals(hdf5, pairs, device):
    import numpy as np
    import torch
    # h5py advanced indexing requires sorted unique IDs. Read once and restore order.
    ids = np.asarray([[row.get("file_id", row.get("vib_id")), row["ref_id"]] for row in pairs])
    unique, inverse = np.unique(ids, return_inverse=True)
    signals = np.asarray(hdf5[unique], dtype=np.float32)[inverse].reshape(len(pairs), 2, 24000)
    if not np.isfinite(signals).all():
        raise ValueError("Nonfinite HDF5 signal encountered")
    return torch.from_numpy(signals).to(device)


def run_vibration(args, dataset, checkpoint, output, rows):
    import h5py
    import torch
    fallback = args.reference_policy == "excluded-same-source-fallback"
    pairs, exclusions = (make_fallback_protocol if fallback else make_protocol)(rows, args.seed, args.references)
    write_json(output / "excluded_samples.json", exclusions)
    write_json(output / "protocol.json", {
        "seed": args.seed, "reference_draws_per_query": args.references,
        "reference_policy": ("AUXILIARY: random.Random(seed).choice from all healthy rows in the SAME SOURCE DATASET, with replacement; condition/channel/rpm/load are not matched"
                             if fallback else "random.Random(seed).choice, replacement, all healthy rows in same condition; includes same-row references"),
        "split_policy": ("only queries excluded by upstream due to absent same-condition healthy reference; no train/val/test claim"
                         if fallback else "eligible queries ordered by file_id; ordinal modulo 10: 0..6=train,7..8=val,9=test"),
        "interpretation": ("AUXILIARY fallback experiment, NOT the upstream protocol and NOT comparable with paper/main evaluation"
                           if fallback else "reconstructed released-code split; released checkpoint membership is unknown; not an independent held-out test"),
        "metadata_rows": len(rows), "selected_unique_queries": len(pairs) // args.references,
        "same_condition_eligible_unique_queries": len(rows) - len(exclusions),
        "excluded_from_upstream_unique_queries": len(exclusions), "total_pairs": len(pairs),
        "excluded_by_source": dict(Counter(row["source"] for row in exclusions)),
        "split_unique_queries": {name: sum(row["split"] == name for row in pairs) // args.references for name in sorted({row["split"] for row in pairs})},
    })
    if args.limit:
        pairs = pairs[:args.limit]
    path = output / "vibration_predictions.jsonl"
    finished = read_resume(path, "pair_index")
    expected = {row["pair_index"] for row in pairs}
    if not set(finished) <= expected:
        raise ValueError("Resumed predictions are outside the selected evaluation protocol")
    remaining = [row for row in pairs if row["pair_index"] not in finished]
    started = time.monotonic()
    if remaining:
        models = {name: load_adapter(checkpoint, args.device, include_lora=lora)
                  for name, lora in [("pre_lora_adapter", False), ("final_lora_adapter", True)]}
        with h5py.File(dataset / "data.hdf5", "r") as data, path.open("a") as stream, torch.inference_mode():
            vibration = signal_store(data, args.signal_storage)
            for start in range(0, len(remaining), args.vibration_batch_size):
                batch = remaining[start:start + args.vibration_batch_size]
                signals = read_signals(vibration, batch, args.device)
                # Both released models share the identical convolutional encoder;
                # only the three alignment Linear layers receive LoRA updates.
                features = models["pre_lora_adapter"].feature_encoder(signals)
                outputs = {name: classifier_logits(model, features).softmax(-1).cpu().tolist() for name, model in models.items()}
                for index, item in enumerate(batch):
                    record = dict(item)
                    for name, probabilities in outputs.items():
                        record[name] = {"prediction": max(range(10), key=probabilities[index].__getitem__),
                                        "probabilities": probabilities[index]}
                    stream.write(json.dumps(record) + "\n")
                    finished[record["pair_index"]] = record
                stream.flush()
                if start == 0 or (start // args.vibration_batch_size) % 20 == 0:
                    elapsed = time.monotonic() - started
                    print(json.dumps({"stage": "vibration", "complete_pairs": len(finished), "total_pairs": len(pairs),
                                      "elapsed_seconds_this_session": elapsed, "pairs_per_second": (start + len(batch)) / elapsed}), flush=True)
        del models
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    groups = defaultdict(list)
    for record in finished.values():
        for group in ["all", "split/" + record["split"], "source/" + record["source"],
                      "split_source/" + record["split"] + "/" + record["source"]]:
            groups[group].append(record)
    metrics = {name: {group: classification_metrics((row["label"], row[name]["prediction"]) for row in subset)
                     for group, subset in groups.items()}
               for name in ["pre_lora_adapter", "final_lora_adapter"]}
    write_json(output / "vibration_metrics.json", {
        "complete": len(finished) == len(pairs), "smoke_test_only": bool(args.limit),
        "count_pairs": len(finished), "count_unique_queries": len({row["file_id"] for row in finished.values()}),
        "reference_policy": args.reference_policy,
        "interpretation": "internal adapter classifier logits, not generated LLM label accuracy; paired observations repeat queries; fallback policy, if selected, is a separate auxiliary experiment",
        "model_definitions": {
            "pre_lora_adapter": "base adapter from the SAME exported checkpoint with LoRA deltas disabled; includes that checkpoint's saved BatchNorm buffers, and is NOT necessarily the original pre-finetuning checkpoint",
            "final_lora_adapter": "same exported base adapter and BatchNorm buffers with its saved LoRA deltas enabled"},
        "metrics": metrics})


def load_llm(checkpoint, qwen, device):
    import torch
    from peft import PeftModel
    import src.fine_tuning as fine_tuning
    # get_bearllm chooses visible cuda:0; --device must refer to that visible device.
    if device not in ("cuda", "cuda:0", "cpu"):
        raise ValueError("For corpus use CUDA_VISIBLE_DEVICES to select a physical GPU, then --device cuda:0")
    fine_tuning.qwen_weights = str(qwen)
    fine_tuning.adapter_weights = str(checkpoint / "vibration_adapter.pth")
    model = fine_tuning.get_bearllm(train_mode=False)
    model.to(device)
    model = PeftModel.from_pretrained(model, str(checkpoint)).eval()
    modified = model.get_input_embeddings()
    adapter = modified.adapter.eval()
    # The loaded PEFT adapter remains held explicitly. No cache file or global data.
    model.set_input_embeddings(modified.embedding)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, adapter


def prompt_ids(tokenizer, instruction):
    import torch
    from src.fine_tuning import description_len, mod_xt_for_qwen, signal_token_id
    if instruction.count("#state_place_holder#") != 1:
        raise ValueError("Expected exactly one signal placeholder")
    first, last = mod_xt_for_qwen(instruction)
    return torch.tensor(tokenizer.encode(first, add_special_tokens=False) + [signal_token_id] * description_len
                        + tokenizer.encode(last, add_special_tokens=False), dtype=torch.long)


def embed_batch(model, adapter, sequences, signals, pad_token_id, padding_side="left"):
    import torch
    from src.fine_tuning import description_len, signal_token_id
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long, device=signals.device)
    attention = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        offset = width - len(sequence) if padding_side == "left" else 0
        ids[index, offset:offset + len(sequence)] = sequence.to(signals.device)
        attention[index, offset:offset + len(sequence)] = 1
    mask = ids == signal_token_id
    if not torch.all(mask.sum(-1) == description_len):
        raise ValueError("Every batch row must contain five signal tokens")
    embeddings = model.get_input_embeddings()(ids.masked_fill(mask, 0))
    signal_embeddings = adapter(signals).to(embeddings.dtype)
    embeddings[mask] = signal_embeddings.reshape(-1, embeddings.shape[-1])
    return ids, attention, embeddings


def reference_nll(model, adapter, tokenizer, prompts, records, signals):
    import torch
    from torch.nn import functional as F
    target_tokens = [tokenizer.encode(record["response"], add_special_tokens=False) + [tokenizer.eos_token_id] for record in records]
    sequences = [torch.cat([prompt, torch.tensor(target, dtype=torch.long)]) for prompt, target in zip(prompts, target_tokens)]
    _, attention, embeddings = embed_batch(model, adapter, sequences, signals, tokenizer.pad_token_id, "right")
    output = model(inputs_embeds=embeddings, attention_mask=attention, use_cache=False)
    result = []
    for index, (prompt, target) in enumerate(zip(prompts, target_tokens)):
        logits = output.logits[index, len(prompt) - 1:len(prompt) + len(target) - 1].float()
        loss = F.cross_entropy(logits, torch.tensor(target, device=logits.device), reduction="sum")
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite reference response likelihood; refusing to report metrics")
        result.append({"reference_nll_sum": loss.item(), "reference_token_count": len(target)})
    del output
    return result


def corpus_summary(records, total, smoke):
    groups = defaultdict(list)
    for record in records:
        for group in ["all", "task/" + TASKS[record["task_id"]], "source/" + record["source"]]:
            groups[group].append(record)
    result = {}
    for group, subset in groups.items():
        token_count = sum(row["reference_token_count"] for row in subset)
        nll_sum = sum(row["reference_nll_sum"] for row in subset)
        binary = [row for row in subset if row["task_id"] == 0]
        multiclass = [row for row in subset if row["task_id"] != 0]
        result[group] = {"count": len(subset), "truncated_outputs": sum(row["truncated"] for row in subset),
                         "mean_generated_tokens": sum(row["generated_tokens"] for row in subset) / len(subset),
                         "reference_token_nll": nll_sum / token_count,
                         "reference_perplexity": math.exp(min(700, nll_sum / token_count)),
                         "text_overlap": {key: sum(row[key] for row in subset) / len(subset)
                                          for key in ["normalized_exact_match", "unigram_f1", "rouge_l_f1"]}}
        if binary:
            result[group]["generated_binary_detection"] = classification_metrics(
                [(int(row["label_id"] != 0), row["parsed_prediction"]) for row in binary], classes=2)
        if multiclass:
            result[group]["generated_classification"] = classification_metrics(
                [(row["label_id"], row["parsed_prediction"]) for row in multiclass])
        result[group]["adapter_classification"] = classification_metrics(
            [(row["label_id"], row["adapter_prediction"]) for row in subset])
    reference_issues = []
    for record in records:
        reference_label = parse_binary(record["response"]) if record["task_id"] == 0 else parse_class(record["response"])
        expected = int(record["label_id"] != 0) if record["task_id"] == 0 else record["label_id"]
        if reference_label != expected:
            reference_issues.append({"id": record["id"], "task_id": record["task_id"],
                                     "label_id": record["label_id"], "parsed_reference_label": reference_label})
    return {"complete": len(records) == total, "smoke_test_only": smoke, "count": len(records), "total": total,
            "interpretation": "training-corpus replay: all released corpus entries are used for training by upstream, and all 600 query IDs map to the reconstructed vibration training split; not held-out validation",
            "reference_label_audit": {"method": "apply the same fixed label parser to official reference responses; no examples are removed from reported scores",
                                      "mismatch_count": len(reference_issues), "mismatches": reference_issues,
                                      "caveat": "references can themselves omit a diagnosis; high text overlap does not establish correctness"},
            "metric_definitions": {
                "generation": "greedy natural generation; max_new_tokens as configured; official repetition penalty retained",
                "parsed_prediction": "task0 leading yes/no; other tasks earliest explicit canonical class mention, absent=-1(error)",
                "normalized_exact_match": "lowercase, punctuation-to-space, whitespace-normalized exact match",
                "unigram_f1": "multiset word overlap F1 after the same normalization",
                "rouge_l_f1": "exact word-level LCS F1 after the same normalization, without stemming",
                "reference_token_nll": "teacher-forced reference response including terminal EOS; prompts excluded; not generated-token accuracy",
                "limitations": "lexical overlap is not a maintenance/risk factuality or safety assessment; no human or LLM judge; no paper BERTScore claim"},
            "metrics": result}


def run_corpus(args, dataset, checkpoint, qwen, output, metadata):
    import h5py
    import torch
    from transformers import AutoTokenizer
    records = json.loads((dataset / "corpus.json").read_text())
    if args.limit:
        # Smoke spans tasks and labels instead of only the leading healthy rows.
        buckets = defaultdict(list)
        for row in records:
            buckets[(row["label_id"], row["task_id"])].append(row)
        records = [buckets[key][index] for index in range(max(map(len, buckets.values())))
                   for key in sorted(buckets) if index < len(buckets[key])][:args.limit]
    metadata_by_id = {row["file_id"]: row for row in metadata}
    path = output / "corpus_predictions.jsonl"
    finished = read_resume(path, "id")
    if not set(finished) <= {row["id"] for row in records}:
        raise ValueError("Resumed corpus predictions are outside the selected protocol")
    remaining = [row for row in records if row["id"] not in finished]
    if remaining:
        model, adapter = load_llm(checkpoint, qwen, args.device)
        tokenizer = AutoTokenizer.from_pretrained(qwen, local_files_only=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        started = time.monotonic()
        with h5py.File(dataset / "data.hdf5", "r") as data, path.open("a") as stream, torch.inference_mode():
            vibration = signal_store(data, args.signal_storage)
            for start in range(0, len(remaining), args.corpus_batch_size):
                batch = remaining[start:start + args.corpus_batch_size]
                signals = read_signals(vibration, batch, args.device)
                prompts = [prompt_ids(tokenizer, row["instruction"]) for row in batch]
                ids, attention, embeddings = embed_batch(model, adapter, prompts, signals, tokenizer.pad_token_id)
                generated = model.generate(input_ids=ids, inputs_embeds=embeddings, attention_mask=attention,
                                           max_new_tokens=args.max_new_tokens, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
                # Score references separately; no reference is supplied during generation.
                nll = reference_nll(model, adapter, tokenizer, prompts, batch, signals)
                predictions = adapter_logits(adapter, signals).argmax(-1).cpu().tolist()
                eos_ids = model.generation_config.eos_token_id
                eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
                for index, item in enumerate(batch):
                    tokens = generated[index, ids.shape[1]:].cpu().tolist()
                    stop = next((position for position, token in enumerate(tokens) if token in eos_ids), len(tokens))
                    ended = stop < len(tokens)
                    tokens = tokens[:stop]
                    prediction = tokenizer.decode(tokens, skip_special_tokens=True)
                    record = {**item, "source": metadata_by_id[item["vib_id"]]["source"],
                              "prediction": prediction,
                              "parsed_prediction": parse_binary(prediction) if item["task_id"] == 0 else parse_class(prediction),
                              "adapter_prediction": predictions[index], "generated_tokens": len(tokens),
                              "truncated": not ended and len(tokens) >= args.max_new_tokens,
                              **nll[index], **text_metrics(prediction, item["response"])}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    finished[record["id"]] = record
                stream.flush()
                print(json.dumps({"stage": "corpus", "complete": len(finished), "total": len(records),
                                  "elapsed_seconds_this_session": time.monotonic() - started}), flush=True)
                write_json(output / "corpus_metrics.json", corpus_summary(list(finished.values()), len(records), bool(args.limit)))
    write_json(output / "corpus_metrics.json", corpus_summary(list(finished.values()), len(records), bool(args.limit)))


def run_heldout_generation(args, dataset, checkpoint, qwen, output, rows):
    """Generate one unconstrained classification response per held-out query."""
    import h5py
    import torch
    from transformers import AutoTokenizer
    pairs, exclusions = make_protocol(rows, args.seed, args.references)
    records = [row for row in pairs if row["split"] == args.split and row["reference_index"] == 0]
    del pairs
    if args.limit:
        records = records[:args.limit]
    corpus = json.loads((dataset / "corpus.json").read_text())
    template = next(row for row in corpus if row["task_id"] == 1)
    write_json(output / "heldout_protocol.json", {
        "split": args.split, "seed": args.seed, "unique_queries": len(records), "smoke_test_only": bool(args.limit),
        "reference_policy": f"first of {args.references} seeded upstream same-condition healthy reference draws per query",
        "instruction_corpus_id": template["id"], "instruction": template["instruction"],
        "interpretation": "greedy classification generation on reconstructed split, one response per unique query; independent held-out status requires the checkpoint to have been trained only on this reconstructed training split",
        "original_checkpoint_membership": "unknown; do not assert released-checkpoint held-out accuracy",
    })
    path = output / "heldout_predictions.jsonl"
    finished = read_resume(path, "file_id")
    if not set(finished) <= {row["file_id"] for row in records}:
        raise ValueError("Resumed held-out predictions are outside selected split")
    remaining = [row for row in records if row["file_id"] not in finished]
    if remaining:
        model, adapter = load_llm(checkpoint, qwen, args.device)
        tokenizer = AutoTokenizer.from_pretrained(qwen, local_files_only=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        prompt = prompt_ids(tokenizer, template["instruction"])
        started = time.monotonic()
        with h5py.File(dataset / "data.hdf5", "r") as data, path.open("a") as stream, torch.inference_mode():
            vibration = signal_store(data, args.signal_storage)
            for start in range(0, len(remaining), args.corpus_batch_size):
                batch = remaining[start:start + args.corpus_batch_size]
                signals = read_signals(vibration, batch, args.device)
                ids, attention, embeddings = embed_batch(model, adapter, [prompt] * len(batch), signals, tokenizer.pad_token_id)
                generated = model.generate(input_ids=ids, inputs_embeds=embeddings, attention_mask=attention,
                                           max_new_tokens=args.max_new_tokens, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
                predictions = adapter_logits(adapter, signals).argmax(-1).cpu().tolist()
                eos = model.generation_config.eos_token_id
                eos = {eos} if isinstance(eos, int) else set(eos)
                for index, item in enumerate(batch):
                    tokens = generated[index, ids.shape[1]:].cpu().tolist()
                    stop = next((position for position, token in enumerate(tokens) if token in eos), len(tokens))
                    prediction = tokenizer.decode(tokens[:stop], skip_special_tokens=True)
                    record = {**item, "prediction": prediction, "parsed_prediction": parse_class(prediction),
                              "adapter_prediction": predictions[index], "generated_tokens": stop,
                              "truncated": stop == len(tokens) and len(tokens) >= args.max_new_tokens}
                    stream.write(json.dumps(record) + "\n")
                    finished[record["file_id"]] = record
                stream.flush()
                if start == 0 or (start // args.corpus_batch_size) % 10 == 0:
                    print(json.dumps({"stage": "heldout-generation", "split": args.split, "complete": len(finished),
                                      "total": len(records), "elapsed_seconds_this_session": time.monotonic() - started}), flush=True)
    groups = defaultdict(list)
    for record in finished.values():
        groups["all"].append(record)
        groups["source/" + record["source"]].append(record)
    write_json(output / "heldout_metrics.json", {
        "complete": len(finished) == len(records), "smoke_test_only": bool(args.limit), "count": len(finished),
        "truncated_outputs": sum(row["truncated"] for row in finished.values()),
        "mean_generated_tokens": sum(row["generated_tokens"] for row in finished.values()) / len(finished) if finished else None,
        "metrics": {"generated_classification": {name: classification_metrics((row["label"], row["parsed_prediction"]) for row in group)
                                                   for name, group in groups.items()},
                    "adapter_classification": {name: classification_metrics((row["label"], row["adapter_prediction"]) for row in group)
                                                 for name, group in groups.items()}}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["vibration", "corpus", "heldout-generation", "all"], default="all")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="For heldout-generation only")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mbhm_official")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, help="Alternative checkpoint directory containing vibration_adapter.pth and PEFT files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--references", type=int, default=3)
    parser.add_argument("--reference-policy", choices=["same-condition", "excluded-same-source-fallback"], default="same-condition",
                        help="Fallback scores ONLY the 12,724 upstream-excluded rows using same-dataset healthy references; separate output directory required")
    parser.add_argument("--vibration-batch-size", type=int, default=128)
    parser.add_argument("--signal-storage", choices=["memory", "hdf5"], default="memory",
                        help="Preload the complete 13 GB HDF5 array into RAM, or stream HDF5 on smaller machines")
    parser.add_argument("--corpus-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, help="Explicit smoke-test limit; never use for final full evaluation")
    args = parser.parse_args()
    if min(args.references, args.vibration_batch_size, args.corpus_batch_size, args.max_new_tokens) <= 0:
        parser.error("All counts must be positive")
    if args.reference_policy != "same-condition" and args.stage != "vibration":
        parser.error("Auxiliary fallback supports --stage vibration only")
    from dotenv import dotenv_values
    import torch
    from transformers import set_seed
    env = dotenv_values(ROOT / ".env")
    dataset, checkpoint, qwen = (Path(env[key]).resolve() for key in ["MBHM_DATASET", "BEARLLM_WEIGHTS", "QWEN_WEIGHTS"])
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {"schema_version": 1, "stage": args.stage, "seed": args.seed, "references": args.references,
                     "reference_policy": args.reference_policy,
                     "split": args.split,
                     "signal_storage": args.signal_storage,
                     "vibration_batch_size": args.vibration_batch_size, "corpus_batch_size": args.corpus_batch_size,
                     "max_new_tokens": args.max_new_tokens, "limit": args.limit, "dataset": str(dataset),
                     "checkpoint": str(checkpoint), "qwen": str(qwen),
                     "metadata_sha256": sha256(dataset / "metadata.sqlite"), "corpus_sha256": sha256(dataset / "corpus.json"),
                     "vibration_adapter_sha256": sha256(checkpoint / "vibration_adapter.pth"),
                     "lora_sha256": sha256(checkpoint / "adapter_model.safetensors")}
    config_path = args.output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Output directory has a different run configuration; choose a new directory")
    write_json(config_path, configuration)
    set_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = read_metadata(dataset)
    started = time.monotonic()
    if args.stage in ("vibration", "all"):
        run_vibration(args, dataset, checkpoint, args.output, rows)
    if args.stage in ("corpus", "all"):
        run_corpus(args, dataset, checkpoint, qwen, args.output, rows)
    if args.stage == "heldout-generation":
        run_heldout_generation(args, dataset, checkpoint, qwen, args.output, rows)
    write_json(args.output / "completion.json", {"status": "complete", "smoke_test_only": bool(args.limit),
               "elapsed_seconds_this_session": time.monotonic() - started,
               "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
               "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"})


if __name__ == "__main__":
    main()
