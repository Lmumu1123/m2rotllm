#!/usr/bin/env python3
"""Check batched memory injection against released ModifiedEmbedding and PEFT."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from dotenv import dotenv_values
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed
    from functions.dcn import dcn
    from src.fine_tuning import get_bearllm
    from evaluate_mbhm import adapter_logits, embed_batch, load_adapter, prompt_ids, write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = dotenv_values(ROOT / ".env")
    set_seed(42)
    torch.set_num_threads(4)
    checkpoint = Path(env["BEARLLM_WEIGHTS"])
    tokenizer = AutoTokenizer.from_pretrained(env["QWEN_WEIGHTS"], local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    raw = json.loads((Path(env["MBHM_DATASET"]) / "demo_data.json").read_text())
    corpus = json.loads((Path(env["MBHM_DATASET"]) / "corpus.json").read_text())
    query, reference = [dcn(np.asarray(raw[key], dtype=np.float64)) for key in ["vib_data", "ref_data"]]
    # Two different signal pairs and prompt lengths expose row-order/padding errors.
    signals = torch.tensor(np.stack([[query, reference], [reference, reference]])).cuda()
    prompts = [prompt_ids(tokenizer, raw["instruction"]),
               prompt_ids(tokenizer, next(row["instruction"] for row in corpus if row["task_id"] == 1))]
    model = PeftModel.from_pretrained(get_bearllm(False), str(checkpoint)).eval()
    modified = model.get_input_embeddings()
    modified.signal_converter.get_signal = lambda *arguments: signals
    adapter = load_adapter(checkpoint, "cuda:0", True)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    with torch.inference_mode():
        logit_error = (adapter_logits(adapter, signals) - adapter_logits(modified.adapter, signals)).abs().max().item()
        model.set_input_embeddings(modified.embedding)
        ids, attention, embeddings = embed_batch(model, modified.adapter, prompts, signals, tokenizer.pad_token_id)
        native_embeddings = modified(ids)
        embedding_error = (native_embeddings - embeddings).abs().max().item()
        model.set_input_embeddings(modified)
        native = model.generate(input_ids=ids, attention_mask=attention, max_new_tokens=64,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id)
        model.set_input_embeddings(modified.embedding)
        injected = model.generate(input_ids=ids, attention_mask=attention, inputs_embeds=embeddings,
                                  max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    result = {"adapter_logits_max_error": logit_error, "prompt_embeddings_max_error": embedding_error,
              "batch_size": 2, "prompt_lengths": [len(prompt) for prompt in prompts],
              "native_vs_injected_generation_exact_match": torch.equal(native, injected),
              "responses": [tokenizer.decode(row[ids.shape[1]:], skip_special_tokens=True) for row in injected],
              "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    assert logit_error == 0 and embedding_error == 0 and torch.equal(native, injected), result


if __name__ == "__main__":
    main()
