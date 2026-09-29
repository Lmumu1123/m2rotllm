import torch
from dotenv import dotenv_values
from peft import PeftModel
from transformers import AutoTokenizer, set_seed
from src.fine_tuning import description_len, signal_token_id, get_bearllm, mod_xt_for_qwen
from src import fine_tuning
import numpy as np
from functions.dcn import dcn
import json
from pathlib import Path
from tempfile import TemporaryDirectory

env = dotenv_values()
mbhm_dataset = env['MBHM_DATASET']
qwen_weights = env['QWEN_WEIGHTS']
bearllm_weights = env['BEARLLM_WEIGHTS']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_demo_data(data_file):
    with open(data_file, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('Input JSON must be an object.')
    instruction = data.get('instruction', '')
    if not isinstance(instruction, str) or instruction.count('#state_place_holder#') != 1:
        raise ValueError('instruction must contain exactly one #state_place_holder#.')
    for key in ('vib_data', 'ref_data'):
        signal = np.asarray(data.get(key, []), dtype=np.float64)
        if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError(f'{key} must be a nonempty, finite 1D signal.')
        if not np.any(signal):
            raise ValueError(f'{key} must have nonzero signal energy.')
    return data


def create_cache(demo_data, cache_file):
    query_data = np.array(demo_data['vib_data'], dtype=np.float64)
    ref_data = np.array(demo_data['ref_data'], dtype=np.float64)
    query_data = dcn(query_data)
    ref_data = dcn(ref_data)
    rv = np.array([query_data, ref_data])
    if not np.isfinite(rv).all():
        raise ValueError('DCN produced nonfinite values; check signal energy and scale.')
    np.save(cache_file, rv)


def run_demo(data_file=None, max_new_tokens=2048, seed=42, output_file=None, checkpoint_dir=None):
    if max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive.')
    demo_data = load_demo_data(data_file or f'{mbhm_dataset}/demo_data.json')
    checkpoint = Path(checkpoint_dir or bearllm_weights).expanduser().resolve()
    for filename in ('vibration_adapter.pth', 'adapter_config.json'):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(checkpoint / filename)
    if not any((checkpoint / name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin')):
        raise FileNotFoundError(f'No PEFT adapter weights found in {checkpoint}')
    set_seed(seed)

    place_holder_ids = torch.ones(description_len, dtype=torch.long) * signal_token_id
    text_part1, text_part2 = mod_xt_for_qwen(demo_data['instruction'])

    tokenizer = AutoTokenizer.from_pretrained(qwen_weights)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    user_part1_ids = tokenizer(text_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_part2_ids = tokenizer(text_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_ids = torch.cat([user_part1_ids, place_holder_ids, user_part2_ids])
    user_ids = user_ids.to(device)
    attention_mask = torch.ones_like(user_ids)
    attention_mask = attention_mask.to(device)

    original_adapter_path = fine_tuning.adapter_weights
    try:
        fine_tuning.adapter_weights = str(checkpoint / 'vibration_adapter.pth')
        model = get_bearllm(train_mode=False)
    finally:
        fine_tuning.adapter_weights = original_adapter_path
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval()

    # Keep the upstream signal conversion, but give each invocation its own cache.
    with TemporaryDirectory(prefix='bearllm-') as cache_dir:
        cache_file = str(Path(cache_dir) / 'signal.npy')
        create_cache(demo_data, cache_file)
        model.get_input_embeddings().signal_converter.test_file = cache_file
        with torch.inference_mode():
            output = model.generate(user_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(output[0, user_ids.shape[0]:], skip_special_tokens=True)
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output_text + '\n', encoding='utf-8')
    print(output_text)
    return output_text


if __name__ == "__main__":
    run_demo()
