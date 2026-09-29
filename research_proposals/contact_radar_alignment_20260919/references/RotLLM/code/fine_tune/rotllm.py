import numpy as np
import torch
from dotenv import dotenv_values
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import h5pickle
from src.fine_tune.convert_weights import VibrationEncoder, VibrationProjection


def get_obj_loc(token_tensor: torch.Tensor):
    device = token_tensor.device
    batch_size = token_tensor.shape[0]
    out_res_start = torch.zeros((batch_size,), dtype=torch.long).to(device)
    out_res_end = torch.zeros((batch_size,), dtype=torch.long).to(device)
    out_res = (token_tensor == 151646).nonzero(as_tuple=True)  # <|object_ref_start|>
    out_res_start[out_res[0]] = out_res[1]
    out_res = (token_tensor == 151647).nonzero(as_tuple=True)  # <|object_ref_end|>
    out_res_end[out_res[0]] = out_res[1]
    fit = (out_res_end - out_res_start == 10).nonzero(as_tuple=True)[0]
    out_res = torch.zeros((batch_size,), dtype=torch.long).to(device)
    out_res[fit] = out_res_start[fit]
    return out_res


def decode_file_id(token_tensor: torch.Tensor, obj_loc: torch.Tensor):
    device = token_tensor.device
    locs = obj_loc.nonzero(as_tuple=True)[0].tolist()
    res = np.zeros((len(locs),), dtype=int)
    id_mask = torch.tensor([0, 0, 0, 0, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 0], dtype=torch.long).to(device)
    replace_matrix = np.zeros((len(locs), 2), dtype=int)
    for i, idx in enumerate(locs):
        id_start = obj_loc[idx]
        id_token = token_tensor[idx][id_start: id_start + 11]
        id_token = (id_token - 15) * id_mask  # 15->0
        id_token = id_token.sum().item()
        res[i] = id_token
        replace_matrix[i] = [idx, id_start.cpu().item()]
    indices = np.argsort(res)
    file_id = res[indices]
    recover_indices = np.argsort(indices)
    return file_id, recover_indices, replace_matrix


class VibEmbed(nn.Module):
    def __init__(self, raw_embed):
        super(VibEmbed, self).__init__()
        self.embed = raw_embed  # input(batch, token_len) -> output(batch, token_len, embed_dim)
        vibration_data_path = dotenv_values()['DCN_DATASET_PATH']
        self.vib_data = h5pickle.File(vibration_data_path, 'r')['data']
        self.encoder = VibrationEncoder(3, feature_channels=180, conv_layers=4)
        self.proj = VibrationProjection()
        weights = torch.load(dotenv_values()["RotLLM_WEIGHTS"] + "/encoder_weights.pth", weights_only=True)
        self.encoder.load_state_dict(weights)
        weights = torch.load(dotenv_values()["RotLLM_WEIGHTS"] + "/proj_weights.pth", weights_only=True)
        self.proj.load_state_dict(weights)

    def get_vib_data(self, x, obj_loc):
        device = x.device
        file_id, indices, mat = decode_file_id(x, obj_loc)
        vib_data = self.vib_data[file_id]
        vib_data = torch.tensor(vib_data, dtype=torch.float32)
        vib_data = vib_data.reshape(vib_data.size(0), 3, 8000)
        vib_data = vib_data[indices].to(device)
        return vib_data, mat

    def forward(self, x):
        text_embed = self.embed(x)
        with torch.no_grad():
            obj_loc = get_obj_loc(x)
            if obj_loc.sum() == 0:
                return text_embed
            vib_data, mat = self.get_vib_data(x, obj_loc)
            vib_feas = self.encoder(vib_data)
        vib_embed = self.proj(vib_feas)
        for i, (idx, start) in enumerate(mat):
            text_embed[idx, start: start + 11] = vib_embed[i]
        return text_embed


def test_print_qwen_dim():
    raw_model = AutoModelForCausalLM.from_pretrained(
        dotenv_values()["QWEN_WEIGHTS"],
        torch_dtype="auto",
        device_map="auto"
    )
    raw_embed = raw_model.get_input_embeddings()
    test_token_tensor = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    test_embed = raw_embed(test_token_tensor)
    print(test_embed.shape)


def test_print_num_ids():
    tokenizer = AutoTokenizer.from_pretrained(dotenv_values()["QWEN_WEIGHTS"])
    for i in range(100):
        test_input = f'reference signal_id: {i:06d}'
        model_inputs = tokenizer([test_input], return_tensors="pt", padding=True)['input_ids']
        print(test_input, model_inputs[0].tolist())


def get_mod_qwen():
    qwen = AutoModelForCausalLM.from_pretrained(
        dotenv_values()["QWEN_WEIGHTS"],
        torch_dtype="auto",
        device_map="auto"
    )
    embed = qwen.get_input_embeddings()
    mod_embed = VibEmbed(embed)
    mod_embed.to(qwen.device)
    qwen.set_input_embeddings(mod_embed)
    return qwen


def test_mod_qwen():
    tokenizer = AutoTokenizer.from_pretrained(dotenv_values()["QWEN_WEIGHTS"],
                                              padding_side='left')
    system_prompt = ("You are a mechanical expert with extensive expertise in rotating parts"
                     "such as bearings and gears. Please answer my question based on the reference signal status.")
    test_inputs = ["What is the reference signal status? reference signal is <|object_ref_start|>000010000<|object_ref_end|>",
                   "How to deal with <|object_ref_start|>000012300<|object_ref_end|> vibration?",
                   "How does bearing damage affect equipment operation?",]
    test_input_text = []
    for test_input in test_inputs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        test_input_text.append(text)

    qwen = get_mod_qwen()

    test_tokens = tokenizer(test_input_text, return_tensors="pt", padding=True).to(qwen.device)
    generated_ids = qwen.generate(
        **test_tokens,
        max_new_tokens=512
    )

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for res in response:
        print(res)


if __name__ == "__main__":
    test_mod_qwen()
