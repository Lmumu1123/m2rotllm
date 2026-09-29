import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import dotenv_values
import torch

def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(dotenv_values()["QWEN_WEIGHTS"])

    system_prompt = ("You are a mechanical expert with extensive expertise in rotating parts"
                     "such as bearings and gears. Please answer my question based on the reference signal status.")
    user_question = "What is the reference signal status?"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text)
    model_inputs = tokenizer([text], return_tensors="pt")
    print(model_inputs)

    state_text = ['', 'Normal', 'Fault']
    component_text = ['Bearing', 'Gear']
    location_text = ['Inner Ring', 'Ball', 'Outer Ring', 'Surface', 'Root', 'Tooth']
    severity_text = ['Minor', 'General', 'Severe', 'Wear', 'Fracture', 'Missing']

    for test_text in [state_text, component_text, location_text, severity_text]:
        model_inputs = tokenizer(test_text, return_tensors="pt", padding=True)['input_ids']
        print(model_inputs)

        response = tokenizer.batch_decode(model_inputs, skip_special_tokens=True)
        print(response)

    for _ in range(10):
        test_num = random.randint(0, 237297)
        test_num_text = f'{test_num:06d}'
        model_inputs = tokenizer([test_num_text], return_tensors="pt", padding=True)['input_ids']
        response = tokenizer.batch_decode(model_inputs, skip_special_tokens=True)[0]
        print(test_num, model_inputs, response)  # 6 digits


def convert_text_to_token(text_list, tokenizer):
    return tokenizer(text_list, return_tensors="pt", padding=True)['input_ids'].numpy()


def construct_vocab_token_matrix():
    tokenizer = AutoTokenizer.from_pretrained(dotenv_values()["QWEN_WEIGHTS"])
    component_text = ['', 'Bearing ', 'Gear ']
    component_token = convert_text_to_token(component_text, tokenizer)
    location_text = ['', 'Inner Ring ', 'Ball ', 'Outer Ring ', 'Surface ', 'Root ', 'Tooth ']
    location_token = convert_text_to_token(location_text, tokenizer)
    severity_text = ['', 'Minor ', 'General ', 'Severe ', 'Wear ', 'Fracture ', 'Missing ']
    severity_token = convert_text_to_token(severity_text, tokenizer)
    state_text = ['Normal', 'Fault']
    state_token = convert_text_to_token(state_text, tokenizer)
    label_ids = [[0, 0, 0, 0],
                 [1, 1, 1, 1],
                 [1, 1, 2, 1],
                 [1, 1, 3, 1],
                 [1, 2, 1, 1],
                 [1, 2, 2, 1],
                 [1, 2, 3, 1],
                 [1, 3, 1, 1],
                 [1, 3, 2, 1],
                 [1, 3, 3, 1],
                 [2, 4, 2, 1],
                 [2, 5, 2, 1],
                 [2, 6, 4, 1],
                 [2, 6, 5, 1],
                 [2, 6, 6, 1]]
    token_matrix = np.zeros((15, 11), int)
    for i, label in enumerate(label_ids):
        token_matrix[i, :3] = component_token[label[0]]
        token_matrix[i, 3:6] = location_token[label[1]]
        token_matrix[i, 6:10] = severity_token[label[2]]
        token_matrix[i, 10:] = state_token[label[3]]
    print(token_matrix)
    token_tensor = torch.tensor(token_matrix)
    response = tokenizer.batch_decode(token_tensor, skip_special_tokens=True)
    print(response)
    np.save(f"{dotenv_values()['RotLLM_WEIGHTS']}/vocab_token_matrix.npy", token_matrix)


def construct_vocab_embed():
    token_matrix = np.load(f"{dotenv_values()['RotLLM_WEIGHTS']}/vocab_token_matrix.npy")
    token_tensor = torch.tensor(token_matrix)
    model = AutoModelForCausalLM.from_pretrained(
        dotenv_values()["QWEN_WEIGHTS"],
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    emd = model.get_input_embeddings()
    vocab = emd(token_tensor).detach().numpy()
    np.save(f"{dotenv_values()['RotLLM_WEIGHTS']}/vocab_embed.npy", vocab)


if __name__ == "__main__":
    test_tokenizer()
