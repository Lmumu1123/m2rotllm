from dotenv import dotenv_values
import torch
from src.models.SFN import SpecFoldNet


def main(v_num=1):
    ckpt_path = f'{dotenv_values()["LOG_PATH"]}/lightning_logs/version_{v_num}/checkpoints/best_loss.ckpt'
    ckpt_data = torch.load(ckpt_path, weights_only=True)
    weights = ckpt_data['state_dict']
    weights = {k.replace('model.', ''): v for k, v in weights.items()}
    print(weights.keys())
    torch.save(weights, f'{dotenv_values()["RotLLM_WEIGHTS"]}/sfn_weights.pth')
    model = SpecFoldNet()
    model.load_state_dict(weights)


main()