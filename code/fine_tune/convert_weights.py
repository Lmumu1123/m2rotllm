import numpy as np
from dotenv import dotenv_values
import torch
from code.models.SFN import VibrationEncoder
import torch.nn as nn


class VibrationProjection(nn.Module):
    def __init__(self, feature_dim=720, embed_dim=2048):
        super(VibrationProjection, self).__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Sequential(nn.Linear(feature_dim, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 15),
                                  nn.Softmax(dim=1),
                                  nn.Linear(15, 11 * embed_dim))
        self.adapt = nn.Linear(feature_dim, 11 * embed_dim)

    def forward(self, x):
        x = x.view(x.size(0), self.feature_dim)
        x0 = self.adapt(x)
        x = self.proj(x)
        x = x + x0
        x = x.view(x.size(0), 11, 2048)
        return x


def main(v_num=1):
    sfn_weights_path = f'{dotenv_values()["RotLLM_WEIGHTS"]}/sfn_weights.pth'
    sfn_weights = torch.load(sfn_weights_path, weights_only=True)
    print(sfn_weights.keys())

    encoder_weights = {k.replace('encoder.', ''): v for k, v in sfn_weights.items() if 'encoder' in k}
    torch.save(encoder_weights, f'{dotenv_values()["RotLLM_WEIGHTS"]}/encoder_weights.pth')

    proj_weights = {k.replace('fc.', 'proj.'): v for k, v in sfn_weights.items() if 'fc.' in k}

    raw_fc_weight = sfn_weights['fc.0.weight']
    raw_fc_bias = sfn_weights['fc.0.bias']
    print(raw_fc_weight.shape, raw_fc_bias.shape)
    embed_weight = np.load(f'{dotenv_values()["RotLLM_WEIGHTS"]}/vocab_embed.npy')
    embed_weight = torch.tensor(embed_weight, dtype=torch.float32)
    embed_weight = embed_weight.view(15, 11 * 2048).T
    print(embed_weight.shape)
    embed_bias = torch.zeros(11 * 2048, dtype=torch.float32)
    adapt_weight = torch.zeros(11 * 2048, 720, dtype=torch.float32)
    adapt_bias = torch.zeros(11 * 2048, dtype=torch.float32)
    proj_weights['proj.4.weight'] = embed_weight
    proj_weights['proj.4.bias'] = embed_bias
    proj_weights['adapt.weight'] = adapt_weight
    proj_weights['adapt.bias'] = adapt_bias
    torch.save(proj_weights, f'{dotenv_values()["RotLLM_WEIGHTS"]}/proj_weights.pth')
    encoder = VibrationEncoder(3, feature_channels=180, conv_layers=4)
    encoder.load_state_dict(encoder_weights)
    proj = VibrationProjection()
    proj.load_state_dict(proj_weights)
    x = torch.randn(64, 1, 24000)
    x = x.view(x.size(0), 3, 8000)
    x = encoder(x)
    x = proj(x)
    print(x.size())


if __name__ == "__main__":
    main()
