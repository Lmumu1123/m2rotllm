import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.se(self.avg_pool(x))
        max_out = self.se(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ConvExt(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=8):
        super(ConvExt, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvMultiScale, self).__init__()
        if out_channels % 4 != 0:
            raise ValueError('out_channels should be divisible by 4')
        out_channels = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, 4, padding=0)
        self.conv3 = nn.Conv1d(in_channels, out_channels, 3, 4, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, 5, 4, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, 7, 4, padding=3)
        self.norm = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(out_channels * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.norm(x)
        x = self.relu(x)
        x = self.ca(x) * x
        x = torch.cat([x1, x], dim=1)
        return x


class SpecEncoder(nn.Module):
    def __init__(self):
        super(SpecEncoder, self).__init__()
        self.conv_wide = ConvExt(1, 128, 8, 8)
        conv_layers = [ConvMultiScale(128, 128) for _ in range(3)]
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_wide(x)
        x = self.conv_layers(x)
        return x


class SpecDecoder(nn.Module):
    def __init__(self):
        super(SpecDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128 * 47, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.mlp(x)


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.encoder = SpecEncoder()
        self.decoder = SpecDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    from thop import profile
    test_signal = torch.randn(1, 1, 24000).to('cuda')
    model = FCN()
    model = model.to('cuda')
    res = model(test_signal)
    print(res.shape)

    model = FCN()
    input_tensor = torch.randn(1, 1, 24000)
    flops, params = profile(model, inputs=(input_tensor,))
    model_name = str(FCN.__name__)
    print(model_name, flops, params)

