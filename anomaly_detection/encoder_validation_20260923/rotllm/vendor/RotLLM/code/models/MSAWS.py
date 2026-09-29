import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channel // reduction, in_channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        out = avg_out + max_out
        out = out * x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out


def make_layer(in_channel, out_channel, block_num, stride=1):
    layers = [ResidualBlock(in_channel, out_channel, stride)]
    for i in range(block_num - 1):
        layers.append(ResidualBlock(out_channel, out_channel))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNet, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.ca = ChannelAttention(64)
        self.layer1 = make_layer(64, 64, 2, stride=1)
        self.layer2 = make_layer(64, 128, 2, stride=2)
        self.layer3 = make_layer(128, 256, 2, stride=2)
        self.layer4 = make_layer(256, 256, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_channel)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.ca(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


class MSAWS(nn.Module):
    def __init__(self, in_channel=1, out_channel=15):
        super(MSAWS, self).__init__()
        self.__name__ = 'MSAWS'
        self.in_channel = in_channel
        self.resnet = ResNet(1, out_channel)

    def stft_transform(self, x):
        batch_size = x.shape[0]
        output = torch.empty(batch_size, self.in_channel, 251, 97)
        for i in range(self.in_channel):
            tmp = torch.stft(x[:, i, :], n_fft=500, normalized=True, return_complex=True)
            output[:, i, :, :] = torch.abs(tmp)
        output = output.to(x.device)
        return output

    def forward(self, x):
        x = self.stft_transform(x)
        x = self.resnet(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    model = MSAWS()
    summary(model, input_size=(1, 12000), device='cpu')