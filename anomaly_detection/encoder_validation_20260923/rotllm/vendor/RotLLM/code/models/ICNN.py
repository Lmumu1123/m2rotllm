import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomPool1d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(RandomPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        out_seq_len = (seq_len - self.kernel_size) // self.stride + 1

        x_unfold = F.unfold(x, (1, self.kernel_size), stride=self.stride)
        x_unfold = x_unfold.view(batch_size, self.kernel_size, channels, out_seq_len)

        rand_weights = torch.rand(x_unfold.size(), device=x.device)
        rand_weights = x_unfold * rand_weights

        max_index = torch.argmax(rand_weights, dim=1)
        out = x_unfold.gather(1, max_index.unsqueeze(1)).squeeze(1)

        return out


class ICNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=15):
        super(ICNN, self).__init__()
        self.__name__ = 'ICNN'
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=32, stride=16),
            nn.LeakyReLU(),
            RandomPool1d(kernel_size=3, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            RandomPool1d(kernel_size=3, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            RandomPool1d(kernel_size=3, stride=1),
            nn.AdaptiveMaxPool1d(8))
        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, out_channel)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


if __name__ == "__main__":
    from torchsummary import summary
    model = ICNN()
    summary(model, input_size=(1, 12000), device='cpu')
