import torch
import torch.nn as nn


class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()

        self.downsample = nn.MaxPool1d(1, 8)

        self.ec1 = nn.Sequential(nn.Conv1d(1, 50, 20, 2),
                                 nn.Tanh(),
                                 nn.Conv1d(50, 30, 10, 2),
                                 nn.Tanh(),
                                 nn.MaxPool1d(2, 2, padding=1))

        self.ec2 = nn.Sequential(nn.Conv1d(1, 50, 6, 1),
                                 nn.Tanh(),
                                 nn.Conv1d(50, 40, 6, 1),
                                 nn.Tanh(),
                                 nn.MaxPool1d(2, 2),
                                 nn.Conv1d(40, 30, 6, 1),
                                 nn.Tanh(),
                                 nn.Conv1d(30, 30, 6, 2),
                                 nn.Tanh(),
                                 nn.MaxPool1d(2, 2))

        self.fc = nn.Sequential(nn.Linear(30 * 184, 128),
                                nn.ReLU(),
                                nn.Linear(128, 15))

    def forward(self, x):
        x = self.downsample(x)
        ec1_out = self.ec1(x)
        ec2_out = self.ec2(x)
        ec_out = torch.mul(ec1_out, ec2_out)
        x = ec_out.view(-1, 30 * 184)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    model = MCNN()
    summary(model, input_size=(1, 12000), device='cpu')
