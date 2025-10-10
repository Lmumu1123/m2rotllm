import torch
import torch.nn as nn


def regular_conv1(channel):
    return nn.Conv1d(channel, channel, kernel_size=1, padding=0)


def regular_conv3(channel):
    return nn.Conv1d(channel, channel, kernel_size=3, padding=1)


class MRM(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, 4 * in_channel, kernel_size=1, padding=0)
        self.p1 = nn.Conv1d(in_channel, in_channel, kernel_size=1, padding=0)
        self.p3 = nn.Conv1d(in_channel, in_channel, kernel_size=3, padding=1)
        self.p5 = nn.Conv1d(in_channel, in_channel, kernel_size=5, padding=2)
        self.p7 = nn.Conv1d(in_channel, in_channel, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channel * 4, in_channel, kernel_size=1, padding=0)

    def forward(self, x0):
        x = self.conv1(x0)
        x1, x3, x5, x7 = torch.split(x, x.size(1) // 4, dim=1)
        x1 = self.p1(x1)
        x3 = self.p3(x3)
        x5 = self.p5(x5)
        x7 = self.p7(x7)
        x = torch.cat((x1, x3, x5, x7), dim=1)
        x = self.conv2(x)
        x = torch.add(x, x0)
        return x


def soft_threshold(x, tao):
    return torch.sign(x) * torch.max(torch.abs(x) - tao, torch.zeros_like(x))


class SDM(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, 1),
            nn.Linear(1, in_channel),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0):
        qv = self.gap(torch.abs(x0))
        qv = qv.view(qv.size(0), -1)
        beta = self.fc(qv)
        beta = self.sigmoid(beta)
        tao = qv * beta
        tao = tao.view(tao.size(0), tao.size(1), 1)
        x = soft_threshold(x0, tao)
        x = torch.add(x, x0)
        return x


class MSDM(nn.Module):
    # 多尺度收缩去噪模块
    def __init__(self, in_channel, out_channel):
        super(MSDM, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.mrm = MRM(out_channel)
        self.sdm = SDM(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.mrm(x)
        x = self.sdm(x)
        return x


class CFM(nn.Module):
    # 中心融合子网络
    def __init__(self, in_channel, out_channel, last_channel=0):
        super(CFM, self).__init__()
        self.conv1_a = regular_conv1(in_channel)
        self.conv1_b = regular_conv1(in_channel)
        self.conv2_a = nn.Sequential(
            regular_conv3(in_channel),
            nn.Sigmoid()
        )
        self.conv2_b = nn.Sequential(
            regular_conv3(in_channel),
            nn.Sigmoid()
        )
        self.conv3_a = regular_conv3(in_channel)
        self.conv3_b = regular_conv3(in_channel)
        self.conv = nn.Conv1d(in_channel * 2, out_channel, kernel_size=3, padding=1)
        self.downsample = nn.MaxPool1d(3, 2, padding=1)
        if last_channel > 0:
            self.conv_last = nn.Conv1d(out_channel + last_channel, out_channel, kernel_size=3, padding=1)
        self.gam = GAM(in_channel)

    def forward(self, xa, xb, f0):
        xa = self.conv1_a(xa)
        xb = self.conv1_b(xb)
        xa_tmp = self.conv2_a(xb)
        xb_tmp = self.conv2_b(xa)
        xa_tmp = torch.mul(xa_tmp, xa)
        xb_tmp = torch.mul(xb_tmp, xb)
        xa = torch.add(xa, xa_tmp)
        xb = torch.add(xb, xb_tmp)
        xa = self.conv3_a(xa)
        xb = self.conv3_b(xb)
        x = torch.cat((xa, xb), dim=1)
        x = self.conv(x)
        if f0 is not None:
            f0 = self.downsample(f0)
            x = torch.cat((x, f0), dim=1)
            x = self.conv_last(x)
        x = self.gam(x)
        return x


class GAM(nn.Module):
    def __init__(self, in_channel):
        super(GAM, self).__init__()
        self.a1 = torch.nn.Parameter(torch.randn(1))
        self.a2 = torch.nn.Parameter(torch.randn(1))
        self.a3 = torch.nn.Parameter(torch.randn(1))
        self.conv = regular_conv1(in_channel)
        self.bn = nn.BatchNorm1d(in_channel)

    def forward(self, x):
        x1 = x.reshape(x.size(0), -1, 1) * self.a1
        x2 = x.reshape(x.size(0), 1, -1) * self.a2
        x3 = x.reshape(x.size(0), -1, 1) * self.a3
        x_tmp = torch.matmul(x1, x2)
        x_tmp = torch.matmul(x_tmp, x3)
        x_tmp = torch.reshape(x_tmp, x.size())
        x_tmp = self.conv(x_tmp)
        x_tmp = self.bn(x_tmp)
        x = torch.add(x, x_tmp)
        return x


def make_layer(in_channel, out_channel):
    conv_a = nn.Sequential(
        MSDM(in_channel, out_channel),
        GAM(out_channel))
    conv_b = nn.Sequential(
        MSDM(in_channel, out_channel),
        GAM(out_channel))
    cfm = CFM(out_channel, out_channel, in_channel)
    return conv_a, conv_b, cfm


def forward_a(out_a, out_b, out_c, conv_a, conv_b, cfm):
    out_a = conv_a(out_a)
    out_b = conv_b(out_b)
    out_c = cfm(out_a, out_b, out_c)
    return out_a, out_b, out_c


class CFCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=15):
        super(CFCNN, self).__init__()
        self.downsample = nn.AvgPool1d(2, 2)
        self.conv1_a = nn.Conv1d(in_channel, 16, kernel_size=32, stride=16)
        self.conv1_b = nn.Conv1d(in_channel, 16, kernel_size=32, stride=16)
        self.cfm1 = CFM(16, 16)
        self.conv2_a, self.conv2_b, self.cfm2 = make_layer(16, 32)
        self.conv3_a, self.conv3_b, self.cfm3 = make_layer(32, 64)
        self.conv4_a, self.conv4_b, self.cfm4 = make_layer(64, 128)
        self.gap = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Linear(128, out_channel))

    def forward(self, x):
        x = self.downsample(x)
        out_a, out_b, out_c = forward_a(x, x, None, self.conv1_a, self.conv1_b, self.cfm1)
        out_a, out_b, out_c = forward_a(out_a, out_b, out_c, self.conv2_a, self.conv2_b, self.cfm2)
        out_a, out_b, out_c = forward_a(out_a, out_b, out_c, self.conv3_a, self.conv3_b, self.cfm3)
        out_a, out_b, out_c = forward_a(out_a, out_b, out_c, self.conv4_a, self.conv4_b, self.cfm4)
        out = torch.cat((out_a, out_b, out_c), dim=1)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary
    model = CFCNN(1, 15)
    summary(model, input_size=(1, 12000), device='cpu')
