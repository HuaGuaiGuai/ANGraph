import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = F.relu(out)
        return out


class ANG_P(nn.Module):
    def __init__(self):
        super(ANG_P, self).__init__()
        self.conv1 = nn.Linear(36, 128)  # 第一个全连接层

        self.block1 = ResidualBlock(128, 128)
        self.block2 = ResidualBlock(128, 128)
        self.block3 = ResidualBlock(128, 128)
        self.block4 = ResidualBlock(128, 128)

        self.conv2 = nn.Linear(128, 64)  # 中间层
        self.bn2 = nn.BatchNorm1d(64)

        self.block5 = ResidualBlock(64, 64)
        self.block6 = ResidualBlock(64, 64)

        self.conv3 = nn.Linear(64, 16)  # 输出层
        self.bn3 = nn.BatchNorm1d(16)
        self.output_1 = nn.Linear(16, 1)
        self.output_2 = nn.Linear(16, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = F.relu(self.conv2(out))
        out = self.block5(out)
        out = self.block6(out)

        out = self.conv3(out)
        out = F.relu(out)
        output_1 = self.output_1(out)
        output_2 = self.output_2(out)
        return output_1, output_2