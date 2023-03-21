import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torch.optim as optim
import sys

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.adpavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(2048, 200)
        self.fc2 = nn.Linear(200, 20)
        self.afc1 = nn.Tanh()
        self.fc1 = nn.Linear(20, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            '''elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)'''

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def auto_fact(self, input):
        # input_redistr = self.redistribution(input)
        input_redistr = input
        max = torch.max(input_redistr).item()
        min = torch.min(input_redistr).item()
        mean = torch.mean(input_redistr).item()
        if abs(max) > abs(min):
            a = abs(max)
        else:
            a = abs(min)
        if a != 0:
            m = 2.5 / a  # 缩放因子
            # m = 4.5 / a
            out = input_redistr * m
            return out
        else:
            return input

    def auto_fact_v2(self, input):
        # input_redistr = self.redistribution(input)
        input_redistr = input
        max = torch.max(input_redistr).item()
        min = torch.min(input_redistr).item()
        mean = torch.mean(input_redistr).item()
        if abs(max) > abs(min):
            a = abs(max)
        else:
            a = abs(min)
        if a != 0:
            std = torch.var(input_redistr / a).item()  # 输入数据的方差
            if std <= 1.4 or std >= 2.4:
                z = (2.2 / std) ** 0.5  # 方差控制因子
            else:
                z = 1
            m = z / a  # 缩放因子
            out = input_redistr * m
            return out
        else:
            return input

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        x = self.fc2(x)  # 在res50中增加
        x = x * 4.127e-05
        x = self.afc1(x)
        x = self.fc1(x)
        x = x.squeeze(-1)
        return x


def ResNet50():
    return ResNet([3, 4, 6, 3])


def ResNet101():
    return ResNet([3, 4, 23, 3])


def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__ == "__main__":
    input = torch.randn(64, 3, 224, 224)
    model = ResNet50()

    with SummaryWriter(comment='Regression model') as w:
        w.add_graph(model, (input,))
