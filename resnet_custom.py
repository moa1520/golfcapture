import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionBlock, ProjectorBlock


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_features)
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.in_features != self.out_features:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_features = 64
        self.conv1 = nn.Conv2d(3, self.in_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 64)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 128)
        self.layer5 = ResidualBlock(128, 256, stride=2)
        self.layer6 = ResidualBlock(256, 256)
        self.layer7 = ResidualBlock(256, 512, stride=2)
        self.layer8 = ResidualBlock(512, 512)

        # self.dense = nn.Conv2d(512, 512, kernel_size=int(im_size / 256), padding=0, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(1536, num_classes)

        self.projector1 = ProjectorBlock(64, 512)
        self.projector2 = ProjectorBlock(128, 512)
        self.projector3 = ProjectorBlock(256, 512)
        self.attn1 = AttentionBlock(512, normalize_attn=True)
        self.attn2 = AttentionBlock(512, normalize_attn=True)
        self.attn3 = AttentionBlock(512, normalize_attn=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        l1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer3(l1)
        l2 = F.max_pool2d(self.layer4(x), kernel_size=2, stride=2, padding=0)
        x = self.layer5(l2)
        l3 = F.max_pool2d(self.layer6(x), kernel_size=2, stride=2, padding=0)
        x = self.layer7(l3)
        x = self.layer8(x)
        g = self.avgpool(x)

        c1, g1 = self.attn1(self.projector1(l1), g)
        c2, g2 = self.attn2(self.projector2(l2), g)
        c3, g3 = self.attn3(self.projector3(l3), g)
        g = torch.cat((g1, g2, g3), dim=1)

        return [g, c1, c2, c3]
