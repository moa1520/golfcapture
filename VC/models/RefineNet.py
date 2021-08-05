import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.is_downsample = (in_channels != out_channels)
        self.downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.is_downsample:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class RefineNet(nn.Module):
    def __init__(self, opt, criterion=nn.MSELoss(reduction='mean')):
        super(RefineNet, self).__init__()
        self.criterion = criterion
        self.num_joints = opt.num_joints

        self.rf_1 = BasicBlock(opt.num_channels * opt.num_blocks, opt.num_channels)
        self.rf_2 = BasicBlock(opt.num_channels, opt.num_joints)
        self.rf_3 = BasicBlock(opt.num_joints, 1)
        self.fc = nn.Linear((opt.input_size // 4) ** 2 + opt.num_joints * 2, opt.num_joints * 2)

        self.rf_pose = None
        self.loss = None

    def get_loss(self, target_pose):
        return self.criterion(self.rf_pose, target_pose)

    def forward(self, pose, features, target_pose=None):
        b, n = pose.shape[0], self.num_joints
        features = self.rf_1(features)
        features = self.rf_2(features)
        features = self.rf_3(features)
        offset = self.fc(torch.cat([features.reshape(b, -1), pose.reshape(b, -1)], dim=1))
        self.rf_pose = pose + offset.reshape(b, n, -1) / 100.0

        if target_pose is not None:
            self.loss = self.get_loss(target_pose)

        return self.rf_pose, self.loss
