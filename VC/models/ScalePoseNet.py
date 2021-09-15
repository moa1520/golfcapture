import torch
import torch.nn as nn

from VC.models.loss import JointsMSELoss


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
        self.downsample = nn.Sequential(
            conv3x3(in_channels, out_channels, stride), nn.BatchNorm2d(out_channels))

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


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True):
        super(DeconvBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, stride=1, mode='bilinear', align_corners=False):
        super(UpsampleBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.up1 = nn.Upsample(scale_factor=scale_factor,
                               mode=mode, align_corners=align_corners)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.up1(out)

        return out


class FeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kargs):
        super(FeatureBlock, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.bottle = nn.Conv2d(
            in_channels * 7, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(True)

        self.down_1 = self._downsampling(
            BasicBlock, in_channels, in_channels * 2)
        self.down_2 = self._downsampling(
            BasicBlock, in_channels * 2, in_channels * 4)
        # self.down_3 = self._downsampling(BasicBlock, in_channels, in_channels)
        # self.down_4 = self._downsampling(BasicBlock, 256, 512)

        # self.up_1 = self._upsampling(DeconvBlock, 64, out_channels, 3, 2, 1, 1, True)
        # self.up_2 = self._upsampling(DeconvBlock, 128, out_channels, 3, 4, 1, 3, True)
        # self.up_3 = self._upsampling(DeconvBlock, 256, out_channels, 3, 8, 1, 7, True)
        # self.up_4 = self._upsampling(DeconvBlock, 512, out_channels, 3, 16, 1, 15, True)

        self.up_1 = self._upsampling(
            UpsampleBlock, in_channels=in_channels * 2, out_channels=in_channels * 2, scale_factor=2)
        self.up_2 = self._upsampling(
            UpsampleBlock, in_channels=in_channels * 4, out_channels=in_channels * 4, scale_factor=4)
        # self.up_3 = self._upsampling(UpsampleBlock, in_channels=in_channels, out_channels=in_channels, scale_factor=8)
        # self.up_4 = self._upsampling(UpsampleBlock, in_channels=512, out_channels=out_channels, scale_factor=16)

    def _downsampling(self, block, in_channels, out_channels):
        layers = []
        layers.append(block(in_channels, out_channels))
        layers.append(self.avgpool)

        return nn.Sequential(*layers)

    def _upsampling(self, block, **kargs):
        layers = []
        layers.append(block(**kargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        # x3 = self.down_3(x2)
        # x4 = self.down_4(x3)

        out_x1 = self.up_1(x1)
        out_x2 = self.up_2(x2)
        # out_x3 = self.up_3(x3)
        # out_x4 = self.up_4(x4)

        output = torch.cat((
            x, out_x1, out_x2
        ), dim=1)

        output = self.relu(self.bottle(output))

        return output


class ScalePoseNet(nn.Module):
    def __init__(self, opt, criterion=JointsMSELoss(use_target_weight=False)):
        super(ScalePoseNet, self).__init__()
        self.criterion = criterion
        self.num_channels = opt.num_channels
        self.input_size = opt.input_size
        self.conv = nn.Conv2d(3, self.num_channels,
                              kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.num_blocks = opt.num_blocks

        for i in range(self.num_blocks):
            self.add_module('fb_{}'.format(i), FeatureBlock(
                self.num_channels, self.num_channels))

        # self.out_layer = nn.Conv2d(in_channels=64, out_channels=opt.num_joints, kernel_size=3, stride=1, padding=1)
        for i in range(self.num_blocks):
            self.add_module('out_{}'.format(i),
                            nn.Conv2d(in_channels=self.num_channels, out_channels=opt.num_joints, kernel_size=3,
                                      stride=1, padding=1))

        self.heatmaps_list = None
        self.loss = None

    def get_loss(self, target_heatmaps):
        loss = 0.0
        for i, heatmaps in enumerate(self.heatmaps_list):
            loss += i * \
                self.criterion(heatmaps, target_heatmaps) / \
                len(self.heatmaps_list)

        return loss

    def forward(self, x, y=None):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)

        self.heatmaps_list = []
        features_list = []
        for i in range(self.num_blocks):
            x = self._modules['fb_{}'.format(i)](x)
            features_list.append(x.clone())
            out = self._modules['out_{}'.format(i)](x)
            self.heatmaps_list.append(out)

        [b, n, h, w] = self.heatmaps_list[-1].shape
        tmp = self.heatmaps_list[-1].clone().detach().view(b,
                                                           n, -1).argmax(dim=2).view(b, n, 1)
        pose = torch.cat(
            (tmp % w, torch.div(tmp,  w, rounding_mode='trunc')), dim=2)
        pose = pose / w

        features = torch.cat(features_list, dim=1).detach()

        if y is not None:
            self.loss = self.get_loss(y)

        return pose, self.heatmaps_list[-1], features, self.loss
