import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from models.resnet import resnet18
from VC.configs.options import BaseOptions


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


def linspace(start, end, steps):
    delta = end - start
    div = delta / (steps - 1)
    value = start
    y = []
    y.append(start)
    while value < end:
        value += div
        y.append(value)

    return torch.Tensor(y)


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


def generate_heatmaps_torch(pose, size, sigma):
    # sigma = int(sigma * size / 128)
    sigma = int(sigma)
    heatmaps = torch.zeros(
        (pose.shape[0], size + 2 * sigma, size + 2 * sigma)).cuda()
    # heatmaps = np.zeros((pose.shape[0], size + 2 * sigma, size + 2 * sigma))
    win_size = 2 * sigma + 1

    x, y = torch.meshgrid(linspace(-sigma, sigma, steps=win_size),
                          linspace(-sigma, sigma, steps=win_size))
    # x, y = np.meshgrid(np.linspace(-sigma, sigma, num=win_size, endpoint=True),
    #                    np.linspace(-sigma, sigma, num=win_size, endpoint=True))
    dst = torch.sqrt(x*x + y*y)
    # dst = np.sqrt(x * x + y * y)
    mu = 0.000
    gauss = torch.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))
    # gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))
    club_gauss = gauss.clone() * 0.8

    for i in range(pose.shape[0]):
        # for i, [X, Y] in enumerate(pose):
        X = pose[i][0]
        Y = pose[i][1]
        X, Y = X.type(torch.int), Y.type(torch.int)
        if X < 0 or X >= size or Y < 0 or Y >= size:
            continue

        if i == 20:
            start = (pose[6] + pose[9]) / 2
            # start = pose[6]
            end = pose[20]

            xx_ = linspace(start[0], end[0], steps=50)
            yy_ = linspace(start[1], end[1], steps=50)
            for k in range(xx_.shape[0]):
                # for xx, yy in zip(xx_, yy_):
                xx = xx_[k]
                yy = yy_[k]
                heatmaps[i, yy.type(torch.int):yy.type(
                    torch.int) + win_size, xx.type(torch.int):xx.type(torch.int) + win_size] = club_gauss
            # for xx, yy in np.linspace(start, end, endpoint=True, num=50):
            #     heatmaps[i, int(yy): int(yy) + win_size, int(xx)
            #                     : int(xx) + win_size] = club_gauss

        heatmaps[i, Y: Y + win_size, X: X + win_size] = gauss

    heatmaps = heatmaps[:, sigma:-sigma, sigma:-sigma]

    return heatmaps


class Plan1_concat(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, heatmap_size, bidirectional=True, dropout=True):
        super(Plan1_concat, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.heatmap_size = heatmap_size

        net = resnet18(pretrained=False)
        if pretrain:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth',
                                                  progress=True)
            new_model_dict = net.state_dict()

            pretrained_dict = {k: v for k,
                               v in state_dict.items() if k in new_model_dict}
            new_model_dict.update(pretrained_dict)
            net.load_state_dict(new_model_dict)
        net.cuda()

        self.cnn1 = nn.Sequential(*list(net.children())[:4])
        self.cnn2 = nn.Sequential(*list(net.children())[4:-1])
        self.rnn = nn.LSTM(int(512 * width_mult if width_mult > 1.0 else 512),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x):
        k = int(x.size(0) / 216384)
        heatmaps = x[150528 * k:].reshape(1, k, 21, 56, 56)
        x = x[:150528 * k].reshape(1, k, 3, 224, 224)

        batch_size, timesteps, C, H, W = x.size()

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        heatmaps = heatmaps.view(
            batch_size * timesteps, 21, self.heatmap_size, self.heatmap_size)
        c_out = self.cnn1(c_in)
        c_out = torch.cat((c_out, heatmaps), dim=1)
        c_out = self.cnn2(c_out)

        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        out = F.softmax(out, dim=1)

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

        # features = torch.cat(features_list, dim=1).detach()

        if y is not None:
            self.loss = self.get_loss(y)

        heatmaps = generate_heatmaps_torch(pose[0] * 56, 56, 1)

        return heatmaps


opt = BaseOptions().parse()
cuda = torch.device('cuda')
model = Plan1_concat(pretrain=True, width_mult=1, lstm_layers=2, lstm_hidden=512,
                     bidirectional=True, dropout=False, heatmap_size=56).to(device=cuda)
pose_model = ScalePoseNet(opt).cuda()

dict_action = torch.load(
    'checkpoints/plan1_concat/swingnet_18000(sizeup_best).pth.tar', map_location=cuda)
model.load_state_dict(dict_action['model_state_dict'])
pose_model.load_state_dict(torch.load(
    'VC/models/ScalePoseNet_latest', map_location=cuda))


seq_length = 64
dummy_input = torch.randn(1, seq_length, 3, 224, 224, device=cuda).flatten()
dummy_input_heatmap = torch.randn(
    1, seq_length, 21, 56, 56, device=cuda).flatten()
dummy_input = torch.cat([dummy_input, dummy_input_heatmap])
dummy_input2 = torch.randn(1, 3, 512, 512, device=cuda)


torch.onnx.export(model, (dummy_input),
                  "onnx/action_detection.onnx", verbose=True, opset_version=11, dynamic_axes={'input': {0: 'seq_length'}})
torch.onnx.export(pose_model, dummy_input2,
                  "onnx/scale_pose.onnx", verbose=True, opset_version=11)
