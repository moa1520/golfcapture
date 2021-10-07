import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from models.resnet import resnet18
from VC.configs.options import BaseOptions
from VC.models.ScalePoseNet import ScalePoseNet


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


class TotalActionDetection(nn.Module):
    def __init__(self):
        super(TotalActionDetection, self).__init__()
        self.action_detection_model = Plan1_concat(pretrain=True, width_mult=1, lstm_layers=2, lstm_hidden=512,
                                                   bidirectional=True, dropout=False, heatmap_size=56).cuda()
        self.pose_model = ScalePoseNet(opt).cuda()

        dict_action = torch.load(
            'checkpoints/plan1_concat/swingnet_18000(sizeup_best).pth.tar', map_location=cuda)
        self.action_detection_model.load_state_dict(
            dict_action['model_state_dict'])
        self.pose_model.load_state_dict(torch.load(
            'VC/models/ScalePoseNet_latest', map_location=cuda))

    def forward(self, images, heatmaps):
        k = 0
        batch, seq_length, C, H, W = images.size()
        while k * seq_length < images.shape[1]:
            image_batch = images[:, batch * seq_length:, :, :, :]
            heatmap_batch = heatmaps[:, batch * seq_length:, :, :, :]

        return images


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
