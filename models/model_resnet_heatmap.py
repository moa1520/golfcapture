import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from models.resnet import resnet18


class Plan1(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(Plan1, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        net = resnet18(pretrained=pretrain)
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

    def forward(self, x, heatmaps):
        batch_size, timesteps, C, H, W = x.size()

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        heatmaps = heatmaps.view(batch_size * timesteps, 1, 56, 56)
        c_out = self.cnn1(c_in)
        c_out += heatmaps
        c_out = self.cnn2(c_out)

        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out


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


class Plan1_repeat(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, heatmap_size, bidirectional=True, dropout=True):
        super(Plan1_repeat, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.heatmap_size = heatmap_size

        net = resnet18(pretrained=pretrain)
        net.cuda()

        self.cnn1 = nn.Sequential(*list(net.children())[:4])
        self.cnn2 = nn.Sequential(*list(net.children())[4])
        self.cnn3 = nn.Sequential(*list(net.children())[5])
        self.cnn4 = nn.Sequential(*list(net.children())[6])
        self.cnn5 = nn.Sequential(*list(net.children())[7:-1])
        self.rnn = nn.LSTM(int(512 * width_mult if width_mult > 1.0 else 512),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x, heatmaps):
        batch_size, timesteps, C, H, W = x.size()

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        heatmaps = heatmaps.view(
            batch_size * timesteps, 1, self.heatmap_size, self.heatmap_size)
        out1 = self.cnn1(c_in)

        out1 = out1 + heatmaps
        out2 = self.cnn2(out1)

        out2 = out2 + out1
        out3 = self.cnn3(out2)

        out2 = F.max_pool2d(out2, kernel_size=3, stride=2, padding=1)
        out2 = out2.sum(dim=1).unsqueeze(1)
        out3 = out3 + out2
        out4 = self.cnn4(out3)

        out3 = F.max_pool2d(out3, kernel_size=3, stride=2, padding=1)
        out3 = out3.sum(dim=1).unsqueeze(1)
        out4 = out4 + out3
        c_out = self.cnn5(out4)

        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out
