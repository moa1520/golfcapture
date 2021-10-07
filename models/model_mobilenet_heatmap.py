from models.MobileNetV2_transfer import MobileNetV2
import torch
import torch.nn as nn
import torch.nn.functional as F


class Plan1_concat(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, heatmap_size, bidirectional=True, dropout=True):
        super(Plan1_concat, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.heatmap_size = heatmap_size

        net = MobileNetV2(width_mult=width_mult)
        self.cnn1 = nn.Sequential(*list(net.children())[0][:4])
        self.cnn2 = nn.Sequential(*list(net.children())[0][4:19])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if pretrain:
            state_dict = torch.load(
                'mobilenet_v2.pth.tar', map_location=device)
            new_model_dict = self.cnn1.state_dict()
            pretrained_dict = {k: v for k,
                               v in state_dict.items() if k in new_model_dict}
            new_model_dict.update(pretrained_dict)
            self.cnn1.load_state_dict(new_model_dict)

        self.rnn = nn.LSTM(int(1280 * width_mult if width_mult > 1.0 else 1280),
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
            batch_size * timesteps, 21, self.heatmap_size, self.heatmap_size)
        c_out = self.cnn1(c_in)
        c_out = torch.cat((c_out, heatmaps), dim=1)
        c_out = self.cnn2(c_out)
        c_out = c_out.mean(3).mean(2)

        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out
