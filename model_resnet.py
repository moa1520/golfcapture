import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url

from resnet import resnet18


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        net = resnet18(pretrained=False)
        if pretrain:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth',
                                                  progress=True)
            new_model_dict = net.state_dict()

            pretrained_dict = {k: v for k, v in state_dict.items() if k in new_model_dict}
            new_model_dict.update(pretrained_dict)
            net.load_state_dict(new_model_dict)
        net.cuda()

        self.cnn = net
        # self.rnn = nn.LSTM(int(512 * width_mult if width_mult > 1.0 else 512),
        #                    self.lstm_hidden, self.lstm_layers,
        #                    batch_first=True, bidirectional=bidirectional)
        self.rnn = nn.LSTM(int(1536 * width_mult if width_mult > 1.0 else 1536),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            if torch.cuda.is_available():
                return (
                    Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).cuda(),
                             requires_grad=True),
                    Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).cuda(),
                             requires_grad=True))
            else:
                return (Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden), requires_grad=True),
                        Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden), requires_grad=True))
        else:
            if torch.cuda.is_available():
                return (
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))
            else:
                return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden), requires_grad=True),
                        Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden), requires_grad=True))

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out, c1, c2, c3 = self.cnn(c_in)
        # c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out, c1, c2, c3
