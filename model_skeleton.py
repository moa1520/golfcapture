import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from MobileNetV2 import MobileNetV2
from VC.models.RefineNet import RefineNet
from VC.models.ScalePoseNet import ScalePoseNet
from VC.options import BaseOptions
from dataloader import CustomGolfDB, Normalize, ToTensor


class fusion(nn.Module):
    def __init__(self, in_feature=1664, out_feature=1280):
        super(fusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_feature, 512),
            nn.ReLU(),
            nn.Linear(512, out_feature),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        opt = BaseOptions().parse()

        net = MobileNetV2(width_mult=width_mult)  # MobileNetV2
        self.scale_posenet = ScalePoseNet(opt)  # ScalePoseNet
        self.rf_model = RefineNet(opt)  # RefineNet

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_dict_posenet = torch.load(
            'VC/models/ScalePoseNet_latest', map_location=device)
        self.scale_posenet.load_state_dict(state_dict_posenet)

        # state_dict_ref = torch.load(
        #     'VC/models/RefineNet_latest', map_location=device)
        # self.rf_model.load_state_dict(state_dict_ref)

        state_dict_mobilenet = torch.load(
            'mobilenet_v2.pth.tar', map_location=device)
        if pretrain:
            net.load_state_dict(state_dict_mobilenet)

        self.scale_posenet.eval()
        # self.rf_model.eval()

        self.fusion = nn.Sequential(
            nn.Linear(1664, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1280),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.rnn = nn.LSTM(int(1280 * width_mult if width_mult > 1.0 else 1280),
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

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # PoseNet forward -> out shape: (batch * timesteps, 384)
        p_in = x.view(batch_size * timesteps, C, H, W)
        for i in range(batch_size * timesteps):
            heatmaps = self.scale_posenet(p_in[i, :].unsqueeze(0))
            if i == 0:
                p_outs = heatmaps.mean(3).mean(2)
            else:
                p_outs = torch.cat([p_outs, heatmaps.mean(3).mean(2)], dim=0)

        # CNN forward -> out shape: (batch * timesteps, 1280)
        c_out = self.cnn(p_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        # Fusion
        f_in = torch.cat([c_out, p_outs], dim=1)
        f_in = f_in.view(batch_size, timesteps, -1)  # (batch, timesteps, 1664)
        f_out = self.fusion(f_in)  # (batch, timesteps, 1664)

        # LSTM forward
        # r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(f_out, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out


if __name__ == '__main__':
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    model.cuda()
    dataset = CustomGolfDB(
        video_path='total_videos/',
        label_path='custom_label/train_label.json',
        seq_length=16,
        transform=transforms.Compose([ToTensor(), Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        train=True,
        input_size=512
    )
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             drop_last=True)
    dataiter = iter(data_loader)
    sample = next(dataiter)
    images, labels = sample['images'].cuda(), sample['labels'].cuda()

    output = model(images)
    print(output.shape)
