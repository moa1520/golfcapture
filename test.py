import pytorch_model_summary
import torch
from torch.hub import load_state_dict_from_url

from models.model_resnet_gru_attn import EventDetector


def main():
    net = EventDetector(pretrain=True,
                        width_mult=1.,
                        lstm_layers=1,
                        lstm_hidden=256,
                        bidirectional=True,
                        dropout=False)
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth',
                                          progress=True)
    new_model_dict = net.state_dict()

    pretrained_dict = {k: v for k, v in state_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    net.load_state_dict(new_model_dict)
    net.cuda()

    print(pytorch_model_summary.summary(net, torch.zeros(3, 4, 3, 512, 512).cuda()))


if __name__ == '__main__':
    main()
    # 12,363,849 GRU
    # 12,758,089 LSTM
