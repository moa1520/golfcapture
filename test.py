import pytorch_model_summary
import torch

from model_resnet import EventDetector


def main():
    model = EventDetector(pretrain=False, width_mult=1, lstm_layers=1, lstm_hidden=256, bidirectional=True,
                          dropout=False)

    input = torch.zeros(4, 8, 3, 512, 512).cuda()
    model.cuda()
    print(pytorch_model_summary.summary(model, input))


if __name__ == '__main__':
    main()
