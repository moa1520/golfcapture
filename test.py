import pytorch_model_summary
import torch

from model_2plus1_revised import r2plus1d_18


def main():
    model = r2plus1d_18(pretrained=False, num_classes=9, seq_length=32)
    model.cuda()
    input = torch.zeros(4, 3, 32, 512, 512).cuda()
    output = model(input)


    print(output.shape)


if __name__ == '__main__':
    main()
