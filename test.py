import numpy as np
import torch
import pytorch_model_summary

from VC.configs.options import BaseOptions
from VC.models.ScalePoseNet import ScalePoseNet


def linspace(start, end, steps):
    delta = end - start
    div = delta / steps
    value = start
    y = []
    y.append(start)
    while value < end:
        value += div
        y.append(value)

    print(y)


def main():
    linspace(0, 30, 5)


def test():
    cuda = torch.device('cuda')
    opt = BaseOptions().parse()
    model = ScalePoseNet(opt).cuda()
    input = torch.randn(1, 3, 512, 512, device=cuda)
    model.load_state_dict((torch.load(
        'VC/models/ScalePoseNet_latest', map_location=cuda)))

    out = model(input)
    print(out.shape)


if __name__ == '__main__':
    test()
