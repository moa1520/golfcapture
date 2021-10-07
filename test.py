import json

import matplotlib.pyplot as plt
import torch
import numpy as np


def test():
    a = np.array([0.5352, 0.5039])
    b = np.array([0.6406, 0.6484])

    xx_ = torch.linspace(a[0], b[0], steps=50)
    yy_ = torch.linspace(a[1], b[1], steps=50)

    for x, y in zip(xx_, yy_):
        print(x, y)


if __name__ == '__main__':
    test()
