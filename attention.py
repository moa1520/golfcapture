import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.cnn = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.cnn(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.cnn = nn.Conv2d(in_features, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, H, W = l.size()
        c = self.cnn(l + g)  # batch_size * 1 * W * H
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, H, W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N, C, -1).sum(dim=2)  # batch_size * C
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)

        return c.view(N, 1, H, W), g
