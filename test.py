from VC.utils.func import generate_heatmaps
import matplotlib.pyplot as plt
import numpy as np
import pytorch_model_summary
import torch
import torch.nn as nn
from torch.autograd.grad_mode import F
from torch.utils.data import DataLoader
from torchvision import transforms

import util
from dataloader import KeypointDB, NormalizeForHeatmap, ToTensorForHeatmap
from models.MobileNetV2 import MobileNetV2
from models.model_resnet_heatmap import (Plan1, Plan1_concat, Plan1_repeat,
                                         Plan2)
from models.resnet import resnet18
from models.resnet_attn import resnet18
from util import UnNormalize
import cv2


def test():
    # img_path = '/media/tk/SSD_250G/tk_unlabeled_videos/front_20_frames/bad_front_swing0018/bad_front_swing0018_0080.png'
    heatmap_path = '/home/tk/Desktop/ktk/bad_front_swing0002_0000.npy'
    npy = np.load(heatmap_path)
    print(npy.shape)

    # img = cv2.imread(img_path)
    # frame_size = [img.shape[0], img.shape[1]]
    # ratio = 224 / max(frame_size)
    # new_size = tuple([int(x * ratio) for x in frame_size])
    # delta_w = 224 - new_size[1]
    # delta_h = 224 - new_size[0]
    # top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    # left, right = delta_w // 2, delta_w - (delta_w // 2)

    # img = cv2.resize(img, (new_size[1], new_size[0]))
    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
    #                          value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # heatmaps = np.load(heatmap_path)[0] * 224
    # heatmaps = generate_heatmaps(heatmaps, 224, 3)
    # heatmaps = heatmaps.sum(0)

    # plt.imshow(heatmaps, cmap='jet')
    # plt.show()


def plan1_test():
    model = Plan1_repeat(pretrain=True, width_mult=1, lstm_layers=1,
                         lstm_hidden=256, bidirectional=True, dropout=False).cuda()

    ds = KeypointDB(video_path='data/total_videos',
                    label_path='fs_labels/train_label.json',
                    npy_path='data/all_keypoint_npys',
                    heatmap_size=56,
                    seq_length=64,
                    transform=transforms.Compose([ToTensorForHeatmap(), NormalizeForHeatmap(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True,
                    input_size=224)
    dl = DataLoader(ds, batch_size=2, shuffle=False, drop_last=True)
    di = iter(dl)
    sample = next(di)
    img, heatmap = sample['images'].cuda(), sample['heatmaps'].cuda()

    out = model(img, heatmap)
    print(out.shape)

    # unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # B, N, C, H, W = img.size()
    # img = img.view(B * N, C, H, W)
    # heatmap = heatmap.view(B*N, 56, 56)
    # img = unnorm(img)

    # plt.subplot(1, 2, 1)
    # plt.imshow(img[0].permute((1, 2, 0)).detach().numpy())

    # plt.subplot(1, 2, 2)
    # plt.imshow(heatmap[0].detach().numpy())
    # plt.show()


def plan2_test():
    # model = resnet18(pretrained=True)
    # cnn = nn.Sequential(*list(model.children())[:-1])
    ds = KeypointDB(video_path='data/total_videos',
                    label_path='fs_labels/train_label.json',
                    npy_path='data/all_keypoint_npys',
                    heatmap_size=224,
                    seq_length=64,
                    transform=transforms.Compose([ToTensorForHeatmap(), NormalizeForHeatmap(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True,
                    input_size=224)
    dl = DataLoader(ds, batch_size=2, shuffle=False, drop_last=True)
    di = iter(dl)
    sample = next(di)
    img, heatmap = sample['images'], sample['heatmaps']

    unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    B, N, C, H, W = img.size()
    img = img.view(B * N, C, H, W)
    heatmap = heatmap.view(B*N, 224, 224)
    img = unnorm(img)

    plt.subplot(2, 4, 1)
    plt.imshow(img[0].permute((1, 2, 0)).detach().numpy())

    plt.subplot(2, 4, 2)
    plt.imshow(heatmap[0].detach().numpy(), cmap='jet')

    heatmap = torch.unsqueeze(heatmap, dim=1)
    summed = img + heatmap
    concatencated = torch.cat((img, heatmap), dim=1)
    multied = img * heatmap
    print(multied.shape)

    plt.subplot(2, 4, 3)
    plt.imshow(summed[0].sum(dim=0).detach().numpy())

    plt.subplot(2, 4, 4)
    plt.imshow(concatencated[0].sum(dim=0).detach().numpy())

    plt.subplot(2, 4, 5)
    plt.imshow(multied[0].permute((1, 2, 0)).detach().numpy())
    plt.show()

    # img = img[0][0].permute((1, 2, 0)).detach().numpy()
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(heatmap[0][0], cmap='jet')
    # plt.show()

    # input = img.view(2 * 64, 3, 224, 224)

    # out = cnn1(input)
    # heatmap = torch.unsqueeze(heatmap, 1)
    # print(out.shape)
    # print(heatmap.shape)
    # out = out + heatmap
    # print(out.shape)

    # out = cnn2(out)
    # print(out.shape)

    # print(pytorch_model_summary.summary(cnn1, torch.zeros(2, 3, 224, 224)))


if __name__ == '__main__':
    # main()
    test()
    # plan1_test()
