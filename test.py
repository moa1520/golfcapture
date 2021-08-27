from util import UnNormalize
import matplotlib.pyplot as plt
import pytorch_model_summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import KeypointDB, NormalizeForHeatmap, ToTensorForHeatmap
from models.model_resnet_heatmap import EventDetector
from models.resnet import resnet18


def main():
    model = resnet18(pretrained=True)
    cnn1 = nn.Sequential(*list(model.children())[:4])
    cnn2 = nn.Sequential(*list(model.children())[4:-1])
    ds = KeypointDB(video_path='data/total_videos',
                    label_path='front_label/train.json',
                    npy_path='/home/tk/Desktop/ktk/keypoint_npys',
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
    heatmap = heatmap.view(B*N, 1, 56, 56)

    out1 = cnn1(img)
    plt.subplot(1, 2, 1)
    plt.imshow(out1[0].sum(dim=0).detach().numpy(), cmap='jet')

    summed = out1 + heatmap

    plt.subplot(1, 2, 2)
    plt.imshow(summed[0].sum(dim=0).detach().numpy(), cmap='jet')
    plt.show()

    # img = unnorm(img)
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
    main()
