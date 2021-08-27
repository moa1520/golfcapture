import matplotlib.pyplot as plt
import pytorch_model_summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import KeypointDB, NormalizeForHeatmap, ToTensorForHeatmap
from models.model_resnet_heatmap import Plan1, Plan2
from models.resnet import resnet18
from util import UnNormalize
import util


def test():
    model = Plan2(pretrain=True, width_mult=1, lstm_layers=1,
                  lstm_hidden=256, bidirectional=True, dropout=False)
    util.freeze_layers(3, model)

    input = torch.zeros(6, 64, 3, 224, 224).cuda()
    heatmap = torch.zeros(6, 64, 224, 224).cuda()
    model.cuda()

    print(pytorch_model_summary.summary(model, input, heatmap))


def main():
    # model = resnet18(pretrained=True)
    # cnn = nn.Sequential(*list(model.children())[:-1])
    ds = KeypointDB(video_path='data/total_videos',
                    label_path='front_label/train.json',
                    npy_path='keypoint_npys',
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

    plt.subplot(1, 4, 1)
    plt.imshow(img[0].permute((1, 2, 0)).detach().numpy())

    plt.subplot(1, 4, 2)
    plt.imshow(heatmap[0].detach().numpy(), cmap='jet')

    heatmap = torch.unsqueeze(heatmap, dim=1)
    summed = img + heatmap
    concatencated = torch.cat((img, heatmap), dim=1)
    print(concatencated.shape)
    plt.subplot(1, 4, 3)
    plt.imshow(summed[0].sum(dim=0).detach().numpy())

    plt.subplot(1, 4, 4)
    plt.imshow(concatencated[0].sum(dim=0).detach().numpy())
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
