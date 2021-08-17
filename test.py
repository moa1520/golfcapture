import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms

from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder
from dataloader import CustomGolfDB, ToTensor, Normalize


def viz(img, flo):
    img = img[0].permute(1, 2, 0).detach().cpu().numpy()
    flo = flo[0].permute(1, 2, 0).detach().cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    plt.imshow(img_flo / 255.0)
    plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='models/raft-things.pth')
    parser.add_argument('--path', help="dataset for evaluation", default='demo-frames')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    dataset = CustomGolfDB(
        video_path='total_videos/',
        label_path='custom_label/train_label.json',
        seq_length=64,
        transform=transforms.Compose([ToTensor(), Normalize(
            # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        train=True,
        input_size=512
    )
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             drop_last=True)

    dataiter = iter(data_loader)
    batch = next(dataiter)
    images = batch['images']

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('RAFT/models/raft-things.pth'))
    model = model.module
    model.cuda()
    model.eval()

    for image1, image2 in zip(images[0][:-1], images[0][1:]):
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        viz(image1, flow_up)


if __name__ == '__main__':
    main()
    # 12,363,849 GRU
    # 12,758,089 LSTM
