import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import ToTensor, Normalize
from models.model_2plus1 import r2plus1d_18
from test_video_custom import SampleVideo


def main():
    seq_length = 64
    input_size = 160

    model = r2plus1d_18(progress=True, num_classes=8)
    # dataset = TwoPlusOneDB(
    #     video_path='total_videos/',
    #     label_path='fs_labels/train_label.json',
    #     seq_length=seq_length,
    #     transform=transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     train=False,
    #     input_size=input_size
    # )
    dataset = SampleVideo('total_videos/bad_side_swing0012.mp4',
                          transform=transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406],
                                                                              [0.229, 0.224, 0.225])]),
                          input_size=160)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    save_dict = torch.load('model_2Plus1/swingnet_10000.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    print("Loaded model weights")
    print("Testing...")

    for sample in data_loader:
        images, labels = sample['images'], sample['labels']
        bs, step, C, H, W = images.size()
        images = images.view(bs, C, step, H, W)
        batch = 0

        outs = []

        while batch * seq_length < images.shape[2]:
            if (batch + 1) * seq_length > images.shape[2]:
                image_batch = images[:, :, batch * seq_length:, :, :]
            else:
                image_batch = images[:, :, batch * seq_length:(batch + 1) * seq_length, :, :]
            out = model(image_batch.cuda()).cpu().detach().numpy()
            outs.append(out)
            batch += 1
        outs = np.asarray(outs)
        print(outs)
        print(labels)


if __name__ == '__main__':
    main()
