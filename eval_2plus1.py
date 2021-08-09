import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import ToTensor, Normalize
from model_2plus1 import r2plus1d_18
from test_video_custom import SampleVideo


def main():
    seq_length = 64

    model = r2plus1d_18(progress=True, num_classes=8)
    dataset = SampleVideo('total_videos/bad_side_swing1277.mp4',
                          transform=transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406],
                                                                              [0.229, 0.224, 0.225])]),
                          input_size=160)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    save_dict = torch.load('model_2Plus1/swingnet_1800.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    print("Loaded model weights")
    print("Testing...")

    for sample in data_loader:
        images = sample['images'].cuda()
        bs, step, C, H, W = images.size()
        images = images.view(bs, C, step, H, W)
        out = model(images)
        print(out)


if __name__ == '__main__':
    main()
