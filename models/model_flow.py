import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from dataloader import CustomGolfDB, Normalize, ToTensor
from flow import get_flows

# class FlowNet(nn.Module):
#     def __init__(self):
#         super(FlowNet, self).__init__()


if __name__ == '__main__':
    dataset = CustomGolfDB(
        video_path='../total_videos/',
        label_path='../custom_label/train_label.json',
        seq_length=32,
        transform=transforms.Compose([ToTensor(), Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        train=True,
        input_size=512
    )
    data_loader = DataLoader(dataset,
                             batch_size=12,
                             shuffle=True,
                             drop_last=True)
    dataiter = iter(data_loader)
    image_batch = next(dataiter)
    images = image_batch['images']

    get_flows(images)
