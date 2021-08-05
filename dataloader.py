import json
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomGolfDB(Dataset):
    def __init__(self, video_path, label_path, seq_length, transform=None, train=True, input_size=160):
        self.video_path = video_path
        self.label_path = label_path
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.input_size = input_size

    def __len__(self):
        with open(self.label_path, 'r') as json_file:
            label = json.load(json_file)
        return len(label['video_name'])

    def __getitem__(self, index):
        with open(self.label_path, 'r') as json_file:
            label = json.load(json_file)
        video_name = label['video_name'][index]
        width = label['width'][index]
        height = label['height'][index]
        events = np.asarray(label['events'][index])
        images, labels = [], []
        cap = cv2.VideoCapture(
            osp.join(self.video_path, '{}.mp4'.format(video_name)))

        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                      cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        if self.train:
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.resize(img, (new_size[1], new_size[0]))
                    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                             value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events:
                        labels.append(np.where(events == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.resize(img, (new_size[1], new_size[0]))
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                         value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                if pos in events:
                    labels.append(np.where(events == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        # now frame #s correspond to frames in preprocessed video clips
        events -= events[0]

        images, labels = [], []
        cap = cv2.VideoCapture(
            osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))

        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(
            self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


if __name__ == '__main__':
    # ImageNet mean and std (RGB)
    # norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # dataset = GolfDB(data_file='data/train_split_1.pkl',
    #                  vid_dir='data/videos_160/',
    #                  seq_length=64,
    #                  transform=transforms.Compose([ToTensor(), norm]),
    #                  train=True)

    # data_loader = DataLoader(dataset, batch_size=1,
    #                          shuffle=False, num_workers=6, drop_last=False)
    # dataiter = iter(data_loader)
    # sample = next(dataiter)
    # img = sample['images']
    # print(img.shape)
    # sample = next(dataiter)
    # img = sample['images']
    # print(img.shape)

    # for i, sample in enumerate(data_loader):
    #     images, labels = sample['images'], sample['labels']
    #     events = np.where(labels.squeeze() < 8)[0]
    #     print('{} events: {}'.format(len(events), events))

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    video_path = '/home/tk/Desktop/ktk/golf_data/videos'
    label_path = 'custom_data/label.json'

    dataset = CustomGolfDB(video_path=video_path, seq_length=64, transform=transforms.Compose(
        [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), label_path=label_path, train=False)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, drop_last=True)
    dataiter = iter(dataloader)

    sample = next(dataiter)
    img = sample['images'][0][0]
    img = img.permute((1, 2, 0))
    img = img * std + mean
    plt.imshow(img)
    plt.show()
