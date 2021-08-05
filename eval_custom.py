import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import CustomGolfDB, ToTensor, Normalize
from model import EventDetector
from util import correct_preds

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU 1 to use


def eval(model, seq_length, disp):
    dataset = CustomGolfDB(video_path='total_videos',
                           label_path='custom_label/val_label.json',
                           seq_length=seq_length,
                           transform=transforms.Compose(
                               [ToTensor(),
                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                           train=False,
                           input_size=512)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    correct = []
    deltas = []

    for i, sample in enumerate(tqdm(data_loader)):
        images, labels = sample['images'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch *
                                        seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(
                    logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, d, _, c = correct_preds(probs, labels.squeeze(), tol=3)
        if disp:
            print(i, c)
        correct.append(c)
        deltas.append(d)
    print(np.mean(correct, axis=0))
    print(np.mean(deltas, axis=0))
    PCE = np.mean(correct)
    wo_adf = np.mean(np.mean(correct, axis=0)[1:-1])
    return PCE, wo_adf


if __name__ == '__main__':
    seq_length = 16

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    save_dict = torch.load('models_512/swingnet_6000.pth.tar')
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    PCE, wo_adf = eval(model, seq_length, True)
    print('Average PCE: {}'.format(PCE))
    print('Average PCE w/o AD, F: {}'.format(wo_adf))
