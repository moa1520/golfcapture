import time

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import CustomGolfDB, KeypointDB, NormalizeForHeatmap, ToTensorForHeatmap, ToTensor, Normalize
from models.model_resnet_heatmap import Plan1
from models.model_resnet import EventDetector
from util import correct_preds

# Arrange GPU devices starting from 0
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU 1 to use


def eval(model, seq_length, disp, input_size):
    dataset = CustomGolfDB(video_path='data/total_videos',
                           label_path='front_labels/test.json',
                           seq_length=seq_length,
                           transform=transforms.Compose(
                               [ToTensor(),
                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                           train=False,
                           input_size=input_size)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    correct = []

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        # images, labels, heatmaps = sample['images'], sample['labels'], sample['heatmaps']
        # heatmaps = torch.unsqueeze(heatmaps, dim=2)
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
                # heatmap_batch = heatmaps[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch *
                                     seq_length:(batch + 1) * seq_length, :, :, :]
                # heatmap_batch = heatmaps[:, batch *
                #                          seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            # logits = model(image_batch.cuda(), heatmap_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(
                    logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze(), tol=3)
        if disp:
            print(i, c)
        correct.append(c)
    PCEs = np.mean(correct, axis=0)
    print(PCEs)
    PCE = np.mean(correct)
    PCEwo = np.mean(PCEs[1:7])
    return PCE, PCEwo


if __name__ == '__main__':
    start_time = time.time()
    seq_length = 64
    input_size = 224
    saved_path = 'saved_dicts/models_224_front/swingnet_2500.pth.tar'

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    save_dict = torch.load(saved_path)
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    print('Evaluation start : {}'.format(
        saved_path.split('/')[-1].split('.')[0]))
    PCE, PCEwo = eval(model, seq_length, True, input_size)
    print('Evaluation end : {}'.format(
        saved_path.split('/')[-1].split('.')[0]))
    print('Average PCE: {}'.format(PCE))
    print('Average PCE w/o AD, F: {}'.format(PCEwo))
    print('time: {}min'.format((time.time() - start_time) // 60))
