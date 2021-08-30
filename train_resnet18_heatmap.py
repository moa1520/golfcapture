import argparse
import os
import time
from pathlib import Path

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

import util
from dataloader import KeypointDB, NormalizeForHeatmap, ToTensorForHeatmap
from models.model_resnet_heatmap import Plan1, Plan2

if __name__ == '__main__':
    writer = SummaryWriter()

    # training configuration
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--input_size', type=int,
                        help='image size of input', default=224)
    parser.add_argument('--iterations', type=int,
                        help='the number of training iterations', default=2500)
    parser.add_argument('--it_save', type=int,
                        help='save model every what iterations', default=500)
    parser.add_argument('--seq_length', type=int,
                        help='divided frame numbers', default=64)
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='batch size', default=8)
    parser.add_argument('--frozen_layers', '-k', type=int,
                        help='the number of frozen layers', default=5)
    parser.add_argument('--heatmap_size', type=int,
                        help='the size of heatmap', default=56)
    parser.add_argument('--save_folder', type=str,
                        help='divided frame numbers', default='checkpoints/plan1')

    arg = parser.parse_args()

    model = Plan1(pretrain=True,
                  width_mult=1.,
                  lstm_layers=1,
                  lstm_hidden=256,
                  bidirectional=True,
                  dropout=False)
    # util.freeze_layers(arg.frozen_layers, model)
    model.train()
    model.cuda()

    dataset = KeypointDB(
        video_path='data/total_videos',
        label_path='front_labels/train.json',
        npy_path='keypoint_npys',
        heatmap_size=arg.heatmap_size,
        seq_length=arg.seq_length,
        transform=transforms.Compose([ToTensorForHeatmap(), NormalizeForHeatmap(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        train=True,
        input_size=arg.input_size
    )
    data_loader = DataLoader(dataset,
                             batch_size=arg.batch_size,
                             shuffle=True,
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor(
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = util.AverageMeter()

    Path(arg.save_folder).mkdir(parents=True, exist_ok=True)

    i = 0

    pretrained = False
    if pretrained:
        state_dict = torch.load(
            'saved_dicts/224_heatmap_plan2/swingnet_250.pth.tar', map_location=torch.device('cuda'))
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        i = state_dict['iterations']
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    start_time = time.time()

    while i < arg.iterations:
        for sample in data_loader:
            images, labels, heatmaps = sample['images'].cuda(
            ), sample['labels'].cuda(), sample['heatmaps'].cuda()
            logits = model(images, heatmaps)
            labels = labels.view(arg.batch_size * arg.seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                i, loss=losses))
            print('time : {}min'.format((time.time() - start_time) // 60))

            writer.add_scalar('Loss', losses.val, i)
            writer.add_scalar('Avg Loss', losses.avg, i)

            i += 1
            if i % arg.it_save == 0:
                if isinstance(model, torch.nn.DataParallel):
                    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                'model_state_dict': model.module.state_dict(),
                                'iterations': i},
                               arg.save_folder + '/swingnet_{}.pth.tar'.format(i))
                else:
                    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                'model_state_dict': model.state_dict(),
                                'iterations': i}, arg.save_folder + '/swingnet_{}.pth.tar'.format(i))
            if i == arg.iterations:
                break

    writer.close()
