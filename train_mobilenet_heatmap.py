import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import util
from dataloader import KeypointDB, KeypointDB_concat, NormalizeForHeatmap, ToTensorForHeatmap
from models.model_mobilenet_heatmap import Plan1_concat

if __name__ == '__main__':
    writer = SummaryWriter()

    # training configuration
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--input_size', type=int,
                        help='image size of input', default=224)
    parser.add_argument('--iterations', type=int,
                        help='the number of training iterations', default=10000)
    parser.add_argument('--it_save', type=int,
                        help='save model every what iterations', default=500)
    parser.add_argument('--seq_length', type=int,
                        help='divided frame numbers', default=64)
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='batch size', default=4)
    parser.add_argument('--frozen_layers', '-k', type=int,
                        help='the number of frozen layers', default=5)
    parser.add_argument('--heatmap_size', type=int,
                        help='the size of heatmap', default=56)
    parser.add_argument('--save_folder', type=str,
                        help='divided frame numbers', default='checkpoints/mobilenet_plan1_concat')
    parser.add_argument('--continue_train', type=bool,
                        help='continue training', default=False)
    parser.add_argument('--continue_iter', type=int,
                        help='the number of continue training iter', default=10000)
    parser.add_argument('--val', type=bool,
                        help='validation during training', default=False)
    parser.add_argument('--val_term', type=int,
                        help='validation term', default=500)

    arg = parser.parse_args()

    model = Plan1_concat(pretrain=False,
                         width_mult=1,
                         lstm_layers=1,
                         lstm_hidden=256,
                         bidirectional=True,
                         dropout=False,
                         heatmap_size=arg.heatmap_size)
    util.freeze_layers(arg.frozen_layers, model)
    model.train()
    model.cuda()

    dataset = KeypointDB_concat(
        video_path='data/total_videos',
        label_path='fs_labels/train_label.json',
        npy_path='data/all_keypoint_npys',
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

    val_dataset = KeypointDB(
        video_path='data/total_videos',
        label_path='fs_labels/test_label.json',
        npy_path='data/all_keypoint_npys',
        heatmap_size=arg.heatmap_size,
        seq_length=arg.seq_length,
        transform=transforms.Compose([ToTensorForHeatmap(), NormalizeForHeatmap(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        train=False,
        input_size=arg.input_size
    )
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=False
                                 )

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

    if arg.continue_train:
        state_dict = torch.load('{}/swingnet_{}.pth.tar'.format(
            arg.save_folder, arg.continue_iter), map_location=torch.device('cuda'))
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        i = state_dict['iterations']
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])

    start_time = time.time()

    while i < arg.iterations:
        # Training
        for sample in data_loader:
            model.train()
            images, labels, heatmaps = sample['images'].cuda(
            ), sample['labels'].cuda(), sample['heatmaps'].cuda()
            logits = model(images, heatmaps)
            labels = labels.view(arg.batch_size * arg.seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            if i % 100 == 0:
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

            # Validation
            if arg.val:
                if i % arg.val_term == 0:
                    with torch.no_grad():
                        model.eval()
                        correct = []
                        for sample in val_data_loader:
                            # images, labels = sample['images'], sample['labels']
                            images, labels, heatmaps = sample['images'], sample['labels'], sample['heatmaps']
                            heatmaps = torch.unsqueeze(heatmaps, dim=2)
                            # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
                            batch = 0
                            while batch * arg.seq_length < images.shape[1]:
                                if (batch + 1) * arg.seq_length > images.shape[1]:
                                    image_batch = images[:, batch *
                                                         arg.seq_length:, :, :, :]
                                    heatmap_batch = heatmaps[:,
                                                             batch * arg.seq_length:, :, :, :]
                                else:
                                    image_batch = images[:, batch *
                                                         arg.seq_length:(batch + 1) * arg.seq_length, :, :, :]
                                    heatmap_batch = heatmaps[:, batch *
                                                             arg.seq_length:(batch + 1) * arg.seq_length, :, :, :]
                                # logits = model(image_batch.cuda())
                                logits = model(image_batch.cuda(),
                                               heatmap_batch.cuda())
                                if batch == 0:
                                    probs = F.softmax(
                                        logits.data, dim=1).cpu().numpy()
                                else:
                                    probs = np.append(probs, F.softmax(
                                        logits.data, dim=1).cpu().numpy(), 0)
                                batch += 1
                            _, _, _, _, c = util.correct_preds(
                                probs, labels.squeeze())
                            correct.append(c)

                        PCEs = np.mean(correct, axis=0)
                        print(PCEs)
                        PCE = np.mean(correct)
                        PCEwo = np.mean(PCEs[1:7])
                        print('Average PCE: {}'.format(PCE))
                        print('Average PCE w/o AD, F: {}'.format(PCEwo))

            if i == arg.iterations:
                break

    writer.close()
