import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import util
from dataloader import CustomGolfDB, Normalize, ToTensor
from model import EventDetector

if __name__ == '__main__':
    # training configuration
    iterations = 2000
    it_save = 100  # save model every 100 iterations
    seq_length = 64
    bs = 22  # batch size
    k = 10  # frozen layers

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    util.freeze_layers(k, model)
    model.train()
    model.cuda()

    dataset = CustomGolfDB(
        video_path='total_videos/',
        label_path='custom_label/train_label.json',
        seq_length=seq_length,
        # transform=transforms.Compose([ToTensor(), Normalize(
        #     [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        transform=transforms.Compose([ToTensor(), Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        train=True
    )
    data_loader = DataLoader(dataset,
                             batch_size=bs,
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

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:
        for sample in tqdm(data_loader):
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            logits = model(images)
            labels = labels.view(bs * seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/swingnet_{}.pth.tar'.format(i))
            if i == iterations:
                break
