import os
import time

import torch
import torch.nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import util
from dataloader import ToTensor, Normalize, TwoPlusOneDB
from model_2plus1_revised import r2plus1d_18


def main():
    batch_size = 4
    seq_length = 64
    input_size = 160
    iterations = 10000
    it_save = 100
    save_folder = 'models_2Plus1'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Model
    model = r2plus1d_18(progress=True, num_classes=9, seq_length=seq_length)
    model.cuda()

    # Dataloader
    dataset = TwoPlusOneDB(
        video_path='total_videos/',
        label_path='custom_label/train_label.json',
        seq_length=seq_length,
        transform=transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        train=True,
        input_size=input_size
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    weights = torch.FloatTensor([1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    losses = util.AverageMeter()
    start_time = time.time()

    i = 0

    pretrained = False
    if pretrained:
        state_dict = torch.load('model_2Plus1/swingnet_1900.pth.tar')
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        i = state_dict['iterations']
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    while i < iterations:
        model.train()
        for sample in data_loader:
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            bs, step, C, H, W = images.size()
            images = images.view(bs, C, step, H, W)
            out = model(images)
            labels = labels.view(bs * step)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            print('time : {}min'.format((time.time() - start_time) // 60))
            i += 1
            if i % it_save == 0:
                if isinstance(model, torch.nn.DataParallel):
                    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                'model_state_dict': model.module.state_dict(),
                                'iterations': i},
                               save_folder + '/swingnet_{}.pth.tar'.format(i))
                else:
                    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                'model_state_dict': model.state_dict(),
                                'iterations': i}, save_folder + '/swingnet_{}.pth.tar'.format(i))
            if i == iterations:
                break


if __name__ == '__main__':
    main()
