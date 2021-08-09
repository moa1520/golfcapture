import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import util
from VC.models.ScalePoseNet import ScalePoseNet
from VC.options import BaseOptions
from dataloader import CustomGolfDB, Normalize, ToTensor
from model_skeleton import EventDetector

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU 1 to use

if __name__ == '__main__':
    # training configuration
    iterations = 10000
    it_save = 100  # save model every 100 iterations
    seq_length = 16
    bs = 1  # batch size
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
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    device = torch.device('cuda')

    opt = BaseOptions().parse()
    scale_posenet = ScalePoseNet(opt)
    state_dict_posenet = torch.load('VC/models/ScalePoseNet_latest', map_location=device)
    scale_posenet.load_state_dict(state_dict_posenet)
    scale_posenet.cuda()
    scale_posenet.eval()

    dataset = CustomGolfDB(
        video_path='total_videos/',
        label_path='custom_label/train_label.json',
        seq_length=seq_length,
        # transform=transforms.Compose([ToTensor(), Normalize(
        #     [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        transform=transforms.Compose([ToTensor(), Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        train=True,
        input_size=512
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

    save_folder = 'models_skeleton'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    i = 0
    while i < iterations:
        for sample in tqdm(data_loader):
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            batch_size, timesteps, C, H, W = images.size()
            p_in = images.view(batch_size * timesteps, C, H, W)

            for k in range(batch_size * timesteps):
                heatmaps = scale_posenet(p_in[k, :].unsqueeze(0))
                if k == 0:
                    p_outs = heatmaps.mean(3).mean(2)
                else:
                    p_outs = torch.cat([p_outs, heatmaps.mean(3).mean(2)], dim=0)

            logits = model(images, p_outs)
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
                if isinstance(model, torch.nn.DataParallel):
                    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                'model_state_dict': model.module.state_dict()},
                               save_folder + '/swingnet_{}.pth.tar'.format(i))
                else:
                    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                                'model_state_dict': model.state_dict()}, save_folder + '/swingnet_{}.pth.tar'.format(i))
            if i == iterations:
                break
