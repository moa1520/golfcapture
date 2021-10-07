import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import (Normalize, Normalize_pose, SampleVideo,
                        SampleVideo_repeat, ToTensor, ToTensor_pose)
from models.model_resnet_heatmap import Plan1_repeat, Plan1_concat
from util import UnNormalize, get_probs
from VC.configs.options import BaseOptions
from VC.models.ScalePoseNet import ScalePoseNet
from VC.utils.func import generate_heatmaps

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU 1 to use

event_names = {
    0: 'Address',
    1: 'Take-back',
    2: 'Backswing',
    3: 'Top',
    4: 'Downswing',
    5: 'Impact',
    6: 'Follow-through',
    7: 'Finish',
    8: 'No-action'
}

if __name__ == '__main__':
    unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    path = 'data/total_videos/bad_front_swing0423.mp4'
    save_dict_path = 'checkpoints/plan1_concat/swingnet_18000(sizeup_best).pth.tar'
    save_dict_pose_path = 'VC/models/ScalePoseNet_latest'
    seq_length = 64
    heatmap_size = 56
    input_size = 224
    posenet_input_size = 512
    opt = BaseOptions().parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Preparing video: {}'.format(path))

    ds = SampleVideo_repeat(path, transform=transforms.Compose([ToTensor_pose(),
                                                                Normalize_pose([0.485, 0.456, 0.406],
                                                                               [0.229, 0.224, 0.225])]), input_size=input_size, posenet_input_size=posenet_input_size)

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model_detection = Plan1_concat(pretrain=True, width_mult=1, lstm_layers=2, lstm_hidden=512, bidirectional=True,
                                   dropout=False, heatmap_size=heatmap_size).to(device=device)
    model_posenet = ScalePoseNet(opt).to(device=device)
    save_dict_detection = torch.load(save_dict_path, map_location=device)
    save_dict_pose = torch.load(save_dict_pose_path, map_location=device)

    print('Using device:', device)
    model_detection.load_state_dict(save_dict_detection['model_state_dict'])
    model_detection.eval()
    model_posenet.load_state_dict(save_dict_pose)
    model_posenet.eval()
    print("Loaded model weights")

    print('Testing...')
    for sample in dl:
        images, images_pose = sample['images'].cuda(
        ), sample['images_pose'].cuda()  # 1 x n x 3 x 224 x 224
        for i in range(images.shape[1]):
            img = images[:, i, :, :, :]  # 1 x 3 x 224 x 224
            img_pose = images_pose[:, i, :, :, :]
            # pred_pose, _, _, _ = model_posenet.forward(img_pose)
            if i == 0:
                pose = model_posenet.forward(img_pose).unsqueeze(0)
            else:
                pose = torch.cat(
                    [pose, model_posenet.forward(img_pose).unsqueeze(0)], dim=0)
            # pose.append(model_posenet.forward(img_pose))

            # Point to heatmap
            # pose.append(generate_heatmaps(pred_pose[0].cpu().detach(
            # ).numpy() * heatmap_size, heatmap_size, sigma=1))
        heatmaps = pose.unsqueeze(0)
        # heatmaps = torch.Tensor(pose).sum(dim=1).unsqueeze(
        #     0).unsqueeze(2)  # 1 x n x 1 x 56 x 56
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
                heatmap_batch = heatmaps[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch *
                                     seq_length:(batch + 1) * seq_length, :, :, :]
                heatmap_batch = heatmaps[:, batch *
                                         seq_length:(batch + 1) * seq_length, :, :, :]

            detection_input = torch.cat(
                [image_batch.flatten().cuda(), heatmap_batch.flatten().cuda()])

            if batch == 0:
                probs = model_detection(detection_input).detach().cpu().numpy()
                # probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, model_detection(
                    detection_input).detach().cpu().numpy(), 0)
                # probs = np.append(probs, F.softmax(
                #     logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]

    print('Predicted event frames: {}'.format(events))
    cap = cv2.VideoCapture(path)

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))

    # Softmax plt 그래프
    for i in range(8):
        plt.plot(list(range(probs.shape[0])),
                 probs[:, i], label=event_names[i])
    plt.legend()
    plt.xlabel('Frame number')
    plt.ylabel('Confidence')
    plt.tight_layout()
    plt.show()

    save_path = 'demo/' + path.split('/')[-1].split('.')[0]
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # 결과 화면 출력
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.putText(img, '{:.3f}'.format(
            confidence[i]), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
        cv2.putText(img, event_names[i], (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), thickness=2)
        cv2.putText(img, '{}th frame'.format(e), (20, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), thickness=2)
        cv2.imwrite(os.path.join(save_path, '{}_'.format(
            i) + event_names[i] + '.png'), img)

    # heatmap = heatmaps[0][e].sum(0).detach().cpu().numpy()
    # plt.imsave(os.path.join(save_path, 'heatmap_{}_'.format(
    #     i) + event_names[i] + '.png'), heatmap, cmap='jet')

    # cv2.imwrite(os.path.join(save_path, '{}_'.format(
    #     i) + event_names[i] + '_heatmap.png'), heatmaps[0][e].sum(0).detach().cpu().numpy())
    # cv2.imshow(event_names[i], img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
