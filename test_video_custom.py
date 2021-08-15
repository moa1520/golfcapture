import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import SampleVideo
from eval import ToTensor, Normalize
from model_resnet import EventDetector
from util import get_probs

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
    path = 'total_videos/bad_front_swing1243.mp4'
    save_dict_path = 'models_attn/swingnet_10000.pth.tar'
    seq_length = 32
    input_size = 512

    print('Preparing video: {}'.format(path))

    ds = SampleVideo(path, transform=transforms.Compose([ToTensor(),
                                                         Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])]), input_size=input_size)

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(pretrain=True, width_mult=1, lstm_layers=1, lstm_hidden=256, bidirectional=True,
                          dropout=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dict = torch.load(save_dict_path, map_location=device)

    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    print('Testing...')

    probs = get_probs(dl, seq_length, model)

    # probs -> (264, 9) / test_video.mp4 기준

    events = np.argmax(probs, axis=0)[:-1]

    print('Predicted event frames: {}'.format(events))
    cap = cv2.VideoCapture(path)

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))

    # Softmax plt 그래프
    # for i in range(9):
    #     plt.plot(list(range(probs.shape[0])),
    #              probs[:, i], label=event_names[i])
    # plt.legend()
    # plt.xlabel('Frame number')
    # plt.ylabel('Confidence')
    # plt.tight_layout()
    # plt.show()

    save_path = 'demo/' + path.split('/')[-1].split('.')[0]
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # 결과 화면 출력
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        cv2.putText(img, '{:.3f}'.format(
            confidence[i]), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
        cv2.imwrite(os.path.join(save_path, '{}_'.format(i) + event_names[i] + '.png'), img)
        # cv2.imshow(event_names[i], img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
