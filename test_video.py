import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import SampleVideo
from eval import ToTensor, Normalize
from models.model import EventDetector
from util import get_probs

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--path', help='Path to video that you want to test', default='test_video.mp4')
    parser.add_argument('-s', '--seq-length', type=int,
                        help='Number of frames to use per forward pass', default=64)
    args = parser.parse_args()
    seq_length = args.seq_length

    print('Preparing video: {}'.format(args.path))

    ds = SampleVideo(args.path, transform=transforms.Compose([ToTensor(),
                                                              Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        save_dict = torch.load(
            'models_first/swingnet_2000.pth.tar', map_location=device)
    except FileNotFoundError:
        print(
            "Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    print('Testing...')

    probs = get_probs(dl, seq_length, model)

    # probs -> (264, 9) / test_video.mp4 기준

    '''
    이 부분 변경
    '''
    events = np.argmax(probs, axis=0)[:-1]
    # events = []
    # for i in range(8):
    #     if i == 0:
    #         events.append(np.argmax(probs, axis=0)[i])
    #     else:
    #         events.append(
    #             np.argmax(probs[events[-1]+1:], axis=0)[i] + events[-1]+1)
    ''''''

    print('Predicted event frames: {}'.format(events))
    cap = cv2.VideoCapture(args.path)

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))

    # 결과 화면 출력
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        cv2.putText(img, '{:.3f}'.format(
            confidence[i]), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
        cv2.imshow(event_names[i], img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
