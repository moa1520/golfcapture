import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from eval import ToTensor, Normalize
from model import EventDetector
from model_2plus1_revised import r2plus1d_18

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


class SampleVideo(Dataset):
    def __init__(self, path, input_size, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                      cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return frames
        images = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            img = cv2.resize(img, (new_size[1], new_size[0]))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        cap.release()
        # only for compatibility with transforms
        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--path',
        help='Path to video that you want to test',
        default='total_videos/bad_front_swing1129.mp4')
    args = parser.parse_args()
    seq_length = 64

    print('Preparing video: {}'.format(args.path))

    ds = SampleVideo(args.path, transform=transforms.Compose([ToTensor(),
                                                              Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])]), input_size=512)

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = r2plus1d_18(num_classes=9, seq_length=seq_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dict = torch.load('model_2Plus1/swingnet_200.pth.tar', map_location=device)

    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    print('Testing...')
    for sample in dl:
        images = sample['images']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch *
                                        seq_length:(batch + 1) * seq_length, :, :, :]

            if device == torch.device('cuda'):
                logits = model(image_batch.cuda())
            else:
                logits = model(image_batch)

            # logits -> (64, 9)

            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(
                    logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    # probs -> (264, 9) / test_video.mp4 기준

    events = np.argmax(probs, axis=0)[:-1]

    print('Predicted event frames: {}'.format(events))
    cap = cv2.VideoCapture(args.path)

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

    save_path = 'demo/' + args.path.split('/')[-1].split('.')[0]
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
