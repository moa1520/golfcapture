import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import EventDetector
import numpy as np
import torch.nn.functional as F
from scipy import ndimage
import matplotlib.pyplot as plt


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


class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
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
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
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
        '-p', '--path', help='Path to video that you want to test', default='Tiger_Woods_Original_cut.mp4')
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
                          lstm_hidden=640,
                          bidirectional=True,
                          dropout=False)

    try:
        save_dict = torch.load('models_LSTM_hidden_640/swingnet_2000.pth.tar')
    except FileNotFoundError:
        print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        # print(images.shape) # (1, 264, 3, 160, 160)
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch *
                                     seq_length:(batch + 1) * seq_length, :, :, :]

            logits, activation_maps, pred_class = model(image_batch.cuda())

            output_softmax = F.softmax(logits.data, dim=1).cpu().numpy()

            if batch == 0:
                probs = output_softmax
                # arr_activation_maps = [activation_maps]
                # arr_pred_class = [pred_class]
            else:
                probs = np.append(probs, output_softmax, 0)
                # arr_activation_maps.append(activation_maps)
                # arr_pred_class.append(pred_class)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))
    cap = cv2.VideoCapture(args.path)

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))

    # for i, e in enumerate(events):
    #     # e -> frame number

    #     img2 = ds['images']['images'][e]
    #     img2_in = img2.unsqueeze(0).unsqueeze(0)

    #     k = e // seq_length
    #     l = e - (seq_length * k)

    #     # Activation map
    #     activation_maps = arr_activation_maps[k][l].cpu(
    #     ).detach().numpy()  # (64, 1280, 5, 5) -> (1280, 5, 5)
    #     pred_class = arr_pred_class[k][:, l, :].cpu(
    #     ).detach().numpy().transpose(1, 0)  # (1, 1280) -> (1280, 1)
    #     activation_maps = ndimage.zoom(
    #         activation_maps, (1, 32, 32), order=1)  # (1280, 160, 160)

    #     final_output = np.dot(activation_maps.transpose(
    #         1, 2, 0), pred_class).reshape((160, 160))

    #     img2 = np.transpose(img2.numpy(), (1, 2, 0))

    #     plt.title(event_names[i])
    #     plt.imshow(img2, alpha=0.5)
    #     plt.imshow(final_output, cmap='jet', alpha=0.5)
    #     plt.axis('off')
    #     plt.colorbar()
    #     plt.show()

    # # 결과 화면 출력
    # for i, e in enumerate(events):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, e)
    #     _, img = cap.read()
    #     cv2.putText(img, '{:.3f}'.format(
    #         confidence[i]), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
    #     cv2.imshow(event_names[i], img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
