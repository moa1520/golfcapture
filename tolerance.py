import os
from glob import glob
import numpy as np


def main():
    rates = []
    root = 'D:/tk_unlabeled_videos/custom_training_videos/train'
    for folder in sorted(os.listdir(root)):
        img_folder = os.path.join(root, folder)
        imgs = sorted(os.listdir(img_folder))
        for i in range(len(imgs)):
            imgs[i] = int(imgs[i].split('_')[-1].split('.')[0])
        print(imgs[5] - imgs[0])
        rates.append(imgs[7] - imgs[0])
    rates = np.asarray(rates)
    print(rates.mean())


if __name__ == '__main__':
    main()
