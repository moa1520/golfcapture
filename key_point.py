import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

label_name = {
    0: '0: Top Head', 1: '1: Nose', 2: '2: Neck', 3: '3: Chest',
    4: '4: Right Shoulder', 5: '5: Right Elbow', 6: '6: Right Wrist', 7: '7: Left Shoulder',
    8: '8: Left Elbow', 9: '9: Left Wrist', 10: '10: Right Hip', 11: '11: Right Knee',
    12: '12: Right Ankle', 13: '13: Left Hip', 14: '14: Left Knee', 15: '15: Left Ankle',
    16: '16: Left Big Toe', 17: '17: Left Heel', 18: '18: Right Big Toe', 19: '19: Right Heel',
    20: '20: Golf Club Head'
}


def show_img(image, label):
    plt.imshow(image)
    for i in range(len(label)):
        plt.scatter(label[i][0], label[i][1], label=label_name[i])
    plt.legend()
    plt.show()


def main():
    img_dir = '/media/tk/SSD_250G/image/'
    label_dir = '/media/tk/SSD_250G/label/pose/'

    images = glob(img_dir + '*.png')
    labels = glob(label_dir + '*.npy')

    for image, label in zip(images, labels):
        image = Image.open(image)
        label = np.load(label)
        show_img(image, label)


if __name__ == '__main__':
    main()
