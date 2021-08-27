import os
import numpy as np
from glob import glob
import pandas as pd
import json
from PIL import Image


def main():
    labels = {
        'video_name': [],
        'width': [],
        'height': [],
        'events': []
    }
    root = '/media/tk/SSD_250G/8_frames/test_front'
    dirs = glob(root + '/*')
    for dir in dirs:
        print(dir)
        video_name = dir.split('/')[-1]
        img_names = os.listdir(dir)
        if len(img_names) != 8:
            print('사진이 8개가 아님')
            print(img_names)
            continue
        frames = []
        for img_name in img_names:
            frames.append(int(img_name.split('.')[0].split('_')[-1]))
        labels['video_name'].append(video_name)
        labels['events'].append(sorted(frames))

        files_path = glob(dir + '/*.png')
        img = np.asarray(Image.open(files_path[0]))
        height = img.shape[0]
        width = img.shape[1]
        labels['width'].append(width)
        labels['height'].append(height)

    if not os.path.isdir('front_labels'):
        os.makedirs('front_labels')
    with open('front_labels/test.json', 'w') as outfile:
        json.dump(labels, outfile)


if __name__ == '__main__':
    main()
