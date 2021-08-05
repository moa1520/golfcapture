import os
from pathlib import Path

import cv2
import numpy as np


def video2images(video_path, out_dir=None, zero_fill=8):
    cap = cv2.VideoCapture(video_path)
    isOpened = cap.isOpened()
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    i = 0
    assert isOpened, "Can't find video"

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for index in range(video_length):
            (flag, data) = cap.read()
            filename = "{}.jpg".format(str(index).zfill(zero_fill))  # start from zero
            filepath = os.path.join(out_dir, filename)
            if flag:
                cv2.imwrite(filepath, data, [cv2.IMWRITE_JPEG_QUALITY, 100])


def image2video(image_dir, out_dir, name, fps=60):
    image_path_list = []
    for filename in sorted(os.listdir(image_dir)):
        if not filename.endswith('.jpg'):
            continue
        image_path_list.append(os.path.join(image_dir, filename))
    image_path_list.sort()
    temp = cv2.imread(image_path_list[0])
    size = (temp.shape[1], temp.shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(os.path.join(out_dir, name + '.mp4'), fourcc, fps, size)
    for image_path in image_path_list:
        if image_path.endswith(".jpg"):
            image_data_temp = cv2.imread(image_path)
            video.write(image_data_temp)
    print("Video done！")


if __name__ == '__main__':
    # video2images('/media/lbs/HDD/VC_Dataset/demo/src/video_data/good/side/gh_lee_2.mp4', '/media/lbs/HDD/VC_Dataset/demo/src/image_data/good_side_gh_lee_2')
    # dir_data = '/media/lbs/HDD/VC_Dataset/demo/src/video_data'
    # dir_img = '/media/lbs/HDD/VC_Dataset/demo/src/image_data'
    # for dirpath, dirnames, filenames in os.walk(dir_data):
    #     for filename in filenames:
    #         if not filename.endswith('.mp4'):
    #             continue
    #
    #         video_path = os.path.join(dirpath, filename)
    #         quality = video_path.split('/')[-3]
    #         view = video_path.split('/')[-2]
    #         out_dir = os.path.join(dir_img, quality + '_' + view + '_' + os.path.splitext(filename)[0])
    #         video2images(video_path, out_dir)
    #         print(video_path, 'converted successfully!')

    dir_data = '/media/lbs/HDD/VC_Dataset/demo/result/image'
    dir_video = '/media/lbs/HDD/VC_Dataset/demo/result/video'

    for video in os.listdir(dir_data):
        if video not in ['good_front_swing001', 'good_front_swing002', 'good_front_swing003', 'good_front_swing004',
                         'good_side_swing001', 'good_side_swing002', 'good_side_swing003', 'good_side_swing004',
                         'good_side_gh_lee_1', 'good_side_gh_lee_2']:
            continue
        image_dir = os.path.join(dir_data, video)

        image2video(image_dir, dir_video, video, fps=30)
        print(video, 'converted successfully!')