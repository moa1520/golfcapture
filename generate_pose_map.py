import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from skimage.draw import line_aa, polygon, disk


LIMB_SEQ = [
    [0, 1], [1, 2], [2, 4], [4, 5], [5, 6],
    [2, 7], [7, 8], [8, 9], [9, 20], [2, 3],
    [3, 10], [10, 11], [11, 12], [12, 18], [12, 19],
    [3, 13], [13, 14], [14, 15], [15, 16], [15, 17]
]

COLORS = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [0, 255, 255], [255, 0, 255],
    [255, 255, 0], [0, 255, 255], [255, 0, 255],
    [128, 0, 0], [0, 128, 0], [0, 0, 128],
    [128, 0, 0], [0, 128, 0], [0, 0, 128],
    [128, 128, 0], [0, 128, 128], [128, 0, 128],
    [255, 128, 128], [128, 255, 128], [128, 128, 255],
]


MISSING_VALUE = -1


def draw_pose_from_cords(pose_joints, img_size, radius=6, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(
                pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = disk((joint[0], joint[1]), radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


def produce_ma_mask(kp_array, img_size, point_radius=4):
    from skimage.morphology import dilation, erosion, square
    mask = np.zeros(shape=img_size, dtype=bool)
    limbs = [
        [0, 1], [1, 2], [2, 4], [4, 5], [5, 6],
        [3, 10], [10, 11], [11, 12], [12, 18], [12, 19],
        [3, 13], [13, 14], [14, 15], [15, 16], [15, 17],
        [2, 7], [7, 8], [8, 9], [9, 20], [2, 3]
    ]
    limbs = np.array(limbs)
    for f, t in limbs:
        from_missing = kp_array[f][0] == MISSING_VALUE or kp_array[f][1] == MISSING_VALUE
        to_missing = kp_array[t][0] == MISSING_VALUE or kp_array[t][1] == MISSING_VALUE
        if from_missing or to_missing:
            continue

        norm_vec = kp_array[f] - kp_array[t]
        norm_vec = np.array([-norm_vec[1], norm_vec[0]])
        norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)

        vetexes = np.array([
            kp_array[f] + norm_vec,
            kp_array[f] - norm_vec,
            kp_array[t] - norm_vec,
            kp_array[t] + norm_vec
        ])
        yy, xx = polygon(vetexes[:, 0], vetexes[:, 1], shape=img_size)
        mask[yy, xx] = True

    for i, joint in enumerate(kp_array):
        if kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE:
            continue
        yy, xx = disk((joint[0], joint[1]),
                      radius=point_radius, shape=img_size)
        mask[yy, xx] = True

    mask = dilation(mask, square(5))
    mask = erosion(mask, square(5))
    return mask


def generate_pose_pic(dir_npy, image_size):
    cords = np.load(dir_npy)[:, :2]
    cords = np.concatenate(
        [np.expand_dims(cords[:, 1], -1), np.expand_dims(cords[:, 0], -1)], axis=1)
    colors, mask = draw_pose_from_cords(cords, image_size)
    mmm = produce_ma_mask(cords, image_size).astype(
        float)[..., np.newaxis].repeat(3, axis=-1)
    mmm[mask] = colors[mask]

    return mmm


if __name__ == '__main__':
    npy = '/media/tk/SSD_250G/label/pose/bad_front_swing001_17.npy'
    image_size = (720, 1280)  # image_size (H x W)
    mmm = generate_pose_pic(npy, image_size)

    plt.imshow(mmm)
    plt.show()
