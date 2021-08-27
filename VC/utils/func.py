import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import torch
from scipy.stats import multivariate_normal
from ..configs.constants import *


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    for step in schedule:
        if epoch >= step:
            lr *= gamma

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_sigma(epoch, sigma, schedule, gamma=0.5):
    """Sets the learning rate to the initial LR decayed by schedule"""
    for step in schedule:
        if epoch >= step:
            sigma *= gamma

    return sigma


def plot_data(imgPath, posePath):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose = np.load(posePath)

    print(img.shape)
    print(len(pose))
    for i, [X, Y, _] in enumerate(pose):
        cv2.circle(img, (X, Y), 3, (255, 0, 0), -1)
        # cv2.putText(img, str(i), (X, Y), fontFace= cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0), thickness=3)

    plt.imshow(img)
    plt.show()


def get_square_img(img):
    square_size = max(img.shape[0], img.shape[1])
    square_img = np.zeros((square_size, square_size, 3), np.uint8)
    y_pad = (square_size - img.shape[0]) // 2
    x_pad = (square_size - img.shape[1]) // 2
    square_img[y_pad: y_pad + img.shape[0], x_pad: x_pad + img.shape[1], :] = img

    return [square_img, x_pad, y_pad]


def convert_relative_pose(rel_pose):
    pose = np.zeros((len(JOINTMAP), 2), dtype=float)
    for child, parent in JOINTMAP:
        if parent == -1:
            pose[child] = rel_pose[child]
        else:
            pose[child] = rel_pose[child] + pose[parent]

    return pose


def get_relative_pose(pose):
    rel_pose = np.zeros((len(JOINTMAP), 2), dtype=float)
    for child, parent in JOINTMAP:
        if parent == -1:
            rel_pose[child] = pose[child]
        else:
            rel_pose[child] = pose[child] - pose[parent]

    return pose


def get_relative_pose_torch(pose):
    rel_pose = torch.zeros_like(pose)
    for child, parent in JOINTMAP:
        if parent == -1:
            rel_pose[:, child, :] = pose[:, child, :]
        else:
            rel_pose[:, child, :] = pose[:, child, :] - pose[:, parent, :]

    return pose


def get_data(imgPath, size, sigma, posePath=None, bbPath=None, blur=False):
    data = dict()
    img = cv2.imread(imgPath)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    [square_img, x_pad, y_pad] = get_square_img(img)
    resized_img = cv2.resize(square_img, (size, size))
    scale = size / square_img.shape[0]
    data['image'] = resized_img

    if posePath is not None:
        pose_data = np.load(posePath).astype(float)
        pose = pose_data[:, :2]
        vis = pose_data[:, 2]
        pose += np.array([x_pad, y_pad])
        norm_pose = pose / square_img.shape[0]
        scaled_pose = (pose * scale / 4).astype(int)
        heatmaps = generate_heatmaps(scaled_pose, size // 4, sigma)

        data['pose'] = norm_pose
        data['vis'] = vis
        data['heatmaps'] = heatmaps

        if blur and np.random.rand() > 0.8:
            resized_pose = (pose * scale).astype(int)
            blurred_img = blur_club(resized_img, np.random.randint(2) * 2 + 1, (resized_pose[6] + resized_pose[9]) / 2, resized_pose[20])
            data['image'] = blurred_img

    if bbPath is not None:
        bb_data = np.load(bbPath).astype(float)
        bbox = bb_data[2, :]
        bbox += np.array([x_pad, y_pad, 0, 0])
        bbox *= scale
        data['bbox'] = bbox
        # bbox = bbox.astype(int)
        # bb_resized_img = resized_img.copy()
        # bb_resized_img = cv2.rectangle(bb_resized_img, (bbox[0], bbox[1]), (bbox[0] + bbox[3], bbox[1] + bbox[2]), (0, 0, 255), 10)
        # plt.imshow(bb_resized_img)
        # plt.show()

    # # plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
    # # plt.savefig('/home/lbs/Downloads/fig0.png')
    # plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    # plt.savefig('/home/lbs/Downloads/fig1.png')
    # plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    # plt.scatter(scaled_pose[:,0] * 4, scaled_pose[:,1] * 4, c='r', s=5)
    # plt.savefig('/home/lbs/Downloads/fig2.png')
    # # plt.show()
    # plt.imshow(np.sum(heatmaps, axis=0))
    # plt.savefig('/home/lbs/Downloads/fig3.png')
    # # plt.show()
    # club = np.uint8(np.transpose(np.tile(cv2.resize(heatmaps[20], (size, size)), (3, 1, 1)), (1, 2, 0)) * 255.0)
    # plt.imshow(np.uint8(resized_img * 0.3 + club * 0.7))
    # plt.savefig('/home/lbs/Downloads/fig4.png')
    # # plt.show()

    return data


def points_to_gaussian_heatmap(pose, size, sigma):
    heatmaps = np.zeros((pose.shape[0], size, size))
    scale = size / 512
    s = np.eye(2)*sigma*scale
    x = np.arange(0, size)
    y = np.arange(0, size)
    xx, yy = np.meshgrid(x,y)
    xxyy = np.stack([xx.ravel(), yy.ravel()]).T
    for i, [x, y] in enumerate(pose):
        if i == 20:
            g = multivariate_normal(mean=(x,y), cov=s * 3)
            zz = g.pdf(xxyy)
            heatmaps[i, :, :] = zz.reshape((size,size))

            start = (pose[6] + pose[9]) / 2
            end = pose[20]
            for xx, yy in np.linspace(start, end, endpoint=True, num=50):
                g = multivariate_normal(mean=(xx,yy), cov=s)
                zz = g.pdf(xxyy)
                heatmaps[i, :, :] += zz.reshape((size,size)) / 25
        else:
            g = multivariate_normal(mean=(x,y), cov=s)
            zz = g.pdf(xxyy)
            heatmaps[i, :, :] = zz.reshape((size,size))

        heatmaps[i, :, :] = (heatmaps[i, :, :] - np.min(heatmaps[i, :, :])) / (np.max(heatmaps[i, :, :]) - np.min(heatmaps[i, :, :]))

    return heatmaps


def generate_heatmaps(pose, size, sigma):
    sigma = int(sigma * size / 128)
    heatmaps = np.zeros((pose.shape[0], size + 2 * sigma, size + 2 * sigma))
    win_size = 2 * sigma + 1

    x, y = np.meshgrid(np.linspace(-sigma, sigma, num=win_size, endpoint=True), np.linspace(-sigma, sigma, num=win_size, endpoint=True))
    dst = np.sqrt(x * x + y * y)
    mu = 0.000
    gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))
    club_gauss = gauss.copy() * 0.8

    for i, [X, Y] in enumerate(pose):
        X, Y = int(X), int(Y)
        if X < 0 or X >= size or Y < 0 or Y >= size:
            continue

        if i == 20:
            start = (pose[6] + pose[9]) / 2
            # start = pose[6]
            end = pose[20]

            for xx, yy in np.linspace(start, end, endpoint=True, num=50):
                heatmaps[i, int(yy): int(yy) + win_size, int(xx): int(xx) + win_size] = club_gauss

        heatmaps[i, Y: Y + win_size, X: X + win_size] = gauss

    heatmaps = heatmaps[:, sigma:-sigma, sigma:-sigma]

    return heatmaps


def blur_club(img, sigma, start, end, num=50):
    size = img.shape[0]
    aug_img = cv2.copyMakeBorder(img, sigma, sigma, sigma, sigma, cv2.BORDER_CONSTANT)
    res_img = aug_img.copy()
    win_size = 2 * sigma + 1

    for x, y in np.linspace(start, end, endpoint=True, num=num):
        x, y = int(x), int(y)
        if x < 0 or x >= size or y < 0 or y >= size:
            continue

        res_img[y: y + win_size, x: x + win_size] = cv2.blur(aug_img[y: y + win_size, x: x + win_size], (win_size, win_size))

    return res_img[sigma: -sigma, sigma: -sigma]


def reshape_multiview_tensors(image_tensor, heatmap_tensor):
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    heatmap_tensor = heatmap_tensor.view(
        heatmap_tensor.shape[0] * heatmap_tensor.shape[1],
        heatmap_tensor.shape[2],
        heatmap_tensor.shape[3],
        heatmap_tensor.shape[4]
    )

    return image_tensor, heatmap_tensor


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def save_result_images(img, pose, out_dir, name='', heatmaps=None, label=None):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    cv2.imwrite(os.path.join(out_dir, '{}img.png'.format(name)), img)
    img_pose = img.copy()
    pose = pose * img.shape[0]
    for i in range(1, len(JOINTMAP)):
        child = tuple(pose[JOINTMAP[i][0]].astype(int))
        # if i == len(JOINTMAP) - 1:
        #     parent = tuple(((pose[6] + pose[9])/2).astype(int))
        # else:
        parent = tuple(pose[JOINTMAP[i][1]].astype(int))
        if i == len(JOINTMAP) - 1:
            cv2.circle(img_pose, child, 2, (0, 0, 255), -1)
        else:
            color = (0, 255 * (len(JOINTMAP) - i) / len(JOINTMAP), 255 * i / len(JOINTMAP))
            cv2.line(img_pose, child, parent, color, 2)

    if label is not None:
        for i in range(len(pose)):
            coord = pose[i]
            cv2.circle(img_pose, coord.astype(int), 3, (0, 255 * label[i] if label is not None else 0, 255), -1)

    cv2.imwrite(os.path.join(out_dir, '{}img_pose.png'.format(name)), img_pose)

    if heatmaps is not None:
        heatmap = np.sum(heatmaps, axis=0)
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        colored_heatmap = cv2.resize(colored_heatmap, (img.shape[0], img.shape[1]))
        cv2.imwrite(os.path.join(out_dir, '{}heatmap.png'.format(name)), colored_heatmap)

        img_heatmap = img_pose * 0.7 + colored_heatmap * 0.3
        cv2.imwrite(os.path.join(out_dir, '{}img_heatmap.png'.format(name)), img_heatmap)


def save_demo_images(batch_images, batch_pose, dir_out, idx):
    for i, img in enumerate(batch_images):
        img_pose = img.copy()
        img_pose, x_pad, y_pad = get_square_img(img_pose)
        pose = batch_pose[i] * img_pose.shape[0]
        for j in range(1, len(JOINTMAP)):
            child = tuple(pose[JOINTMAP[j][0]].astype(int))
            if j == len(JOINTMAP) - 1:
                parent = tuple(((pose[6] + pose[9])/2).astype(int))
            else:
                parent = tuple(pose[JOINTMAP[j][1]].astype(int))
            color = (0, 64 + 192 * (len(JOINTMAP) - j) / (len(JOINTMAP) - 1), 64 + 192 * j / (len(JOINTMAP) - 1))
            if j == len(JOINTMAP) - 1:
                cv2.circle(img_pose, child, 2, (0, 0, 255), -1)
            else:
                cv2.line(img_pose, child, parent, color, 2)

        if x_pad:
            img_pose = img_pose[:, x_pad:-x_pad]
        else:
            img_pose = img_pose[y_pad:-y_pad, :]
        imgPath = os.path.join(dir_out, '{}.jpg'.format(str(idx + i).zfill(6)))
        cv2.imwrite(imgPath, img_pose)
        # print(imgPath, ' saved successfully!')


if __name__ == '__main__':
    directory = '/media/lbs/HDD/VC_Dataset'
    filename = 'good_front_swing001_001'
    imgPath = os.path.join(directory, 'image', 'train', filename + '.png')
    posePath = os.path.join(directory, 'label', 'pose', filename + '.npy')
    bbPath = os.path.join(directory, 'label', 'bb', filename + '.npy')
    # plot_data(imgPath, posePath)
    get_data(imgPath, 1024, 2, posePath=posePath, bbPath=bbPath, blur=True)