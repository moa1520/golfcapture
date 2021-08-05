import os
from pathlib import Path
from shutil import copyfile
import json
import numpy as np


def organize(directory, out_directory):
    Path(os.path.join(out_directory, 'image')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_directory, 'label')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_directory, 'label', 'bb')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_directory, 'label', 'pose')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_directory, 'label', 'seg')).mkdir(parents=True, exist_ok=True)

    for quality in ['good', 'bad']:
        for view in ['front', 'side']:
            imgPath = os.path.join(directory, quality, view, 'image')
            bbPath = os.path.join(directory, quality, view, 'label', 'bb')
            posePath = os.path.join(directory, quality, view, 'label', 'pose')
            segPath = os.path.join(directory, quality, view, 'label', 'seg')
            for swing in os.listdir(imgPath):
                bb_filepath = os.path.join(bbPath, swing + '_detect.json')
                if os.path.exists(bb_filepath):
                    new_bb_filename = '{}_{}_{}'.format(quality, view, swing + '.json')
                    new_bb_filepath = os.path.join(out_directory, 'label', 'bb', new_bb_filename)
                    os.rename(bb_filepath, new_bb_filepath)
                    print(bb_filepath, ' -> ', new_bb_filepath)

                for filename in os.listdir(os.path.join(imgPath, swing)):
                    if not filename.endswith('png'):
                        continue

                    img_filepath = os.path.join(imgPath, swing, filename)
                    new_img_filename = '{}_{}_{}'.format(quality, view, filename)
                    new_img_filepath = os.path.join(out_directory, 'image', new_img_filename)

                    pose_filepath = os.path.join(posePath, swing, os.path.splitext(filename)[0] + '_pose.json')
                    new_pose_filename = '{}_{}_{}'.format(quality, view, filename.replace('.png', '.json'))
                    new_pose_filepath = os.path.join(out_directory, 'label', 'pose', new_pose_filename)

                    seg_filepath = os.path.join(segPath, swing, os.path.splitext(filename)[0] + '_1.png')
                    new_seg_filename = '{}_{}_{}'.format(quality, view, filename)
                    new_seg_filepath = os.path.join(out_directory, 'label', 'seg', new_seg_filename)

                    if os.path.exists(img_filepath) and os.path.exists(pose_filepath) and os.path.exists(seg_filepath):
                        os.rename(img_filepath, new_img_filepath)
                        print(img_filepath, ' -> ', new_img_filepath)
                        os.rename(pose_filepath, new_pose_filepath)
                        print(pose_filepath, ' -> ', new_pose_filepath)
                        os.rename(seg_filepath, new_seg_filepath)
                        print(seg_filepath, ' -> ', new_seg_filepath)


def organize_add(directory, out_directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.endswith('.png'):
                continue

            imgPath = os.path.join(dirpath, filename)
            swingName = os.path.splitext(filename)[0][:-5]
            posePath = os.path.join(directory, 'pose', swingName, filename.replace('.png', '.json'))
            bbpath = os.path.join(directory, 'bbox', swingName + '.json')

            new_imgPath = os.path.join(out_directory, 'image', 'train', filename)
            new_posePath = os.path.join(out_directory, 'label', 'pose', filename.replace('.png', '.json'))
            new_bbPath = os.path.join(out_directory, 'label', 'bb', swingName + '.json')

            if os.path.exists(imgPath) and os.path.exists(posePath):
                os.rename(imgPath, new_imgPath)
                print(imgPath, ' -> ', new_imgPath)
                os.rename(posePath, new_posePath)
                print(posePath, ' -> ', new_posePath)

                if os.path.exists(bbpath):
                    os.rename(bbpath, new_bbPath)
                    print(bbpath, ' -> ', new_bbPath)


def check(directory):
    dir_img = os.path.join(directory, 'image', 'train')
    dir_pose = os.path.join(directory, 'label', 'pose')
    dir_bb = os.path.join(directory, 'label', 'bb')

    for filename in os.listdir(dir_img):
        imgPath = os.path.join(dir_img, filename)
        posePath = os.path.join(dir_pose, filename.replace('.png', '.npy'))
        bbPath = os.path.join(dir_bb, filename.replace('.png', '.npy'))

        if not (os.path.exists(imgPath) and os.path.exists(posePath) and os.path.exists(bbPath)):
            print(imgPath)


def convert_pose_to_npy(directory):
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue

        posePath = os.path.join(directory, filename)
        try:
            with open(posePath, "r") as json_file:
                pose = np.array(json.load(json_file)[3]['keypoint_index'][0])

            new_posePath = posePath.replace('.json', '.npy')
            np.save(new_posePath, pose)
            print(new_posePath, ' saved successfully!')
        except:
            print(posePath, ' error')
            break


def convert_bbox_to_npy(directory):
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue

        bbPath = os.path.join(directory, filename)
        try:
            with open(bbPath, "r") as json_file:
                data = json.load(json_file)
                data = data[list(data.keys())[0]]

                for i in range(int(len(data) / 2)):
                    info = data[i * 2]['info']
                    bb = data[i * 2 + 1]['bb']

                    bbox = np.empty((0, 4), dtype=int)
                    for key in sorted(list(bb.keys())):
                        if 'bb' not in key:
                            continue

                        if bb[key] == -1:
                            bbox = np.vstack((bbox, [0, 0, 0, 0]))
                        elif len(bb[key]) != 4:
                            # xs = bb[key][0, 2, 4, 6]
                            # ys = bb[key][1, 3, 5, 7]

                            bbox = np.vstack((bbox, bb[key][:4]))
                        else:
                            bbox = np.vstack((bbox, bb[key]))

                    bbox_path = os.path.join(directory, info['image_name'].replace('.png', '.npy'))

                    np.save(bbox_path, bbox)
                    print(bbox_path, ' saved successfully!')
        except:
            print(bbPath, ' error')
            break


def split_dataset(directory):
    filenames = os.listdir(directory)
    np.random.shuffle(filenames)

    data = {
        'train': filenames[:int(len(filenames) * 0.8)],
        'test': filenames[int(len(filenames) * 0.8):]
    }

    for key in data.keys():
        for filename in data[key]:
            filepath = os.path.join(directory, filename)
            new_filepath = os.path.join('/media/lbs/HDD/VC_Dataset', key, filename)

            os.rename(filepath, new_filepath)
            print(filepath, '->', new_filepath)

if __name__ == '__main__':
    # organize('/media/lbs/HDD/golfDB_정리/image_data', '/media/lbs/HDD/VC_Dataset')
    # organize_add('/media/lbs/HDD/VC_Dataset/add/', '/media/lbs/HDD/VC_Dataset')
    # convert_pose_to_npy('/media/lbs/HDD/VC_Dataset/label/pose')
    # convert_bbox_to_npy('/media/lbs/HDD/VC_Dataset/label/bb')
    check('/media/lbs/HDD/VC_Dataset')
    # split_dataset('/media/lbs/HDD/VC_Dataset/image')

