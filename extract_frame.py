import os
import cv2
import glob
import matplotlib.pyplot as plt


def saving_frames(file_path, events):
    file_path = file_path.replace('\\', '/')
    file_name = file_path.split('/')[-1].split('.')[0]
    print('file_name', file_name)
    cap = cv2.VideoCapture(file_path)
    for event in events:
        for i in range(event-2, event+3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, image = cap.read()
            if image is None:
                continue
            frame_number = int(cap.get(1))
            saving_path = os.path.join('extracted_frames/', file_name)
            if not os.path.isdir('extracted_frames'):
                os.mkdir('extracted_frames')
            if not os.path.isdir(saving_path):
                os.mkdir(saving_path)
            cv2.imwrite(
                saving_path + '/{}_{}.png'.format(file_name, frame_number), image)
    cap.release()


def making_frame():
    main_dir = '/media/tk/SSD_250G/tk_unlabeled_videos/f_video'
    for file_name in os.listdir(main_dir):
        save_dir = 'all_frames/' + file_name.split('.')[0]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        vidcap = cv2.VideoCapture(os.path.join(main_dir, file_name))

        frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 186)
        ret, image = vidcap.read()
        plt.imshow(image)
        plt.show()
        vidcap.release()

        # count = 0
        # while vidcap.isOpened():
        #     ret, image = vidcap.read()
        #     if not ret:
        #         break
        #     if int(vidcap.get(1)) % 1 == 0:
        #         print('Saved from number: ' + str(int(vidcap.get(1))))
        #         cv2.imwrite(save_dir + '/{}_{}.png'.format(, count), image)
        #         count += 1
        # vidcap.release()


if __name__ == '__main__':
    making_frame()
