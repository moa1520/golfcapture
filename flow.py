import cv2
import numpy as np


def dense_optical_flow(method, video_path, params=[], to_gray=False):
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break

        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        flow = method(old_frame, new_frame, None, *params)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        old_frame = new_frame


if __name__ == '__main__':
    method = cv2.calcOpticalFlowFarneback
    params = [0.5, 3, 15, 3, 5, 1.2, 0]
    frames = dense_optical_flow(
        method, 'test_video.mp4', params=params, to_gray=True)
