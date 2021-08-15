import cv2
import numpy as np


def get_flows(imgs):
    B, N, C, H, W = imgs.size()
    imgs = imgs.permute((0, 1, 3, 4, 2)).detach().cpu().numpy()
    for b in range(1):
        hsv = np.zeros_like(imgs[b, 0, :, :, :])
        hsv[..., 1] = 255
        for n in range(N-1):
            frame1 = imgs[b, n, :, :, :]
            prev = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

            frame2 = imgs[b, n+1, :, :, :]

            nexts = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev, nexts, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            cv2.imshow('frame2', rgb)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # elif k == ord('s'):
            #     cv2.imwrite('opticalfb.png', frame2)
            #     cv2.imwrite('opticalhsv.png', rgb)
            # prev = nexts
