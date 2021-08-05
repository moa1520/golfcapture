import numpy as np

def calc_pckh(pred, gt, L, threshold=0.5):
    ALPHA = 0.6
    dist = np.linalg.norm(pred - gt, axis=1)
    correct = (dist < threshold * ALPHA * L).astype(float)

    return np.sum(correct) / len(pred), correct
