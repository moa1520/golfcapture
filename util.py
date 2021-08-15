import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.utils as utils

from visualization import visualize_attn_softmax


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def show_attention(images, c1, c2, c3):
    I_test = utils.make_grid(images, nrow=6, normalize=True, scale_each=True)
    attn1 = visualize_attn_softmax(I_test, c1, up_factor=16, nrow=6)
    attn2 = visualize_attn_softmax(I_test, c2, up_factor=64, nrow=6)
    attn3 = visualize_attn_softmax(I_test, c3, up_factor=256, nrow=6)
    attn1 = attn1.permute((1, 2, 0)).detach().cpu().numpy()
    attn2 = attn2.permute((1, 2, 0)).detach().cpu().numpy()
    attn3 = attn3.permute((1, 2, 0)).detach().cpu().numpy()
    plt.imshow(attn1)
    plt.show()
    plt.imshow(attn2)
    plt.show()
    plt.imshow(attn3)
    plt.show()


def get_probs(dl, seq_length, model):
    for sample in dl:
        images = sample['images']
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch *
                                        seq_length:(batch + 1) * seq_length, :, :, :]

            logits, c1, c2, c3 = model(image_batch.cuda())

            # I_test = image_batch[0, :, :, :, :]
            # I_test = utils.make_grid(I_test, nrow=6, normalize=True, scale_each=True)
            # attn1 = visualize_attn_softmax(I_test, c1, up_factor=16, nrow=6)
            # attn2 = visualize_attn_softmax(I_test, c2, up_factor=64, nrow=6)
            # attn3 = visualize_attn_softmax(I_test, c3, up_factor=256, nrow=6)
            # attn1 = attn1.permute((1, 2, 0)).detach().cpu().numpy()
            # attn2 = attn2.permute((1, 2, 0)).detach().cpu().numpy()
            # attn3 = attn3.permute((1, 2, 0)).detach().cpu().numpy()
            # plt.imshow(attn1)
            # plt.show()
            # plt.imshow(attn2)
            # plt.show()
            # plt.imshow(attn3)
            # plt.show()

            # logits -> (64, 9)

            if batch == 0:
                probs = F.softmax(logits.data, dim=0).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(
                    logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    return probs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def correct_preds(probs, labels, tol=-1):
    """
    Gets correct events in full-length sequence using tolerance based on number of frames from address to impact.
    Used during validation only.
    :param probs: (sequence_length, 9)
    :param labels: (sequence_length,)
    :return: array indicating correct events in predicted sequence (8,)
    """

    events = np.where(labels < 8)[0]
    preds = np.zeros(len(events))
    if tol == -1:
        tol = int(max(np.round((events[5] - events[1]) / 80), 3))
    print('tolerance: ', tol)
    for i in range(len(events)):
        preds[i] = np.argsort(probs[:, i])[-1]
    deltas = np.abs(events - preds)
    correct = (deltas <= tol).astype(np.uint8)
    return events, preds, deltas, tol, correct


def freeze_layers(num_freeze, net):
    # print("Freezing {:2d} layers".format(num_freeze))
    i = 1
    for child in net.children():
        if i == 1:
            j = 1
            for child_child in child.children():
                if j <= num_freeze:
                    for param in child_child.parameters():
                        param.requires_grad = False
                j += 1
        i += 1
