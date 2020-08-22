import numpy as np
import random
import cv2


def blur(img):
    '''
    Rank 3 numpy array
    '''
    k = random.choice([3, 5, 7, 9])
    return np.expand_dims(cv2.GaussianBlur(img, (k, k), 0), -1)


def cut_out(img):
    '''
    Modificada de la implementacion original
    https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    '''
    h, w, _ = img.shape
    mask = np.ones((h, w), np.float32)
    n_holes = 12
    # length = 7
    # Make rectangles (width > height)
    w_lens = [6, 7, 8]
    h_lens = [1, 2, 3]
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        w_len = random.choice(w_lens)
        h_len = random.choice(h_lens)
        y1 = np.clip(y - h_len // 2, 0, h)
        y2 = np.clip(y + h_len // 2, 0, h)
        x1 = np.clip(x - w_len // 2, 0, w)
        x2 = np.clip(x + w_len // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
    mask = np.expand_dims(mask, axis=-1)
    img = img * mask
    return img


# def cut_out_blur(img):
#     if random.random() > .5:
#         return cut_out(img)
#     else:
#         return blur(img)
