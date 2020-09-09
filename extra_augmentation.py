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


def motion_blur(img):
    '''
    Rank 3 numpy array
    Modificado de: https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
    '''
    # Mas grande el filtro, hay mas efecto
    # kernel_size = 7
    kernels = [3, 5]
    kernel_size = random.choice(kernels)
    if random.random() > .5:
        motion_blur_kernel = np.zeros((kernel_size, kernel_size))
        motion_blur_kernel[:, int((kernel_size - 1) / 2)
                           ] = np.ones(kernel_size)
        motion_blur_kernel /= kernel_size
    else:
        motion_blur_kernel = np.zeros((kernel_size, kernel_size))
        motion_blur_kernel[int((kernel_size - 1) / 2),
                           :] = np.ones(kernel_size)
        motion_blur_kernel /= kernel_size
    random_mb = cv2.filter2D(img, -1, motion_blur_kernel)
    return np.expand_dims(random_mb, axis=-1)
