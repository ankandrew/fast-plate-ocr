import numpy as np
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataAugmentation:
    def __init__(self, do_blur=False, do_cut_out=False) -> None:
        self.do_blur = do_blur
        self.do_cut_out = do_cut_out
        # Cutout
        # Motion Blur

    def data_aug(self):
        if self.do_blur and self.do_cut_out:
            def pf(img):
                '''
                Aplicar 33.3% de las veces cut_out, blur
                y la imagen aumentada normalmente
                '''
                rand = np.random.rand()
                if rand < 1 / 3:
                    return self.cut_out(img)
                elif rand > 1 / 3 and rand < 2 / 3:
                    return self.motion_blur(img)
                else:
                    return img
        elif self.do_blur:
            def pf(img):
                if np.random.rand() > .5:
                    return self.motion_blur(img)
                else:
                    return img
        elif self.do_cut_out:
            def pf(img):
                if np.random.rand() > .5:
                    return self.cut_out(img)
                else:
                    return img
        else:
            pf = None
        datagen = ImageDataGenerator(
            rescale=1 / 255.,
            rotation_range=8,
            width_shift_range=0.05,
            height_shift_range=0.10,
            brightness_range=(0.5, 1.5),
            shear_range=8,
            zoom_range=0.10,
            preprocessing_function=pf
        )

        datagen_validator = ImageDataGenerator(
            rescale=1 / 255.
        )

        return datagen, datagen_validator

    @staticmethod
    def cut_out(img):
        '''
        Modificada de la implementacion original
        https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        '''
        h, w, _ = img.shape
        mask = np.ones((h, w), np.float32)
        n_holes = random.randint(1, 6)
        side_len = random.randint(5, 15)
        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - side_len // 2, 0, h)
            y2 = np.clip(y + side_len // 2, 0, h)
            x1 = np.clip(x - side_len // 2, 0, w)
            x2 = np.clip(x + side_len // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = np.expand_dims(mask, axis=-1)
        img = img * mask
        return img

    @staticmethod
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
