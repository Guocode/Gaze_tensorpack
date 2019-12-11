from scipy.io import loadmat

import h5py
import numpy as np
from tensorpack import RNGDataFlow
import cv2

class MPIIFaceGaze(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=None, dir="C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/"):
        self.dir = dir
        if shuffle is None:
            shuffle = train_or_test == 'train'
        self.shuffle = shuffle

    def __len__(self):
        return

    def __iter__(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            # since cifar is quite small, just do it for safety
            yield self.data[k]


if __name__ == '__main__':

    mat = "C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/p00.mat"
    m = h5py.File(mat)
    Data = m['Data']
    data = Data['data']
    label = Data['label']
    par_a_h = Data['par_a_h']
    par_a_w = Data['par_a_w']
    par_b_h = Data['par_b_h']
    par_b_w = Data['par_b_w']
    screenPattern = Data['screenPattern']
    screen_height = Data['screen_height']
    screen_width = Data['screen_width']
    data_np = data[:]
    print(m)