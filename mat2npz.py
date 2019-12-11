import h5py
import numpy as np

def mat2npz():
    pass

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
    for i in range(3000):
        np.savez_compressed('C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/p00/p00_{}.npz'.format(i), data=data[i,...], label=label[i,...])
        print(i)

    mat2npz()