import os

import cv2
import h5py
import numpy as np

def mat2npz():
    pass

if __name__ == '__main__':
    f = open('data.txt', 'w')  # 设置文件对象

    for pid in range(0,15):
        mat = "C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/p{:0>2d}.mat".format(pid)
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
        if not os.path.exists("C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/p{:0>2d}".format(pid)):
            os.mkdir("C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/p{:0>2d}".format(pid))
        for i in range(data.shape[0]):
            img_save_name = 'C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/p{:0>2d}/p{:0>2d}_{:0>4d}.jpg'.format(pid, pid, i)
            # cv2.imwrite(img_save_name,data[i,...].transpose((1,2,0)))
            # np.savez_compressed(img_save_name[:-3]+"npz",label[i, ...])
            f.write("p{:0>2d}/p{:0>2d}_{:0>4d}.jpg\n".format(pid,pid,i))  # 将字符串写入文件中
            print(pid,i)
    f.close()