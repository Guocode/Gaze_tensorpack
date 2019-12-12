import os

from scipy.io import loadmat

import h5py
import numpy as np
from tensorpack import RNGDataFlow
import cv2


class MPIIFaceGaze(RNGDataFlow):
    def __init__(self, dir="C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/", is_train=True):
        self.dir = dir
        self.shuffle = is_train
        if is_train:
            self.mat_id = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        else:
            self.mat_id = ["11", "12", "13", "14"]
        self.all_data, self.all_label = self._load_all_mat()
        print("Dataset samples :", len(self.all_data))

    def __len__(self):
        return len(self.all_data)

    def __iter__(self):
        idxs = np.arange(len(self.all_data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield self.all_data[k], self.all_label[k]

    def _load_all_mat(self):
        all_data = []
        all_label = []
        for i in self.mat_id:
            mat_full_path = os.path.join(self.dir, "p{}.mat".format(i))
            m = h5py.File(mat_full_path)
            Data = m['Data']
            data = Data['data']
            label = Data['label']
            # par_a_h = Data['par_a_h']
            # par_a_w = Data['par_a_w']
            # par_b_h = Data['par_b_h']
            # par_b_w = Data['par_b_w']
            # screenPattern = Data['screenPattern']
            # screen_height = Data['screen_height']
            # screen_width = Data['screen_width']
            data_np = data[:10]
            label_np = label[:10]
            all_data.append(data_np)
            all_label.append(label_np)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)

        return all_data, all_label


if __name__ == '__main__':
    ds = MPIIFaceGaze(dir="C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/", is_train=False)
    gaze_line_len = 200
    pose_line_len = 200
    for data, label in ds:
        img_show = data.transpose((1, 2, 0)).astype(np.uint8)
        label = label
        # draw 6 face keypoints
        for a in range(6):
            img_show = cv2.circle(img_show, (label[a * 2 + 4], label[a * 2 + 5]), 3, (0, 0, 255), -1)

        pt1 = (label[4:6] + label[6:8]) / 2  # right eye center point
        pt2 = pt1 + gaze_line_len * label[0:2]
        img_show = cv2.line(img_show, pt1=tuple(pt1), pt2=tuple(pt2), color=(255, 0, 255), thickness=3)

        pt1 = (label[8:10] + label[10:12]) / 2  # left eye center point
        pt2 = pt1 + gaze_line_len * label[0:2]
        img_show = cv2.line(img_show, pt1=tuple(pt1), pt2=tuple(pt2), color=(255, 0, 255), thickness=3)

        pt1 = np.asarray((224.0, 224.0), dtype=np.float32)  # left eye center point
        pt2 = pt1 + pose_line_len * label[2:4]
        img_show = cv2.line(img_show, pt1=tuple(pt1), pt2=tuple(pt2), color=(255, 0, 0), thickness=3)

        print(label[0:2])
        print(label[2:4])

        cv2.imshow("a", img_show)
        cv2.waitKey(0)

    # mat = "C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/p00.mat"
    # m = h5py.File(mat)
    # Data = m['Data']
    # data = Data['data']
    # label = Data['label']
    # par_a_h = Data['par_a_h']
    # par_a_w = Data['par_a_w']
    # par_b_h = Data['par_b_h']
    # par_b_w = Data['par_b_w']
    # screenPattern = Data['screenPattern']
    # screen_height = Data['screen_height']
    # screen_width = Data['screen_width']
    # data_np = data[:100]
    # label_np = label[:100]
    # np.savez_compressed('C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/p00.npz', data=data_np[:100,...], label=label_np[:100,...])
    # print(m)
