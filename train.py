# -*- coding: utf-8 -*-
# File: cifar10-resnet.py
# Author: Yuxin Wu

import argparse
import os

import cv2
import tensorflow as tf
from keras_applications import resnet
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50, resnet_v2_block, resnet_v2
from tensorpack.utils import logger
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu

from dataset import MPIIFaceGaze

BATCH_SIZE = 16
NUM_UNITS = None

def resnet_v2_mini(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=16, num_units=2, stride=2),
      resnet_v2_block('block2', base_depth=32, num_units=2, stride=2),
      resnet_v2_block('block3', base_depth=32, num_units=2, stride=2),
      resnet_v2_block('block4', base_depth=64, num_units=2, stride=1),
  ]
  return resnet_v2(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)

class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 112, 112, 3], 'input'),
                tf.placeholder(tf.float32, [None, 2], 'label')]

    def build_graph(self, image, label):
        net, end_points = resnet_v2_mini(image, num_classes=2, is_training=get_current_tower_context().is_training)

        cost = tf.reduce_mean(tf.abs(net - label), name="total_loss")

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))  # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(is_train):
    ds = MPIIFaceGaze(dir="C:/Users/Guo/Documents/MPIIFaceGaze_normalizad/",data_txt="data.txt", is_train=is_train)


    if is_train:
        augmentors = [
            # imgaug.Resize(112)
            # imgaug.CenterPaste((40, 40)),
            # imgaug.RandomCrop((32, 32)),
            # imgaug.Flip(horiz=True),
            # imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            # imgaug.Resize(112)
            # imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = MultiThreadMapData(ds, 2, map_func=MPIIFaceGaze._mapf)
    ds = BatchData(ds, BATCH_SIZE, remainder=not is_train)

    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=18)
    parser.add_argument('--load', help='load model for training')
    args = parser.parse_args()
    NUM_UNITS = args.num_units

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    dataset_train = get_data(is_train=True)
    dataset_test = get_data(is_train=False)

    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('total_loss')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
        ],
        max_epoch=400,
        session_init=SmartInit(args.load),
    )
    # num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(1))
