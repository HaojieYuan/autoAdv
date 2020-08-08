import numpy as np
import tensorflow as tf
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Transforming dataset images to tf records.')
parser.add_argument('--folder', default='/home/haojieyuan/Data/ImageNet/nips2017_dev', type=str)
parser.add_argument('--listfile',
                    default='/home/haojieyuan/Data/ImageNet/nips2017_dev_gt_label.txt', type=str)
parser.add_argument('--out', default='./tmp.tfrecords', type=str)
args = parser.parse_args()

writer = tf.python_io.TFRecordWriter(args.out)

img_prefix = args.folder

f = open(args.listfile)


for line in f:
    img_name = line.strip().split(' ')[0]

    # labels are all ranged 1~class_num
    # subtracting 1 here makes it 0~class_num-1, which is more
    # friendly for training procedure.
    label = int(line.strip().split(' ')[1]) - 1

    image = Image.open(os.path.join(img_prefix, img_name))
    width, height = image.size
    img_b = image.tobytes()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_b])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))}))


    writer.write(record=example.SerializeToString())

writer.close()
