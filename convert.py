# coding: utf-8

import tensorflow as tf
import numpy as np
import scipy.misc
import os
import glob


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert(source_dir, target_dir, out_size, exts=[''], num_shards=128, tfrecords_prefix=''):
    if not tf.gfile.Exists(source_dir):
        raise Exception('source dir {} does not exists'.format(source_dir))
    
    if tfrecords_prefix and not tfrecords_prefix.endswith('-'):
        tfrecords_prefix += '-'

    if tf.gfile.Exists(target_dir):
        tf.gfile.DeleteRecursively(target_dir)
    tf.gfile.MakeDirs(target_dir)

    # get meta-data
    path_list = []
    for ext in exts:
        pattern = '*.' + ext if ext != '' else '*'
        path = os.path.join(source_dir, pattern)
        path_list.extend(glob.glob(path))

    # shuffle path_list
    np.random.shuffle(path_list)
    num_files = len(path_list)
    num_per_shard = num_files // num_shards # Last shard will have more files

    print('# of files: {}'.format(num_files))
    print('# of shards: {}'.format(num_shards))
    print('# files per shards: {}'.format(num_per_shard))

    # convert to tfrecords
    shard_idx = 0
    writer = None
    for i, path in enumerate(path_list):
        if i % num_per_shard == 0 and shard_idx < num_shards:
            shard_idx += 1
            tfrecord_fn = '{}{:0>4d}-of-{:0>4d}.tfrecord'.format(tfrecords_prefix, shard_idx, num_shards)
            tfrecord_path = os.path.join(target_dir, tfrecord_fn)
            print("Writing {} ...".format(tfrecord_path))
            if shard_idx > 1:
                writer.close()
            writer = tf.python_io.TFRecordWriter(tfrecord_path)

        im = scipy.misc.imread(path, mode='RGB')
        im = scipy.misc.imresize(im, out_size)

        label = int(str(path).split('/')[-1].split('_')[0])
        if 0 <= label <= 5:
            age = 0
        elif 6 <= label <= 10:
            age = 1
        elif 11 <= label <= 15:
            age = 2
        elif 16 <= label <= 20:
            age = 3
        elif 21 <= label <= 30:
            age = 4
        elif 31 <= label <= 40:
            age = 5
        elif 41 <= label <= 50:
            age = 6
        elif 51 <= label <= 60:
            age = 7
        elif 61 <= label <= 70:
            age = 8
        else:
            age = 9
        gender = int(str(path).split('/')[-1].split('_')[1])

        example = tf.train.Example(features=tf.train.Features(feature={
            # "shape": _int64_features(im.shape),
            "image": _bytes_features([im.tobytes()]),
            "age": _int64_features([age]),
            "gender": _int64_features([gender]),
        }))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == "__main__":
    convert('./data/UTKFace', './data/UTKFace_16_tfrecords', out_size=[128, 128], 
        exts=['jpg'], num_shards=16, tfrecords_prefix='UTKFace')

