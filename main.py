from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from FaceAging_AAE import FaceAging_AAE
from FaceAging_CGAN import FaceAging_CGAN
import argparse
import pprint


flags = tf.app.flags
flags.DEFINE_boolean('is_train', True, 'whether to train')
# flags.DEFINE_integer('epoch', 50, 'number of epochs')
flags.DEFINE_string('dataset', 'UTKFace', 'training dataset name that stored in ./data')
flags.DEFINE_string('loaddir', 'None', 'dir of loading checkpoints and intermediate training results')
flags.DEFINE_string('savedir', 'save', 'dir of saving checkpoints and intermediate training results')
flags.DEFINE_string('testdir', 'None', 'dir of testing images')
flags.DEFINE_string('model', 'CDCGAN', 'which part of model to train')
flags.DEFINE_boolean('use_trained_model', True, 'whether train from an existing model or from scratch')
flags.DEFINE_boolean('uniform_z', True, 'whether train from an existing model or from scratch')
FLAGS = flags.FLAGS


def main(_):

    pprint.pprint(FLAGS.flag_values_dict())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        model = FaceAging_CGAN(
            session,  # TensorFlow session
            is_training=FLAGS.is_train,
            save_dir=FLAGS.savedir,
            load_dir=FLAGS.loaddir,
            dataset_name=FLAGS.dataset,
            model=FLAGS.model,
            uniform_z=FLAGS.uniform_z,
        )
        if FLAGS.is_train:
            print '\n\tTraining Mode'
            if not FLAGS.use_trained_model:
                ## CGAN
                model.train(
                    num_epochs=250,  # number of epochs
                    use_trained_model=FLAGS.use_trained_model,
                )
            else:
                ## encoder
                model.train(
                    num_epochs=100,
                    use_trained_model=FLAGS.use_trained_model,
                )
        else:
            print '\n\tTesting Mode'
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*jpg'
            )


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    tf.app.run()