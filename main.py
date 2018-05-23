from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from FaceAging import FaceAging
import argparse
import pprint


flags = tf.app.flags
flags.DEFINE_boolean('is_train', True, 'whether to train')
flags.DEFINE_integer('epoch', 50, 'number of epochs')
flags.DEFINE_string('dataset', 'UTKFace', 'training dataset name that stored in ./data')
flags.DEFINE_string('savedir', 'save', 'dir of saving checkpoints and intermediate training results')
flags.DEFINE_string('testdir', 'None', 'dir of testing images')
flags.DEFINE_boolean('use_trained_model', True, 'whether train from an existing model or from scratch')
flags.DEFINE_boolean('use_init_model', True, 'whether train from the init model if cannot find an existing model')
FLAGS = flags.FLAGS



# parser = argparse.ArgumentParser(description='CAAE')
# parser.add_argument('--is_train', type=bool, default=True)
# parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
# parser.add_argument('--dataset', type=str, default='UTKFace', help='training dataset name that stored in ./data')
# parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
# parser.add_argument('--testdir', type=str, default='None', help='dir of testing images')
# parser.add_argument('--use_trained_model', type=bool, default=True, help='whether train from an existing model or from scratch')
# parser.add_argument('--use_init_model', type=bool, default=True, help='whether train from the init model if cannot find an existing model')
# FLAGS = parser.parse_args()

############################
# kernel = 4
# enable_tile_label: no use
# G/D no bn
# change to cGAN 

def main(_):

    pprint.pprint(FLAGS.flag_values_dict())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        model = FaceAging(
            session,  # TensorFlow session
            is_training=FLAGS.is_train,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            dataset_name=FLAGS.dataset,  # name of the dataset in the folder ./data
            enable_tile_label=False
        )
        if FLAGS.is_train:
            print '\n\tTraining Mode'
            if not FLAGS.use_trained_model:
                print '\n\tPre-train the network'
                model.train(
                    num_epochs=30,  # number of epochs
                    use_trained_model=FLAGS.use_trained_model,
                    use_init_model=FLAGS.use_init_model,
                    weigts=(0, 0, 0)
                )
                print '\n\tPre-train is done! The training will start.'
            else:
                model.train(
                    num_epochs=FLAGS.epoch,  # number of epochs
                    use_trained_model=FLAGS.use_trained_model,
                    use_init_model=FLAGS.use_init_model
                )
        else:
            print '\n\tTesting Mode'
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*jpg'
            )


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    tf.app.run()