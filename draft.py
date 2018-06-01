from tensorflow.contrib.training.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.ops import init_ops

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import glob, os, sys
from argparse import ArgumentParser
import utils, config

def gan_train_ops(
    self,
    model,
    loss,
    generator_optimizer,
    discriminator_optimizer,
    check_for_unused_update_ops=True,
):
    # Create global step increment op.
    global_step = training_util.get_or_create_global_step()
    global_step_inc = global_step.assign_add(1)

    update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    all_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, model.EG_scope.name))
    update_ops = list(all_ops & update_ops)
    with ops.name_scope('EG_train'):
        gen_train_op = training.create_train_op(
            total_loss=self.loss_EG,
            optimizer=self.EG_optimizer,
            variables_to_train=self.E_variables + self.G_variables,
            global_step=self.EG_global_step,
            update_ops=update_ops)

    update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    all_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, model.Dz_scope.name))
    update_ops = list(all_ops & update_ops)
    with ops.name_scope('Dz_train'):
        gen_train_op = training.create_train_op(
            total_loss=self.loss_Dz,
            optimizer=self.D_z_optimizer,
            variables_to_train=self.E_variables + self.G_variables,
            global_step=self.EG_global_step,
            update_ops=update_ops)

    update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    all_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, model.Di_scope.name))
    update_ops = list(all_ops & update_ops)
    with ops.name_scope('Di_train'):
        gen_train_op = training.create_train_op(
            total_loss=self.loss_Di,
            optimizer=self.D_img_optimizer,
            variables_to_train=self.E_variables + self.G_variables,
            global_step=self.EG_global_step,
            update_ops=update_ops)

    return namedtuples.GANTrainOps(gen_train_op, disc_train_op, global_step_inc)

def gan_train(
    self,
    train_ops,
    logdir,
    get_hooks_fn=get_joint_train_hooks(),
    master='',
    is_chief=True,
    scaffold=None,
    hooks=None,
    chief_only_hooks=None,
    save_checkpoint_secs=600,
    save_summaries_steps=100,
    config=None
):
    new_hooks = get_hooks_fn(train_ops)
    if hooks is not None:
        hooks = list(hooks) + list(new_hooks)
    else:
        hooks = new_hooks
    return training.train(
            train_ops.global_step_inc_op,
            logdir,
            master=master,
            is_chief=is_chief,
            scaffold=scaffold,
            hooks=hooks,
            chief_only_hooks=chief_only_hooks,
            save_checkpoint_secs=save_checkpoint_secs,
            save_summaries_steps=save_summaries_steps,
            config=config)


def get_joint_train_hooks(self, train_steps=(1, 1)):
    g_steps = train_steps[0]
    d_steps = train_steps[1]
    # Get the number of each type of step that should be run.
    num_d_and_g_steps = min(g_steps, d_steps)
    num_g_steps = g_steps - num_d_and_g_steps
    num_d_steps = d_steps - num_d_and_g_steps

    def get_hooks(train_ops):
        g_op = train_ops.generator_train_op
        d_op = train_ops.discriminator_train_op

        joint_hook = RunTrainOpsHook([g_op, d_op], num_d_and_g_steps)
        g_hook = RunTrainOpsHook(g_op, num_g_steps)
        d_hook = RunTrainOpsHook(d_op, num_d_steps)

        return [joint_hook, g_hook, d_hook]
    return get_hooks

status_message = tf.string_join(
        ['Starting train step: ', 
        tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')

gan_train(
    train_ops,
    hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
         tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
    logdir=FLAGS.train_log_dir,
    get_hooks_fn=get_joint_train_hooks())




def read_parse_preproc(filename_queue):
    ''' read, parse, and preproc single example. '''
    with tf.variable_scope('read_parse_preproc'):
        reader = tf.TFRecordReader()
        key, records = reader.read(filename_queue)
        
        # parse records
        features = tf.parse_single_example(
            records,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "path": tf.FixedLenFeature([], tf.string),
            }
        )

        image = tf.decode_raw(features["image"], tf.uint8)
        image = tf.reshape(image, [128, 128, 3]) # The image_shape must be explicitly specified
        # image = tf.image.resize_images(image, [64, 64])
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1.0 # preproc - normalize
        
        return [image]


def input_pipeline(glob_pattern, batch_size, num_threads, num_epochs, min_after_dequeue=None):
    tfrecords_list = glob.glob(glob_pattern)

    name = "batch_join" if not shuffle else "shuffle_batch_join"
    with tf.variable_scope(name):
        filename_queue = tf.train.string_input_producer(tfrecords_list, shuffle=shuffle, num_epochs=num_epochs)
        example_list = [read_parse_preproc(filename_queue) for _ in range(num_threads)]
        
        if min_after_dequeue is None:
            min_after_dequeue = batch_size * 10
        capacity = min_after_dequeue + 3*batch_size
        batch = tf.train.shuffle_batch_join(tensors_list=example_list, batch_size=batch_size, capacity=capacity, 
                                            min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
        return batch



def sample_z(shape):
    return np.random.normal(size=shape)


def train(model, dataset, input_op, num_epochs, batch_size, n_examples, ckpt_step, renew=False):
    # n_examples = 202599 # same as util.num_examples_from_tfrecords(glob.glob('./data/celebA_tfrecords/*.tfrecord'))
    # 1 epoch = 1583 steps
    print("\n# of examples: {}".format(n_examples))
    print("steps per epoch: {}\n".format(n_examples//batch_size))

    summary_path = os.path.join('./summary/', dataset, model.name)
    ckpt_path = os.path.join('./checkpoints', dataset, model.name)
    if renew:
        if os.path.exists(summary_path):
            tf.gfile.DeleteRecursively(summary_path)
        if os.path.exists(ckpt_path):
            tf.gfile.DeleteRecursively(ckpt_path)
    if not os.path.exists(ckpt_path):
        tf.gfile.MakeDirs(ckpt_path)

    config = tf.ConfigProto()
    best_gpu = utils.get_best_gpu()
    config.gpu_options.visible_device_list = str(best_gpu) # Works same as CUDA_VISIBLE_DEVICES!
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # for epochs 

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # https://github.com/tensorflow/tensorflow/issues/10972        
        # TensorFlow 1.2 has much bugs for text summary
        # make config_summary before define of summary_writer - bypass bug of tensorboard
        
        # It seems that batch_size should have been contained in the model config ... 
        total_steps = int(np.ceil(n_examples * num_epochs / float(batch_size))) # total global step
        config_list = [
            ('num_epochs', num_epochs),
            ('total_iteration', total_steps),
            ('batch_size', batch_size), 
            ('dataset', dataset)
        ]
        model_config_list = [[k, str(w)] for k, w in sorted(model.args.items()) + config_list]
        model_config_summary_op = tf.summary.text(model.name + '/config', tf.convert_to_tensor(model_config_list), 
            collections=[])
        model_config_summary = sess.run(model_config_summary_op)

        # print to console
        print("\n====== Process info =======")
        print("argv: {}".format(' '.join(sys.argv)))
        print("PID: {}".format(os.getpid()))
        print("====== Model configs ======")
        for k, v in model_config_list:
            print("{}: {}".format(k, v))
        print("===========================\n")

        summary_writer = tf.summary.FileWriter(summary_path, flush_secs=30, graph=sess.graph)
        summary_writer.add_summary(model_config_summary)
        pbar = tqdm(total=total_steps, desc='global_step')
        saver = tf.train.Saver(max_to_keep=9999) # save all checkpoints
        global_step = 0

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = sess.run(model.global_step)
            print('\n[!] Restore from {} ... starting global step is {}\n'.format(ckpt.model_checkpoint_path, global_step))
            pbar.update(global_step)

        try:
            # If training process was resumed from checkpoints, input pipeline cannot detect
            # when training should stop. So we need `global_step < total_step` condition.
            while not coord.should_stop() and global_step < total_steps:
                # model.all_summary_op contains histogram summary and image summary which are heavy op
                summary_op = model.summary_op if global_step % 100 == 0 else model.all_summary_op

                batch_X = sess.run(input_op)
                batch_z = sample_z([batch_size, model.z_dim])

                _, summary = sess.run([model.D_train_op, summary_op], {model.X: batch_X, model.z: batch_z})
                _, global_step = sess.run([model.G_train_op, model.global_step], {model.z: batch_z})

                summary_writer.add_summary(summary, global_step=global_step)

                if global_step % 10 == 0:
                    pbar.update(10)

                    if global_step % ckpt_step == 0:
                        saver.save(sess, ckpt_path+'/'+model.name, global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('\nDone -- epoch limit reached\n')
        finally:
            coord.request_stop()

        coord.join(threads)
        summary_writer.close()
        pbar.close()

def parse_fn(serial_exmp):  
    feats = tf.parse_single_example(serial_exmp, features={
        'image':tf.FixedLenFeature([], tf.string),
        'age':tf.FixedLenFeature([],tf.float32),
        'gender':tf.FixedLenFeature([],tf.float32),
        })
    # image = tf.decode_raw(feats['image'], tf.uint8)
    # image = tf.reshape(image, [128, 128, 3])
    # image = tf.cast(image, tf.float32)
    image = tf.image.decode_image(feats['image'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [64, 64])
    image = image / 127.5 - 1.0

    age = feats['age']
    gender = feats['gender']
    return image, age, gender

def input_pipeline_new(tfrecords_list, batch_size, num_epochs, prefetch=False):
    dataset = tf.data.TFRecordDataset(tfrecords_list)
    # dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_calls)  ## num_parallel_calls
    # dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size, num_parallel_batches=num_parallel_batches))
    # dataset = dataset.shuffle(1000).repeat(num_epochs)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, num_epochs))
    if prefetch:
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)        ## prefetch_buffer_size
    return dataset


def how_to_use():
    data_path = os.path.join('data', 'UTKFace_16_tfrecords')
    tf_tr_list = sorted(os.listdir(data_path))[:-1]
    te_tr_list = sorted(os.listdir(data_path))[-1]

    dataset_train = input_pipeline_new(tf_tr_list, batch_size=100, num_epochs=300)
    # one-shot
    iter_train = dataset_train.make_one_shot_iterator()
    image, age, gender = iter_train.get_next()

    loss = model_function(image, age, gender)
    training_op = tf.train.AdagradOptimizer(...).minimize(loss)

    saver_hook = tf.train.CheckpointSaverHook(...)
    summary_hook = tf.train.SummarySaverHook(...)
    with tf.train.MonitoredSession(session_creator=ChiefSessionCreator(...),
                          hooks=[saver_hook, summary_hook]) as sess:
        while not sess.should_stop():
            sess.run(training_op)



if __name__ == "__main__":

    # get information for dataset
    dataset_pattern = './data/UTKFace_128_tfrecords/*.tfrecord'
    n_examples = ???
    # input pipeline
    X = input_pipeline(dataset_pattern, batch_size=100, shuffle=True, num_threads=4, num_epochs=300, min_after_dequeue=None)

    model = config.get_model(FLAGS.model, FLAGS.name, training=True)
    train(model=model, dataset=FLAGS.dataset, input_op=X, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, 
        n_examples=n_examples, ckpt_step=FLAGS.ckpt_step, renew=FLAGS.renew)