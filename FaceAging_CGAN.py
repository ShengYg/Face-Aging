from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from ops import *
import cPickle
from dataset import UTKFace


class FaceAging_CGAN(object):
    def __init__(
        self,
        session,  # TensorFlow session
        size_image=128,  # size the input images
        size_kernel=4,
        size_batch=100,  # mini-batch size for training and testing, must be square of an integer
        num_input_channels=3,  # number of channels of input images
        num_encoder_channels=64,  # number of channels of the first conv layer of encoder
        num_z_channels=100,
        num_categories=10,  # number of categories (age segments) in the training dataset
        num_gen_channels=512,
        enable_tile_label=True,  # enable to tile the label
        tile_ratio=1.0,  # ratio of the length between tiled label and z
        is_training=True,  # flag for training or testing mode
        save_dir='./save',  # path to save checkpoints, samples, and summary
        load_dir='./load',
        dataset_name='UTKFace',  # name of the dataset in the folder ./data
        model='CDCGAN',
    ):
        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.dataset_name = dataset_name
        self.model = model

        # ************************************* input to graph ********************************************************
        self.input_image = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_images'
        )
        self.age = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_categories],
            name='age_labels'
        )
        self.gender = tf.placeholder(
            tf.float32,
            [self.size_batch, 2],
            name='gender_labels'
        )
        self.z_prior = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_z_channels],
            name='z_prior'
        )
        # ************************************* build the graph *******************************************************
        print '\n\tBuilding graph ...'

        self.summary_list = []
        if self.model == 'CDCGAN':
            # build graph
            self.gen_image = self.generator(
                z=self.z_prior,
                y=self.age,
                gender=self.gender,
                is_training=self.is_training,
                enable_tile_label=self.enable_tile_label,
                tile_ratio=self.tile_ratio,
                enable_bn=False
            )
            self.D_fake, self.D_fake_logits = self.discriminator(
                image=self.gen_image,
                y=self.age,
                gender=self.gender,
                is_training=self.is_training,
                enable_tile_label=self.enable_tile_label,
            )
            self.D_real, self.D_real_logits = self.discriminator(
                image=self.input_image,
                y=self.age,
                gender=self.gender,
                is_training=self.is_training,
                enable_tile_label=self.enable_tile_label,
                reuse_variables=True
            )

            # set loss
            self.D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real_logits))
            )
            self.D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake_logits))
            )
            self.G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake_logits))
            )
            self.D_loss = self.D_loss_real + self.D_loss_fake

            # add summary
            self.D_loss_real_summary = tf.summary.scalar('D_loss_real', self.D_loss_real)
            self.D_loss_fake_summary = tf.summary.scalar('D_loss_fake', self.D_loss_fake)
            self.D_loss_summary = tf.summary.scalar('D_loss', self.D_loss)
            self.G_loss_summary = tf.summary.scalar('G_loss', self.G_loss)
            self.D_real_logits_summary = tf.summary.histogram('D_real_logits', self.D_real_logits)
            self.D_fake_logits_summary = tf.summary.histogram('D_fake_logits', self.D_fake_logits)
            self.summary_list.extend([  self.D_loss_real_summary, self.D_loss_fake_summary, 
                                        self.D_real_logits_summary, self.D_fake_logits_summary, 
                                        self.G_loss_summary, self.D_loss_summary])
        else:
            # build graph
            self.z = self.encoder(
                image=self.input_image,
                is_training=self.is_training,
                enable_bn=False
            )
            self.gen_image = self.generator(
                z=self.z,
                y=self.age,
                gender=self.gender,
                is_training=self.is_training,
                enable_tile_label=self.enable_tile_label,
                tile_ratio=self.tile_ratio,
                enable_bn=False
            )
            
            # add summary
            self.z_summary = tf.summary.histogram('z', self.z)
            self.summary_list.append(self.z_summary)

        ## l2 loss
        self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.gen_image))

        # total variation to smooth the generated image
        tv_y_size = self.size_image
        tv_x_size = self.size_image
        self.tv_loss = (
            (tf.nn.l2_loss(self.gen_image[:, 1:, :, :] - self.gen_image[:, :self.size_image - 1, :, :]) / tv_y_size) +
            (tf.nn.l2_loss(self.gen_image[:, :, 1:, :] - self.gen_image[:, :, :self.size_image - 1, :]) / tv_x_size)) / self.size_batch

        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        self.D_variables = [var for var in trainable_variables if 'D_' in var.name]

        # ************************************* collect the summary ***************************************
        
        self.z_prior_summary = tf.summary.histogram('z_prior', self.z_prior)
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.tv_loss_summary = tf.summary.scalar('tv_loss', self.tv_loss)

        # for saving the graph and variables
        self.saver = tf.train.Saver(max_to_keep=10)
        # self.saver_gd = tf.train.Saver(self.G_variables+self.D_variables, max_to_keep=10)
        self.summary_list.extend([  self.z_prior_summary, 
                                    self.EG_loss_summary, 
                                    self.tv_loss_summary,])

    def train(
        self,
        num_epochs=200,  # number of epochs
        learning_rate=0.0002,  # learning rate of optimizer
        beta1=0.5,  # parameter for Adam optimizer
        decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
        enable_shuffle=True,  # enable shuffle of the dataset
        use_trained_model=True,  # use the saved checkpoint to initialize the network
    ):

        # set learning rate decay
        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')

        # optimizer for encoder + generator
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            if self.model == 'CDCGAN':
                # D_loss
                self.D_optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate,
                    beta1=beta1
                ).minimize(
                    loss=self.D_loss,
                    global_step=self.EG_global_step,
                    var_list=self.D_variables
                )

                # G_loss
                self.G_optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate*5,
                    beta1=beta1
                ).minimize(
                    loss=self.G_loss,
                    global_step=self.EG_global_step,
                    var_list=self.G_variables
                )
            else:
                # EG_loss
                self.EG_optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate,
                    beta1=beta1
                ).minimize(
                    loss=self.EG_loss,
                    global_step=self.EG_global_step,
                    var_list=self.E_variables
                )


        # *********************************** tensorboard *************************************************************
        # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
        # self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        # self.summary_list.append(self.EG_learning_rate_summary)
        self.summary = tf.summary.merge(self.summary_list)
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)

        # *************************** load file names of images ******************************************************

        print("\n\tLoading Dataset...")
        if self.dataset_name == 'UTKFace':
            self.dataset = UTKFace()

        file_names = self.dataset.train_list
        sample_files = self.dataset.test_list
        sample_images, sample_label_age, sample_label_gender = self.dataset.get_dataset(sample_files)
        
        # ******************************************* training *******************************************************
        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            print('\n\tLoading model ...')
            self.load_checkpoint(model_path=os.path.join(self.load_dir, 'checkpoint'), saver=self.saver_gd)

        # epoch iteration
        print("\n\tStart Training ...")
        num_batches = len(file_names) // self.size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(file_names)

            for ind_batch in range(num_batches):
                start_time = time.time()

                # read batch images and labels
                batch_files = file_names[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch_images, batch_label_age, batch_label_gender = self.dataset.get_dataset(batch_files)

                # prior distribution on the prior of z
                batch_z_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_z_channels]
                ).astype(np.float32)

                # update
                if self.model == 'CDCGAN':
                    _, _, D_err, G_err, D_real_err, D_fake_err = self.session.run(
                        fetches = [
                            self.D_optimizer,
                            self.G_optimizer,
                            self.D_loss,
                            self.G_loss,
                            self.D_loss_real,
                            self.D_loss_fake,
                        ],
                        feed_dict={
                            self.input_image: batch_images,
                            self.age: batch_label_age,
                            self.gender: batch_label_gender,
                            self.z_prior: batch_z_prior,
                        }
                    )

                    print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tG_err=%.4f\tD_err=%.4f" %
                        (epoch+1, num_epochs, ind_batch+1, num_batches, G_err, D_err))
                    print("\tD_real_err=%.4f\tD_fake_err=%.4f" % (D_real_err, D_fake_err))
                else:
                    _, EG_err = self.session.run(
                        fetches = [
                            self.EG_optimizer,
                            self.EG_loss,
                        ],
                        feed_dict={
                            self.input_image: batch_images,
                            self.age: batch_label_age,
                            self.gender: batch_label_gender,
                            self.z_prior: batch_z_prior,
                        }
                    )

                    print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\t" %
                        (epoch+1, num_epochs, ind_batch+1, num_batches, EG_err))

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                # add to summary
                summary = self.summary.eval(
                    feed_dict={
                        self.input_image: batch_images,
                        self.age: batch_label_age,
                        self.gender: batch_label_gender,
                        self.z_prior: batch_z_prior
                    }
                )
                self.writer.add_summary(summary, self.EG_global_step.eval())

            # save sample images for each epoch
            name = '{:02d}.png'.format(epoch+1)
            self.sample(sample_images, sample_label_age, sample_label_gender, name)
            self.test(sample_images, sample_label_gender, name)

            # save checkpoint for each 5 epoch
            if np.mod(epoch, 5) == 4:
                self.save_checkpoint()

        # save the trained model
        self.save_checkpoint()
        # close the summary writer
        self.writer.close()

    def encoder(self, image, is_training=True, reuse_variables=False, enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    name=name
                )
            if enable_bn:
                name = 'E_conv_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)

        name = 'E_conv' + str(num_layers)
        current = conv2d(
                input_map=current,
                num_output_channels=self.num_z_channels,
                size_kernel=self.size_kernel,
                padding='VALID',
                name=name
            )
        current = tf.reshape(current, [self.size_batch, -1])
        return tf.nn.tanh(current)

    def generator(self, z, y, gender, is_training=True, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0, enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        size_mini_map = int(self.size_image / 2 ** num_layers)

        # z = tf.reshape(z, [-1, 1, 1, self.num_z_channels])
        current_z = self.generator_head(z, 'z', self.num_z_channels, self.num_gen_channels/2, enable_bn, is_training, reuse_variables)
        # y = tf.reshape(y, [-1, 1, 1, self.num_categories])
        current_age = self.generator_head(y, 'age', self.num_categories, self.num_gen_channels/2, enable_bn, is_training, reuse_variables)
        gender = tf.reshape(gender, [-1, 1, 1, 2])
        current_gender = self.generator_head(gender, 'gender', 2, self.num_gen_channels/2, enable_bn, is_training, reuse_variables)
        current = tf.concat(axis=3, values=[current_z, current_age, current_gender])
        self.gen_z_concat_summary = tf.summary.histogram('gen_z_concat', current_z)
        self.gen_age_concat_summary = tf.summary.histogram('gen_age_concat', current_age)
        self.gen_gender_concat_summary = tf.summary.histogram('gen_gender_concat', current_gender)


        # deconv layers with stride 2
        for i in range(num_layers-1):
            name = 'G_deconv' + str(i)
            current = deconv2d(
                    input_map=current,
                    output_shape=[self.size_batch,
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),
                                  int(self.num_gen_channels / 2 ** (i + 1))],
                    size_kernel=self.size_kernel,
                    name=name
                )
            if enable_bn:
                name = 'G_deconv_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)

        name = 'G_deconv' + str(i+1)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          self.num_input_channels],
            size_kernel=self.size_kernel,
            name=name
        )
        # output
        return tf.nn.tanh(current)

    def discriminator(self, image, y, gender, is_training=True, reuse_variables=False, enable_tile_label=True, enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = image

        for i in range(num_layers):
            if i == 0:
                y_shape = y.get_shape().as_list()
                current_shape = current.get_shape().as_list()
                gender_shape = gender.get_shape().as_list()
                
                y = tf.reshape(y, [current_shape[0], 1, 1, y_shape[-1]])
                y = y*tf.ones([current_shape[0], current_shape[1], current_shape[2], y_shape[-1]])

                gender = tf.reshape(gender, [current_shape[0], 1, 1, gender_shape[-1]])
                gender = gender*tf.ones([current_shape[0], current_shape[1], current_shape[2], gender_shape[-1]])

                name = 'D_conv0'
                current = conv2d(input_map=current, num_output_channels=self.num_encoder_channels/4, name=name)
                if enable_bn:
                    name = 'D_bn0'
                    current = tf.contrib.layers.batch_norm(current, scale=False, is_training=is_training, scope=name, reuse=reuse_variables)

                name = 'D_conv0_y'
                y = conv2d(input_map=y, num_output_channels=self.num_encoder_channels/4, name=name)
                if enable_bn:
                    name = 'D_bn0_y'
                    y = tf.contrib.layers.batch_norm(y, scale=False, is_training=is_training, scope=name, reuse=reuse_variables)

                name = 'D_conv0_gender'
                gender = conv2d(input_map=gender, num_output_channels=self.num_encoder_channels/4, name=name)
                if enable_bn:
                    name = 'D_bn0_gender'
                    gender = tf.contrib.layers.batch_norm(gender, scale=False, is_training=is_training, scope=name, reuse=reuse_variables)

                current = tf.concat(axis=3, values=[current, y, gender])
                current = tf.nn.leaky_relu(current, alpha=0.2)
            else:
                name = 'D_conv' + str(i)
                current = conv2d(
                        input_map=current,
                        num_output_channels=self.num_encoder_channels*(2**i),
                        name=name
                    )
                if enable_bn:
                    name = 'D_bn' + str(i)
                    current = tf.contrib.layers.batch_norm(
                        current,
                        scale=False,
                        is_training=is_training,
                        scope=name,
                        reuse=reuse_variables
                    )
                current = tf.nn.leaky_relu(current, alpha=0.2)

        name = 'D_conv' + str(num_layers)
        current = conv2d(
                input_map=current,
                num_output_channels=1,
                padding='VALID',
                name=name,
            )
        current = tf.reshape(current, [self.size_batch, -1])

        return tf.nn.sigmoid(current), current
        
    def generator_head(self, current, name='z', num_channels_in=50, num_channels_out=512, enable_bn=False, is_training=False, reuse_variables=False):
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        size_mini_map = int(self.size_image / 2 ** num_layers)

        current = tf.reshape(current, [-1, 1, 1, int(num_channels_in)])
        current = deconv2d(
                input_map=current,
                output_shape=[self.size_batch, size_mini_map, size_mini_map, int(num_channels_out)],
                size_kernel=self.size_kernel,
                padding='VALID',
                name='G_init_'+name
            )
        if enable_bn:
            current = tf.contrib.layers.batch_norm(
                current,
                scale=False,
                is_training=is_training,
                scope='G_init_bn_'+name,
                reuse=reuse_variables
            )
        current = tf.nn.relu(current)
        return current

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )
        print("Saving checkpoint: {}-{}".format(os.path.join(checkpoint_dir, 'model'), self.EG_global_step.eval()))

    def load_checkpoint(self, model_path=None, saver=None):
        checkpoints = tf.train.get_checkpoint_state(model_path)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            try:
                saver.restore(self.session, os.path.join(model_path, checkpoints_name))
                print("\n\tLoading model {} from {}".format(checkpoints_name, model_path))
                return True
            except:
                print("\n\tLoading model failed")
                return False
        else:
            print("\n\tLoading model failed")
            return False

    def sample(self, images, labels, gender, name):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        batch_z_prior = np.random.uniform(
            self.image_value_range[0],
            self.image_value_range[-1],
            [self.size_batch, self.num_z_channels]
        ).astype(np.float32)
        gen_image = self.session.run(
            self.gen_image,
            feed_dict={
                self.input_image: images,
                self.age: labels,
                self.gender: gender,
                self.z_prior: batch_z_prior
            }
        )
        size_frame = int(np.sqrt(self.size_batch))
        save_batch_images(
            batch_images=gen_image,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )

    def test(self, images, gender, name):
        test_dir = os.path.join(self.save_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        images = images[:int(np.sqrt(self.size_batch)), :, :, :]
        gender = gender[:int(np.sqrt(self.size_batch)), :]
        size_sample = images.shape[0]
        labels = np.arange(size_sample)
        labels = np.repeat(labels, size_sample)
        query_labels = np.ones(
            shape=(size_sample ** 2, size_sample),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = self.image_value_range[-1]
        query_images = np.tile(images, [self.num_categories, 1, 1, 1])
        query_gender = np.tile(gender, [self.num_categories, 1])
        batch_z_prior = np.random.uniform(
            self.image_value_range[0],
            self.image_value_range[-1],
            [self.num_categories, self.num_z_channels]
        ).astype(np.float32)
        query_z = np.tile(batch_z_prior, [self.num_categories, 1])

        gen_image = self.session.run(
            self.gen_image,
            feed_dict={
                self.input_image: query_images,
                self.age: query_labels,
                self.gender: query_gender,
                self.z_prior: query_z
            }
        )
        save_batch_images(
            batch_images=query_images,
            save_path=os.path.join(test_dir, 'input.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=gen_image,
            save_path=os.path.join(test_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )

    def custom_test(self, testing_samples_dir):
        if not self.load_checkpoint():
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        num_samples = int(np.sqrt(self.size_batch))
        file_names = glob(testing_samples_dir)
        if len(file_names) < num_samples:
            print 'The number of testing images is must larger than %d' % num_samples
            exit(0)
        cache_file = os.path.join(self.save_dir, 'test_files.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                sample_files = cPickle.load(fid)
        else:
            raise Exception("{} not found".format(cache_file))
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            images = np.array(sample).astype(np.float32)
        gender_male = np.ones(
            shape=(num_samples, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        gender_female = np.ones(
            shape=(num_samples, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(gender_male.shape[0]):
            gender_male[i, 0] = self.image_value_range[-1]
            gender_female[i, 1] = self.image_value_range[-1]

        self.test(images, gender_male, 'test_as_male.png')
        self.test(images, gender_female, 'test_as_female.png')

        print '\n\tDone! Results are saved as %s\n' % os.path.join(self.save_dir, 'test', 'test_as_xxx.png')


