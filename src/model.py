###################################################
#   Author: Liu Aishan                            #
#   Date: 2018.4.10                               #
#   Architecture of AdvPGAN                       #
###################################################


import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from src.ops import _conv_layer
from src.ops import _residual_block
from src.ops import _conv_tranpose_layer
from src.ops import lrelu
from src.ops import batch_norm
from src.ops import linear
from src.ops import conv2d
from src.utils import load_data
from src.utils import save_obj

import os
import time
import glob
import numpy as np


# TODO
# 1.refactor, remove similar functions like: conv2d, _conv_layer
# 2.implementation: pad_patch_on_image()
# 3.loss function


class AdvPGAN(object):

    def __init__(self, sess, batch_size=16, image_size=256, patch_size=28,
                 channel=3, alpha=1, beta=1, gamma=1, learning_rate=0.0001,
                 epoch=10000, traindata_size=10000,
                 data_dir=None,checkpoint_dir=None,output_dir=None):

        # hyperparameter
        self.df_dim = 64
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_channel = channel
        self.d_vars = []
        self.g_vars = []
        self.sess = sess
        self.data_dir = data_dir
        self.class_num = 1000
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.traindata_size = traindata_size
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

        self.build_model()


    # Generator of cGAN for adversarial patch
    # architecture of G is like image style transfer
    # input: patch image with size n*n
    # output: adversarial patch image with size n*n
    # no gaussian noise needed
    def generator(self, image):
        image = image / 255.0
        conv1 = _conv_layer(image, 32, 9, 1, name="g_conv1")
        conv2 = _conv_layer(conv1, 64, 3, 2, name="g_conv2")
        conv3 = _conv_layer(conv2, 128, 3, 2, name="g_conv3")
        resid1 = _residual_block(conv3, 3, name="g_resid1")
        resid2 = _residual_block(resid1, 3, name="g_resid2")
        resid3 = _residual_block(resid2, 3, name="g_resid3")
        resid4 = _residual_block(resid3, 3, name="g_resid4")
        resid5 = _residual_block(resid4, 3, name="g_resid5")
        conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2, name="g_deconv1")
        conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2, name="g_deconv2")
        conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False, name="g_deconv3")
        preds = tf.nn.tanh(conv_t3)
        output = image + preds
        return tf.nn.tanh(output) * 127.5 + 255./2


    # target model to attack
    def target_model_discriminator(self, image, reuse = False):
        preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
        arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
        with slim.arg_scope(arg_scope):
            logits, _ = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
            logits = logits[:, 1:]
            probs = tf.nn.softmax(logits)
        return logits, probs

    # todo
    # pad the adversarial patch on image
    def pad_patch_on_image(self, image, patch):
        return 1

    # naive discriminator in GAN
    # using for adversarial training
    def naive_discriminator(self, image, y = None, reuse = False):
        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image,  self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(batch_norm((conv2d(h0, self.df_dim * 2, name='d_h1_conv')), name="d_bn1"))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name='d_h2_conv'), name="d_bn2"))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name='d_h3_conv'), name="d_bn3"))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    # build cGAN model
    def build_model(self):

        # load target model
        restore_vars = [var for var in tf.global_variables() if var.name.startswith('InceptionV3/')]
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, os.path.join(self.data_dir, 'inception_v3.ckpt'))

        # get all trainable variables for G and D, respectively
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # input image
        self.real_image = tf.placeholder(tf.float32, [self.batch_size, self.image_size,
                                                      self.image_size, self.image_channel])
        # input patch (condition for GAN)
        self.real_patch = tf.placeholder(tf.float32, [self.batch_size, self.patch_size,
                                                      self.patch_size, self.image_channel])
        # original class label of input image
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.class_num])
        # adversarial patch generated by G
        self.fake_patch = self.generator(self.real_patch)
        # final adversarial image
        self.fake_image = self.pad_patch_on_image(image=self.real_image, patch=self.fake_patch)
        # classify result from target model
        self.fake_logits_f, self.fake_prob_f = self.target_model_discriminator(self.fake_image)
        # fake image result from naive D
        self.fake_logits_d, self.fake_prob_d = self.naive_discriminator(self.fake_image)
        # real image result from naive D
        self.real_logits_d, self.real_prob_d = self.naive_discriminator(self.real_image)


        # The loss of AdvPGAN consists of: GAN loss, patch loss and adversarial example loss
        # GAN loss for D and G
        self.loss_d_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits_d,
                                                                                 labels=tf.ones_like(self.real_logits_d))+
                                         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits_d,
                                                                                 labels=tf.zeros_like(self.fake_logits_d)))
        self.loss_g_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fake_logits_d,
                                                                                 labels=tf.ones(self.fake_logits_d)))

        # patch loss
        self.patch_loss = 0
        # adversarial example loss
        self.ae_loss = 0

        # overall loss for D and G
        self.g_loss = self.alpha * self.loss_g_adv + self.beta * self.patch_loss \
                      + self.gamma * self.ae_loss
        self.d_loss = self.loss_d_adv


        # initialize a saver
        self.saver = tf.train.Saver()

    # train cGAN model
    def train_op(self):
        d_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).\
            minimize(self.d_loss, var_list=self.d_vars)
        g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).\
            minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)

        start_time = time.time()
        counter = 1

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(self.epoch):

            image_set = glob(self.data_dir+r'*.jpg')
            batch_iteration = min(len(image_set), self.traindata_size) / self.batch_size

            for id in range(batch_iteration):
                batch_files = image_set[id*self.batch_size: (id+1)*self.batch_size]
                batch_data_x, batch_data_y = [load_data(file, self.image_size) for file in batch_files]

                batch_data_x = np.array(batch_data_x).astype(np.float32)
                batch_data_y = np.array(batch_data_y).astype(np.float32)

                self.sess.run([d_opt],
                             feed_dict={self.real_image: batch_data_x})

                self.sess.run([g_opt],
                              feed_dict={self.real_image: batch_data_x,
                                         self.y: batch_data_y,
                                         self.real_patch: })

                errD = self.d_loss.eval({self.real_image: batch_data_x})
                errG = self.g_loss.eval({self.real_image: batch_data_x,
                                         self.y: batch_data_y,
                                         self.real_patch: })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, id, batch_iteration,
                         time.time() - start_time, errD, errG))

                # serialize and save image objects
                if np.mod(counter, 100) == 0:
                    save_obj(self.fake_image.eval({self.real_image: batch_data_x,
                                                   self.fake_patch: }),
                             filename=self.output_dir+'/' + str(time.time() +'_image.pkl'))

                # save model
                if np.mod(counter, 500) == 0:
                    self.save(self.checkpoint_dir, counter)

    # save model
    def save(self, checkpoint_dir, step):
        model_name = "AdvPGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    # load model
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False



