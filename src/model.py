##################################################
#   Author: Liu Aishan                            #
#   Date: 2018.4.10                               #
#   Architecture of AdvPGAN                       #
###################################################


import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from ops import _conv_layer
from ops import _residual_block
from ops import _conv_tranpose_layer
from ops import lrelu
from ops import batch_norm
from ops import layer_norm
from ops import linear
from ops import conv2d
from ops import deconv2d
from ops import gen_conv
from ops import gen_deconv
from utils import save_obj
from utils import save_patches
from utils import plot_acc
from utils import randomly_overlay
from utils import shuffle_augment_and_load
from utils import plot_images_and_acc
from utils import load_image
from GTSRB_Classifier import GTSRB_Classifier
from GTSRB_Classifier import GTSRB_Model
from utils import tv_loss
from utils import get_initial_image_patch_pair
from utils import load_data_in_pair

import os
import time
import math
import glob
import numpy as np
from scipy import misc
import pickle



class AdvPGAN(object):

    def __init__(self, sess, batch_size=16, image_size=128, patch_size=16,
                 channel=3, alpha=1, beta=1, gamma=1, learning_rate=0.0001,
                 epoch=100, traindata_size=10000,
                 base_image_num = 16, base_patch_num = 16, tv_weight = 0.0001,
                 target_model_dir=None, checkpoint_dir=None,output_dir=None):

        # hyperparameter
        self.df_dim = 64
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_channel = channel
        self.d_vars = []
        self.g_vars = []
        self.sess = sess
        self.class_num = 43
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.traindata_size = traindata_size
        self.target_model_dir=target_model_dir
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
	self.tv_weight = tv_weight
        self.rho = 1
        self.d_train_freq = 1
        self.image_dir = '/home/zhenxt/zal/GTSRB/data/train_1_100.p'
        self.test_img_dir = '/home/zhenxt/zal/GTSRB/data/train_0_9_100.p'
        self.valid_img_dir = '/home/zhenxt/zal/GTSRB/data/validation_img_1_8.p'
        self.patch_dir = '/home/zhenxt/zal/GTSRB/quickdraw/aircraft_carrier_r_100_resized.p'
        self.test_patch_dir = '/home/zhenxt/zal/GTSRB/quickdraw/aircraft_carrier_r_100_resized.p'
        self.valid_patch_dir = '/home/zhenxt/zal/GTSRB/quickdraw/validation_aircraft_carrier_r_10_resized_ext.p'
	self.patch_all_num = 100
	self.image_all_num = 100
	self.patch_val_num = 10
	self.image_val_num = 10
        self.base_image_num = base_image_num
        self.base_patch_num = base_patch_num
        self.acc_history = []
        self.delta = 0.002

        self.build_model()

  # Generator of cGAN for adversarial patch
    # architecture of G is like image style transfer
    # input: patch image with size n*n
    # output: adversarial patch image with size n*n
    # no gaussian noise needed
    def generator_pix2pix(self, image, reuse=False):
        output_size = self.patch_size
        s = math.ceil(output_size/16.0)*16
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        # gf_dim = 16 # Dimension of gen filters in first conv layer.
        with tf.variable_scope("generator") as scope:

            # image is 128 x 128 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            # do we need here???
            #image = image / 255.0
            # liuas 2018.5.9
            # trick: using lrelu instead of relu

            ngf = 16 # number of generator filters in first conv layer
            # encoder_1: [batch, 16, 16, 3] => [batch, 8, 8, ngf]
            conv1 = conv2d(image, ngf, k_h=4, k_w=4, name='adv_g_enc1')
            conv2 = layer_norm(conv2d(lrelu(conv1, 0.2), ngf*2, k_h=4, k_w=4, name='adv_g_enc2'), name='adv_g_enc2ln')
            conv3 = layer_norm(conv2d(lrelu(conv2, 0.2), ngf*4, k_h=4, k_w=4, name='adv_g_enc3'), name='adv_g_enc3ln')
            conv4 = layer_norm(conv2d(lrelu(conv3, 0.2), ngf*8, k_h=4, k_w=4, name='adv_g_enc4'), name='adv_g_enc4ln')
            deconv1, _, _ = deconv2d(tf.nn.relu(conv4), [self.batch_size, s8, s8, ngf*4], k_h=4, k_w=4, name='adv_g_dec1', with_w=True)
            deconv1 = layer_norm(deconv1, name="adv_g_dec1ln")
            input = tf.concat([deconv1, conv3], axis=3)
            deconv2, _, _ = deconv2d(tf.nn.relu(input), [self.batch_size, s4, s4, ngf*2], k_h=4, k_w=4, name='adv_g_dec2', with_w=True)
            deconv2 = layer_norm(deconv2, name="adv_g_dec2ln")
            input = tf.concat([deconv2, conv2], axis=3)
            deconv3, _, _ = deconv2d(tf.nn.relu(input), [self.batch_size, s2, s2, ngf], k_h=4, k_w=4, name='adv_g_dec3', with_w=True)
            deconv3 = layer_norm(deconv3, name="adv_g_dec3ln")
            input = tf.concat([deconv3, conv1], axis=3)
            deconv4, _, _ = deconv2d(tf.nn.relu(input), [self.batch_size, output_size, output_size, 3], k_h=4, k_w=4, name='adv_g_dec4', with_w=True)

            return tf.tanh(deconv4)



    # Generator of cGAN for adversarial patch
    # architecture of G is like image style transfer
    # input: patch image with size n*n
    # output: adversarial patch image with size n*n
    # no gaussian noise needed
    def generator(self, image, reuse=False):
        with tf.variable_scope("generator") as scope:

            # image is 128 x 128 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            # do we need here???
            #image = image / 255.0
            # liuas 2018.5.9
            # trick: using lrelu instead of relu
            self.conv1 = _conv_layer(image, 32, 9, 1, relu=False, name="adv_g_conv1")
            conv2 = _conv_layer(self.conv1, 64, 3, 2, relu=False, name="adv_g_conv2")
            conv3 = _conv_layer(conv2, 128, 3, 2, relu=False, name="adv_g_conv3")
            resid1 = _residual_block(conv3, 3, name="adv_g_resid1")
            resid2 = _residual_block(resid1, 3, name="adv_g_resid2")
            resid3 = _residual_block(resid2, 3, name="adv_g_resid3")
            resid4 = _residual_block(resid3, 3, name="adv_g_resid4")
            resid5 = _residual_block(resid4, 3, name="adv_g_resid5")
            conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2, name="adv_g_deconv1")
            conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2, name="adv_g_deconv2")
            conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False, name="adv_g_deconv3")
            preds = tf.nn.tanh(conv_t3)
            output = image + preds
            # do we need here???
            #return tf.nn.tanh(output) * 127.5 + 255./2
            return tf.nn.tanh(output)


    # target model to attack
    def target_model_discriminator(self, image, reuse = False):
        # here, we call CNN model for GTSRB
        # return GTSRB_Classifier(self.target_model_dir, image)
        # Modify by ZhangAnlan, just build the model of GTSRB, do not restore variables
        fc_layer3, labels_pred, _ = GTSRB_Model(features=image, keep_prob=1.0, reuse=reuse)
        return fc_layer3, labels_pred

    # pad the adversarial patch on image
    def pad_patch_on_image(self, image, patch, if_random=True):
        patched_image = randomly_overlay(image[0], patch[0], if_random=if_random)
        patched_image = tf.expand_dims(patched_image, 0)
        for i in range(1, self.batch_size):
            temp = randomly_overlay(image[i], patch[i], if_random=if_random)
            temp = tf.expand_dims(temp, 0)
            patched_image = tf.concat([patched_image, temp], 0)
        return patched_image

    # show patched images and accuracy
    def show_images_and_acc(self, image, predict_label, real_label, num, filename):
        select_index = np.random.choice(np.arange(int(self.batch_size)), size=num, replace=False)
        select_image = tf.gather(image, select_index)
        # print(predict_label - real_label)
        result = abs(predict_label - real_label)
        select_result = np.take(result, select_index)
        # print(select_result)
        select_result[select_result>0] = 1
        # print(select_result)
        wrong_num = 0
        for i in range(num):
            if select_result[i]!=0:
                wrong_num += 1
        # print(wrong_num)
        acc = float(wrong_num) / float(num)
        plot_images_and_acc(select_image, select_result, acc, num, filename)

    # naive discriminator in GAN
    # using for adversarial training
    def naive_discriminator(self, image, y = None, reuse = False):
        with tf.variable_scope("discriminator") as scope:

            # image is 128 x 128 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image,  self.df_dim, name='adv_d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(layer_norm((conv2d(h0, self.df_dim * 2, name='adv_d_h1_conv')), name="adv_d_ln1"))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(layer_norm(conv2d(h1, self.df_dim * 4, name='adv_d_h2_conv'), name="adv_d_ln2"))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(layer_norm(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name='adv_d_h3_conv'), name="adv_d_ln3"))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'adv_d_h3_lin')

            return tf.nn.sigmoid(h4), h4

    # build cGAN model
    def build_model(self):
        # input image
        self.real_image = tf.placeholder(tf.float32, [self.batch_size, self.image_size,
                                                      self.image_size, self.image_channel])
        # input patch (condition for GAN)
        self.real_patch = tf.placeholder(tf.float32, [self.batch_size, self.patch_size,
                                                      self.patch_size, self.image_channel])
        # original class label of input image
        # dimension correct?? liuas test!!!
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.class_num])
        # adversarial patch generated by G
        self.fake_patch = self.generator_pix2pix(self.real_patch)
        # overlay adversarial patch on image to generate adversarial example
        self.fake_image = self.pad_patch_on_image(image=self.real_image, patch=self.fake_patch, if_random=False)
        # classify result from target model
        self.fake_logits_f, self.fake_prob_f = self.target_model_discriminator(self.fake_image)
        # fake image result from naive D
        self.fake_logits_d, self.fake_prob_d = self.naive_discriminator(self.fake_image)
        # real image result from naive D
        self.real_logits_d, self.real_prob_d = self.naive_discriminator(self.real_image, reuse=True)

        self.target = 40
        self.target_hat = tf.one_hot(self.target, self.class_num)
        self.target_hat_batch = tf.expand_dims(self.target_hat, 0)
        self.target_hat = tf.expand_dims(self.target_hat, 0)
        for i in range(self.batch_size - 1):
            self.target_hat_batch = tf.concat([self.target_hat_batch, self.target_hat], 0)

        # The loss of AdvPGAN consists of: GAN loss, patch loss and adversarial example loss
        # 1.GAN loss for D and G
        self.loss_d_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits_d,
                                                                                 labels=tf.ones_like(self.real_prob_d))+
                                         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits_d,
                                                                                 labels=tf.zeros_like(self.fake_prob_d)))

        self.loss_g_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits_d,
                                                                                 labels=tf.ones_like(self.fake_prob_d)))

        # 2.patch loss
        # pay attention to the first parameter of tv_loss()
        self.patch_loss = tf.nn.l2_loss(self.real_patch - self.fake_patch)# + tv_loss(self.fake_patch - self.real_patch, self.tv_weight)

        # 3.adversarial example loss
        self.ae_loss = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_logits_f, labels=self.y))#+ self.rho * tf.nn.l2_loss(self.real_image - self.fake_image)

        self.temp_loss = self.rho * tf.nn.l2_loss(self.real_image - self.fake_image)

        #self.ae_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_logits_f, labels=self.target_hat_batch))+ self.rho * tf.nn.l2_loss(self.real_image - self.fake_image)
        # 4.gradient penalty
        self.real_data = tf.reshape(self.real_image, [self.batch_size, -1])
        self.fake_data = tf.reshape(self.fake_image, [self.batch_size, -1])
        self.LAMBDA = 10
        self.gra_pen_alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
        self.differences = self.fake_data - self.real_data
        self.interpolates = self.real_data + (self.gra_pen_alpha * self.differences)
        self.interpolates_reshaped = tf.reshape(self.interpolates, [self.batch_size, self.image_size,
                                                                    self.image_size, self.image_channel])
        _, self.results = self.naive_discriminator(self.interpolates_reshaped, reuse=True)
        self.gradients = tf.gradients(self.results, [self.interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.gradient_penalty = self.LAMBDA * tf.reduce_mean((self.slopes - 1.) ** 2)

        # overall loss for D and G
        self.g_loss = self.alpha * self.loss_g_adv + self.beta * self.patch_loss + self.gamma * self.ae_loss + self.delta * self.temp_loss
        self.d_loss = self.loss_d_adv + self.gradient_penalty

        # accuracy for classification rate of target model
        #self.accuracy = self.test_classify(image=self.real_image, imagelabel=self.y, patch=self.real_patch)
        self.predictions = tf.argmax(self.fake_prob_f, 1)
        self.real_label = tf.argmax(self.y, 1)
        self.accuracy = tf.reduce_mean((tf.cast(tf.equal(self.predictions, self.real_label), tf.float32)))

	# get all trainable variables for G and D, respectively
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'adv_d_' in var.name]
        self.g_vars = [var for var in t_vars if 'adv_g_' in var.name]

        # initialize a saver
        self.saver = tf.train.Saver()

    # print list
    def print_each_list(self, list_name):
        for ele in list_name:
            if isinstance(ele, list):
                self.print_each_list(ele)
            else:
                print(ele)

    def print_nan(self, list_name):
        for ele in list_name:
            if isinstance(ele, list):
                self.print_nan(ele)
            else:
                if math.isinf(ele) or math.isnan(ele):
                    print(ele)

    # train cGAN model
    def train_op(self):
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(self.learning_rate, global_step = global_step, decay_steps = 5000, decay_rate=0.95)
        # liuas 2018.5.9 trick: using Adam for G, SGD for D
        d_opt = tf.train.GradientDescentOptimizer(learning_rate= learning_rate). \
            minimize(self.d_loss, var_list=self.d_vars)
        g_opt_ori = tf.train.AdamOptimizer(learning_rate= self.learning_rate)

	add_gloabl = global_step.assign_add(1)
	with tf.control_dependencies([add_gloabl]):
            g_opt = g_opt_ori.minimize(self.g_loss, var_list=self.g_vars)

        # liuas 2018.5.16 separate minimize() into 3 steps to monitor the gradient
        #d_raw_opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        #d_gvs = d_raw_opt.compute_gradients(self.d_loss, var_list=self.d_vars)
        #d_opt = d_raw_opt.apply_gradients(d_gvs)

        #g_raw_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #g_gvs = g_raw_opt.compute_gradients(self.g_loss, var_list=self.g_vars)
        #g_opt = g_raw_opt.apply_gradients(g_gvs)

        ''' test by ZhangAnlan, the result show that the variables of GTSRB are included in the space of global variables
        for var in tf.global_variables():
            print(var.name)
        exit()
        '''

        # note that if we restore GTSRB variables before global_variables_initializer,
        # the GTSRB variables will be reinitialized
        init_op = tf.global_variables_initializer()
        '''
        comment by ZhangAnlan:
        Except self.d_vars and self.g_vars, there are some other variables that need to be initialized
        so I just initialize all the variables at first and then restore the variables of GTSRB
        '''
        # init_op = tf.initialize_variables(self.d_vars+self.g_vars)

        self.sess.run(init_op)

        start_time = time.time()
        counter = 1

        # variables to control model saving
        self.best_acc = 1.0
        self.current_acc = 0.0

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Modify by ZhangAnlan
        # here we restore the variables of GTSRB model
        restore_vars = [var for var in tf.global_variables() if var.name.startswith('GTSRB')]
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess=self.sess, save_path=self.target_model_dir)

        # liuaishan get validation set and train set
        #val_data_x, val_data_y, val_data_z = shuffle_augment_and_load(self.base_image_num, self.valid_img_dir, self.base_patch_num, self.valid_patch_dir, self.batch_size)
        self.train_pair_set = get_initial_image_patch_pair(self.image_all_num, self.patch_all_num)
	valid_pair_set = get_initial_image_patch_pair(self.image_val_num, self.patch_val_num, True)
	
        val_data_x, val_data_y, val_data_z = load_data_in_pair(valid_pair_set, self.batch_size, self.valid_img_dir,self.valid_patch_dir, self.class_num)
        val_data_x = np.array(val_data_x).astype(np.float32)
        val_data_y = np.array(val_data_y).astype(np.float32)
        val_data_z = np.array(val_data_z).astype(np.float32)
	print(self.sess.run(learning_rate))

        for epoch in range(self.epoch):
            batch_iteration = self.image_all_num * self.patch_all_num / self.batch_size

            for id in range(int(batch_iteration)):

                #batch_data_x, batch_data_y, batch_data_z  = \
                #    shuffle_augment_and_load(self.base_image_num, self.image_dir, self.base_patch_num,
                #                             self.patch_dir, self.batch_size )

		batch_data_x, batch_data_y, batch_data_z  = load_data_in_pair(self.train_pair_set, self.batch_size, self.image_dir,self.patch_dir, self.class_num)
                batch_data_x = np.array(batch_data_x).astype(np.float32)
                batch_data_y = np.array(batch_data_y).astype(np.float32)
                batch_data_z = np.array(batch_data_z).astype(np.float32)

                # liuas 2018.5.7 trick: we train G once while D d_train_freq times in one iteration
                #if (id + 1) % self.d_train_freq == 0:
                self.sess.run([g_opt], feed_dict={self.real_image: batch_data_x,self.y: batch_data_y,self.real_patch: batch_data_z})

                self.sess.run([d_opt],
                              feed_dict={self.real_image: batch_data_x,
                                         self.y: batch_data_y,
                                         self.real_patch: batch_data_z})

                counter += 1

                # test the accuracy
                # liuas 2018.5.9 validation

                if np.mod(counter, 40) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, id, batch_iteration, time.time() - start_time))
                    print("[Validation].......")

                    print("learning_rate: %.8f" % self.sess.run(learning_rate)
			  
                    # test! Show logits and probs when validating
                    print("[top 3 fake_logits].......")
                    current_fake_logits = self.fake_logits_f.eval({self.real_image: val_data_x, self.real_patch: val_data_z})
                    for i in range(len(current_fake_logits)):
			top1 = np.argsort(current_fake_logits[i])[-1]
			top2 = np.argsort(current_fake_logits[i])[-2]
			top3 = np.argsort(current_fake_logits[i])[-3]
                        print("%d %.8f %d %.8f %d %.8f" %(top1, current_fake_logits[i][top1], top2, current_fake_logits[i][top2], top3, current_fake_logits[i][top3]))
                    #print(current_fake_logits)
                    print("[top 3 fake_prob].......")
                    current_fake_prob = self.fake_prob_f.eval({self.real_image: val_data_x, self.real_patch: val_data_z})
                    for i in range(len(current_fake_prob)):
			top1 = np.argsort(current_fake_prob[i])[-1]
			top2 = np.argsort(current_fake_prob[i])[-2]
			top3 = np.argsort(current_fake_prob[i])[-3]
                        print("%d %.8f %d %.8f %d %.8f" %(top1, current_fake_prob[i][top1], top2, current_fake_prob[i][top2], top3, current_fake_prob[i][top3]))

                    errAE = self.ae_loss.eval({self.real_image: val_data_x,
                        self.y: val_data_y,
                        self.real_patch: val_data_z})

                    errD = self.d_loss.eval({self.real_image: val_data_x,
                                             self.y: val_data_y,
                                             self.real_patch: val_data_z})

                    errG = self.g_loss.eval({self.real_image: val_data_x,
                                             self.y: val_data_y,
                                             self.real_patch: val_data_z})

                    acc = self.accuracy.eval({self.real_image: val_data_x,
                                              self.y: val_data_y,
                                              self.real_patch: val_data_z})

                    errTemp = self.temp_loss.eval({self.real_image: val_data_x,
                        self.y: val_data_y,
                        self.real_patch: val_data_z})

                    print("g_loss: %.8f , d_loss: %.8f" % (errG, errD))
                    print("Accuracy of classification: %4.4f" % acc)
                    
                    acc_batch = self.accuracy.eval({self.real_image: batch_data_x,
                                              self.y: batch_data_y,
                                              self.real_patch: batch_data_z})
                    print("train batch acc: %4.4f" % acc_batch)

                    print("ae_loss: %.8f" %errAE)
                    
                    print("temp_loss: %.8f" %errTemp)

                    print("sum_loss: %.8f" %(errAE+errTemp))
                    if acc < self.best_acc:
                        self.best_acc = acc
                        print("Saving model.")
                        self.save(self.checkpoint_dir, counter)
                    print("current acc: %.4f, best acc: %.4f" % (acc, self.best_acc))
		
                # liuas 2018.5.10 test
                if np.mod(counter, 1000) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, id, batch_iteration, time.time() - start_time))

                    # accuracy in the test set 2018.5.10 ZhangAnlan
                    print("[Test].......")

                    batch_data_x, batch_data_y, batch_data_z = \
                        shuffle_augment_and_load(self.base_image_num, self.test_img_dir, self.base_patch_num,
                                                 self.patch_dir, self.batch_size)

                    batch_data_x = np.array(batch_data_x).astype(np.float32)
                    batch_data_y = np.array(batch_data_y).astype(np.float32)
                    batch_data_z = np.array(batch_data_z).astype(np.float32)

                    errD, errG, acc, fake_image, predictions, real_label = \
                        self.sess.run([self.d_loss, self.g_loss, self.accuracy, self.fake_image, self.predictions, self.real_label],
                                      feed_dict={self.real_image: batch_data_x,
                                                 self.y: batch_data_y,
                                                 self.real_patch: batch_data_z})
                    print("g_loss: %.8f , d_loss: %.8f" % (errG, errD))
                    print("Accuracy of classification: %4.4f" % acc)

                    # plot accuracy
                    self.acc_history.append(float(acc))
                    '''
                    plot_acc(self.acc_history, filename=self.output_dir+'/' +'Accrucy.png')

                    # save patches
                    save_patches(self.fake_patch.eval({self.real_image: batch_data_x,
                                                         self.y: batch_data_y,
                                                 self.real_patch: batch_data_z}),
                             filename=self.output_dir+'/' + str(time.time()) +'_fake_patches.png')
                    # show images and acc
                    self.show_images_and_acc(fake_image, predictions, real_label, num=9,
                                             filename=self.output_dir+'/' + str(time.time()) +'_fake_images.png')

                    '''

    # test Generator
    def test_op(self):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit()
        self.test_patch()
        # todo show accuracy of image with input_patch

    # save model
    def save(self, checkpoint_dir, step):
        model_name = "AdvPGAN.model"
        model_dir = "%s_%s" % (self.image_size, self.patch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    # load model
    def load(self, checkpoint_dir):
        print("Reading checkpoint...")

        model_dir = "%s_%s" % (self.image_size, self.patch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    # Add 2018.5.12 ZhangAnlan
    # update 2018.5.15 ZhangAnlan
    # test patch
    # requires: none
    # modifies: none
    # effects: save the plot patches to filename_ori.png and filename_gen.png
    # update 2018.5.18 FanJiaXin
    # add: save image_with_gen_patch and patch_dif
    # todo: let GTSRB model successfully recognizes image_with_ori_patch
    def test_patch(self):
        restore_vars = [var for var in tf.global_variables() if var.name.startswith('GTSRB')]
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess=self.sess, save_path=self.target_model_dir)
        
        image_with_ori_patch = self.pad_patch_on_image(image=self.real_image, patch=self.real_patch, if_random=False)
        _, ori_prob_f = self.target_model_discriminator(image_with_ori_patch, reuse=True)
        predictions_ori = tf.argmax(ori_prob_f, 1)
        acc_ori = tf.reduce_mean((tf.cast(tf.equal(predictions_ori, self.real_label), tf.float32)))
        
        test_pair_set = get_initial_image_patch_pair(8, 8)
        for i in range(1):
            # batch_data_x, batch_data_y, batch_data_z = \
            #     shuffle_augment_and_load(self.batch_size, self.test_img_dir, self.batch_size,
            #                              self.valid_patch_dir, self.batch_size)
            batch_data_x, batch_data_y, batch_data_z = load_data_in_pair(test_pair_set, self.batch_size, self.valid_img_dir,self.valid_patch_dir, self.class_num)
            batch_data_x = np.array(batch_data_x).astype(np.float32)
            batch_data_y = np.array(batch_data_y).astype(np.float32)
            batch_data_z = np.array(batch_data_z).astype(np.float32)
            
            real_patch, fake_patch, fake_image, predictions, real_label, acc= \
                self.sess.run([self.real_patch, self.fake_patch, self.fake_image, self.predictions, self.real_label, self.accuracy],
                              feed_dict={self.real_image: batch_data_x,
                                         self.y: batch_data_y,
                                         self.real_patch: batch_data_z})
            print("Saving test results: %d / 20." % (i+1))
            print(acc)
            print(predictions)
            # show images with gen patch and their acc
            self.show_images_and_acc(fake_image, predictions, real_label, num=9,
                                     filename='../test/image_with_gen_patch_' + str(i) + '.png')
            
            self.show_images_and_acc(image_with_ori_patch.eval({self.real_image: batch_data_x,
                                          self.real_patch: batch_data_z}),
                                     predictions_ori.eval({self.real_image: batch_data_x,
                                          self.real_patch: batch_data_z}),
                                     real_label, num=9,
                                     filename='../test/image_with_ori_patch_' + str(i) + '.png')
            print(acc_ori.eval({self.real_image: batch_data_x,
                               self.y: batch_data_y,
                               self.real_patch: batch_data_z}))
            print(predictions_ori.eval({self.real_image: batch_data_x,
                                          self.real_patch: batch_data_z}))
            
            # Avoid negative value
            diff_patch = tf.abs(fake_patch - real_patch)
            # show original patch, generated patch and difference betwween them
            save_patches(real_patch, '../test/patch_ori_'+ str(i) + '.png')
            save_patches(fake_patch, '../test/patch_gen_'+ str(i) + '.png')
            save_patches(diff_patch, '../test/patch_dif_'+ str(i) + '.png')

        exit()




