###################################################
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
from ops import linear
from ops import conv2d
from utils import save_obj
from utils import save_patches
from utils import plot_acc
from utils import randomly_overlay
from utils import shuffle_augment_and_load
from utils import plot_images_and_acc
from utils import load_image
from GTSRB_Classifier import GTSRB_Classifier
from GTSRB_Classifier import GTSRB_Model

import os
import time
import glob
import numpy as np
from scipy import misc



# TODO
# 1.refactor, remove similar functions like: conv2d, _conv_layer
# 2.overlay adversarial patch with different environment distributions
# 3.loss function(check again!!!)


class AdvPGAN(object):

    def __init__(self, sess, batch_size=16, image_size=128, patch_size=32,
                 channel=3, alpha=1, beta=1, gamma=1, learning_rate=0.0001,
                 epoch=10000, traindata_size=10000,
                 base_image_num = 16, base_patch_num = 16,
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
        self.rho = 1
        self.d_train_freq = 5
        self.image_dir = '/home/dsg/liuas/AnlanZhang/GTSRB/TrafficSignData/train.p'
        self.test_img_dir = '/home/dsg/liuas/AnlanZhang/GTSRB/TrafficSignData/high_resolution_img.p'
        self.patch_dir = '/home/dsg/liuas/AnlanZhang/GTSRB/cifar-10-resized/data_batch_1'
        self.base_image_num = base_image_num
        self.base_patch_num = base_patch_num
        self.acc_history = [] 

        self.build_model()


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
        fc_layer3, labels_pred, _ = GTSRB_Model(features=image, keep_prob=1.0)
        return fc_layer3, labels_pred

    # pad the adversarial patch on image
    def pad_patch_on_image(self, image, patch):
        patched_image = randomly_overlay(image[0], patch[0])
        patched_image = tf.expand_dims(patched_image, 0)
        for i in range(1, self.batch_size):
            temp = randomly_overlay(image[i], patch[i])
            temp = tf.expand_dims(temp, 0)
            patched_image = tf.concat([patched_image, temp], 0)
        return patched_image

    # show patched images and accuracy
    def show_images_and_acc(self, image, predictions, acc, num, filename):
        select_index = np.random.choice(np.arange(int(self.batch_size)), size=num, replace=False)
        select_image = tf.gather(image, select_index)
        result = tf.not_equal(predictions, self.real_label)
        select_result = tf.gather(result, select_index)
        # acc = tf.cast(tf.count_nonzero(select_result), tf.float32)/float(num)
        plot_images_and_acc(select_image, select_result, acc, num, filename)

    # Add 2018.5.12 ZhangAnlan
    # test patch
    # requires: num is a square number
    # modifies: none
    # effects: save the plot patches to filename_ori.png and filename_gen.png
    def test_patch(self, num, filename):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit()
        patch, label = load_image(self.batch_size, self.patch_dir, 10)
        fake_patch = self.fake_patch.eval({self.real_patch: patch})
        save_patches(patch, self.output_dir + '/' + filename + '_ori.png')
        save_patches(fake_patch, self.output_dir + '/' + filename + '_gen.png')
        exit()

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
            h1 = lrelu(batch_norm((conv2d(h0, self.df_dim * 2, name='adv_d_h1_conv')), name="adv_d_bn1"))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name='adv_d_h2_conv'), name="adv_d_bn2"))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name='adv_d_h3_conv'), name="adv_d_bn3"))
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
        self.fake_patch = self.generator(self.real_patch)
        # overlay adversarial patch on image to generate adversarial example
        self.fake_image = self.pad_patch_on_image(image=self.real_image, patch=self.fake_patch)
        # classify result from target model
        self.fake_logits_f, self.fake_prob_f = self.target_model_discriminator(self.fake_image)
        # fake image result from naive D
        self.fake_logits_d, self.fake_prob_d = self.naive_discriminator(self.fake_image)
        # real image result from naive D
        self.real_logits_d, self.real_prob_d = self.naive_discriminator(self.real_image, reuse=True)


        # The loss of AdvPGAN consists of: GAN loss, patch loss and adversarial example loss
        # 1.GAN loss for D and G
        self.loss_d_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits_d,
                                                                                 labels=tf.ones_like(self.real_logits_d))+
                                         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits_d,
                                                                                 labels=tf.zeros_like(self.fake_logits_d)))

        self.loss_g_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits_d,
                                                                                 labels=tf.ones_like(self.fake_logits_d)))

        # 2.patch loss
        # todo TV(), try L1 norm etc.
        self.patch_loss = tf.nn.l2_loss(self.real_patch - self.fake_patch)

        # 3.adversarial example loss
        self.ae_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits_f, labels=self.y)) \
                                       + self.rho * tf.nn.l2_loss(self.real_image - self.fake_image)

        # overall loss for D and G
        self.g_loss = self.alpha * self.loss_g_adv + self.beta * self.patch_loss + self.gamma * self.ae_loss
        self.d_loss = self.loss_d_adv

        # accuracy for misclassification rate of target model
        #self.accuracy = self.test_classify(image=self.real_image, imagelabel=self.y, patch=self.real_patch)
        self.predictions = tf.argmax(self.fake_prob_f, 1)
        self.real_label = tf.argmax(self.y, 1)
        self.accuracy = tf.reduce_mean((tf.cast(tf.not_equal(self.predictions, self.real_label), tf.float32)))

        # load target model
        # restore_vars = [var for var in tf.global_variables() if var.name.startswith('adv_')]
        # saver = tf.train.Saver(restore_vars)
        # saver.restore(self.sess, os.path.join(self.data_dir, 'AdvpGAN.ckpt'))

        # get all trainable variables for G and D, respectively
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'adv_d_' in var.name]
        self.g_vars = [var for var in t_vars if 'adv_g_' in var.name]

        # initialize a saver
        self.saver = tf.train.Saver()

       
    # train cGAN model
    def train_op(self):

        # liuas 2018.5.9 trick: using Adam for G, SGD for D
        # d_opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate). \
        #    minimize(self.d_loss, var_list=self.d_vars)
        # g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
        #    minimize(self.g_loss, var_list=self.g_vars)

        # liuas 2018.5.16 separate minimize() into 3 steps to monitor the gradient
        d_raw_opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        d_gvs = d_raw_opt.compute_gradients(self.d_loss, var_list=self.d_vars)
        d_opt = d_raw_opt.apply_gradients(d_gvs)

        g_raw_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        g_gvs = g_raw_opt.compute_gradients(self.g_loss, var_list=self.g_vars)
        g_opt = g_raw_opt.apply_gradients(g_gvs)

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
        

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Modify by ZhangAnlan
        # here we restore the variables of GTSRB model
        restore_vars = [var for var in tf.global_variables() if var.name.startswith('GTSRB')]
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess=self.sess, save_path=self.target_model_dir)

        for epoch in range(self.epoch):

            batch_iteration = self.traindata_size / self.batch_size

            for id in range(int(batch_iteration)):

                batch_data_x, batch_data_y, batch_data_z  = \
                    shuffle_augment_and_load(self.base_image_num, self.image_dir, self.base_patch_num,
                                             self.patch_dir, self.batch_size )

                batch_data_x = np.array(batch_data_x).astype(np.float32)
                batch_data_y = np.array(batch_data_y).astype(np.float32)
                batch_data_z = np.array(batch_data_z).astype(np.float32)


                '''test by ZhangAnlan, printf the predictions, real_label and accuracy for 20 batches
                pre = self.predictions.eval({self.real_image: batch_data_x,
                                        self.y: batch_data_y,
                                        self.real_patch: batch_data_z})
                print("predictions:")
                print(pre)
                real_l = self.real_label.eval({self.real_image: batch_data_x,
                                            self.y: batch_data_y,
                                            self.real_patch: batch_data_z})
                print("real_label:")
                print(real_l)
                acc = self.accuracy.eval({self.real_image: batch_data_x,
                                        self.y: batch_data_y,
                                         self.real_patch: batch_data_z})
                print("Accuracy of misclassification: %4.4f" %acc)
                if(id == 20):
                    exit()
                continue
                '''

                # liuas 2018.5.7 trick: we train G once while D d_train_freq times in one iteration
                if (id + 1) % self.d_train_freq == 0:
                    self.sess.run([g_opt],
                                  feed_dict={self.real_image: batch_data_x,
                                             self.y: batch_data_y,
                                             self.real_patch: batch_data_z})

                self.sess.run([d_opt],
                              feed_dict={self.real_image: batch_data_x,
                                         self.y: batch_data_y,
                                         self.real_patch: batch_data_z})

                counter += 1

                # test the accuracy
                # liuas 2018.5.9 validation
                if np.mod(counter, 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, id, batch_iteration, time.time() - start_time))
                    print("[Validation].......")

                    batch_data_x, batch_data_y, batch_data_z = \
                        shuffle_augment_and_load(self.base_image_num, self.image_dir, self.base_patch_num,
                                                 self.patch_dir, self.batch_size)

                    batch_data_x = np.array(batch_data_x).astype(np.float32)
                    batch_data_y = np.array(batch_data_y).astype(np.float32)
                    batch_data_z = np.array(batch_data_z).astype(np.float32)

                    errD = self.d_loss.eval({self.real_image: batch_data_x,
                                             self.y: batch_data_y,
                                             self.real_patch: batch_data_z})

                    errG = self.g_loss.eval({self.real_image: batch_data_x,
                                             self.y: batch_data_y,
                                             self.real_patch: batch_data_z})

                    acc = self.accuracy.eval({self.real_image: batch_data_x,
                                              self.y: batch_data_y,
                                              self.real_patch: batch_data_z})
                    print("g_loss: %.8f , d_loss: %.8f" % (errG, errD))

                    '''
                    print(self.predictions.eval({self.real_image: batch_data_x,
                                         self.y: batch_data_y,
                                         self.real_patch: batch_data_z}))
                    print(self.real_label.eval({self.real_image: batch_data_x,
                                         self.y: batch_data_y,
                                         self.real_patch: batch_data_z}))
                    '''
                    print("Accuracy of misclassification: %4.4f" % acc)

                    # here, we have to check the gradients to avoid vanish or explosion
                    print("[Gradients of G].......")
                    for grad, var in g_gvs:
                        print(var.name)
                        print(grad.eval({self.real_image: batch_data_x,
                                              self.y: batch_data_y,
                                              self.real_patch: batch_data_z}))

                    print("[Gradients of D].......")
                    for grad, var in d_gvs:
                        print(var.name)
                        print(grad.eval({self.real_image: batch_data_x,
                                         self.y: batch_data_y,
                                         self.real_patch: batch_data_z}))

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

                    errD = self.d_loss.eval({self.real_image: batch_data_x,
                                             self.y: batch_data_y,
                                             self.real_patch: batch_data_z})

                    errG = self.g_loss.eval({self.real_image: batch_data_x,
                                             self.y: batch_data_y,
                                             self.real_patch: batch_data_z})

                    acc = self.accuracy.eval({self.real_image: batch_data_x,
                                              self.y: batch_data_y,
                                              self.real_patch: batch_data_z})
                    print("g_loss: %.8f , d_loss: %.8f" % (errG, errD))
                    print("Accuracy of misclassification: %4.4f" % acc)

                    # plot accuracy
                    self.acc_history.append(float(acc))
                    plot_acc(self.acc_history, filename=self.output_dir+'/' +'Accrucy.png')
                    # save patches
                    save_patches(self.fake_patch.eval({self.real_image: batch_data_x,
                                                   self.fake_patch: batch_data_z}),
                             filename=self.output_dir+'/' + str(time.time()) +'_fake_patches.png')
                    # show images and acc
                    self.show_images_and_acc(self.fake_image.eval({self.real_image: batch_data_x,
                                                   self.fake_patch: batch_data_z}),
                             self.predictions.eval({self.real_image: batch_data_x,
                                                   self.fake_patch: batch_data_z}),
                             acc, 9, filename=self.output_dir+'/' + str(time.time()) +'_fake_images.png')

                # save model
                if np.mod(counter, 500) == 0:
                    self.save(self.checkpoint_dir, counter)

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




