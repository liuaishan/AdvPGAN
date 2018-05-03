import argparse
import os
import scipy.misc
import numpy as np

from src.model import AdvPGAN
import tensorflow as tf

# hyperparameter
flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "epoch number to train")
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("traindata_size", 1e8, "train set size")
flags.DEFINE_integer("patch_size", 28, "patch size")
flags.DEFINE_integer("channel", 3, "channel number")
flags.DEFINE_integer("image_size", 256, "size of image")
flags.DEFINE_float("alpha", 1.0, "parameter alpha")
flags.DEFINE_float("beta", 1.0, "parameter beta")
flags.DEFINE_float("gamma", 1.0, "parameter gamma")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_float("base_image_num", 4, "base number of image to augment")
flags.DEFINE_float("base_patch_num", 4, "base number of patch to augment")
flags.DEFINE_string("data_dir", "..\\data\\model_1\\model_best_test.ckpt", "data directory")
flags.DEFINE_string("target_model_dir", "..\\data", "data directory")
flags.DEFINE_string("checkpoint_dir", "../checkpoint", "directory to save model")
flags.DEFINE_string("phase", "train", "current phase of model, e.g to train or test")
flags.DEFINE_string("output_dir", "../output", "directory to save serialized image")
FLAGS = tf.app.flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    sess = tf.InteractiveSession()
    #with tf.Session() as sess:
    model = AdvPGAN(sess, batch_size=FLAGS.batch_size, image_size=FLAGS.image_size,
                    patch_size=FLAGS.patch_size, channel=FLAGS.channel,
                    alpha=FLAGS.alpha, beta=FLAGS.beta, gamma=FLAGS.gamma,
                    learning_rate=FLAGS.learning_rate, epoch=FLAGS.epoch,
                    traindata_size=FLAGS.traindata_size,
                    base_image_num=FLAGS.base_image_num,
                    base_patch_num=FLAGS.base_patch_num,
                    data_dir=FLAGS.data_dir, target_model_dir= FLAGS.target_model_dir,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    output_dir=FLAGS.output_dir)

    if FLAGS.phase == 'train':
        model.train_op()
    else:
        model.train_op()
    sess.close()

if __name__ == '__main__':
    tf.app.run()
