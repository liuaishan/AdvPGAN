import argparse
import os
import scipy.misc
import numpy as np

from model import AdvPGAN
import tensorflow as tf

# hyperparameter
flags = tf.app.flags
flags.DEFINE_integer("epoch", 10000, "epoch number to train")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("traindata_size", 10000, "train set size")
flags.DEFINE_integer("patch_size", 16, "patch size")
flags.DEFINE_integer("channel", 3, "channel number")
flags.DEFINE_integer("image_size", 128, "size of image")
flags.DEFINE_float("alpha", 1.0, "parameter alpha")
flags.DEFINE_float("beta", 1.0, "parameter beta")
flags.DEFINE_float("gamma", 1.0, "parameter gamma")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_float("base_image_num", 4, "base number of image to augment")
flags.DEFINE_float("base_patch_num", 4, "base number of patch to augment")
# flags.DEFINE_string("target_model_dir", "C:\\Users\\SEELE\\Desktop\\AdvGAN\\AdvPGAN\\data\\GTSRB\\model_best_test", "data directory")
flags.DEFINE_string("target_model_dir", "/home/dsg/liuas/AnlanZhang/GTSRB/model_new/test_model_2/model_best_test", "data directory")
# flags.DEFINE_string("checkpoint_dir", "..\\checkpoint", "directory to save model")
flags.DEFINE_string("checkpoint_dir", "../checkpoint", "directory to save model")
flags.DEFINE_string("phase", "train", "current phase of model, e.g to train or test")
# flags.DEFINE_string("output_dir", "..\\output", "directory to save serialized image")
flags.DEFINE_string("output_dir", "../output", "directory to save serialized image")
FLAGS = tf.app.flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # liuas 2018.5.9 try to avoid abort error
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.InteractiveSession()
    #with sess:
    model = AdvPGAN(sess, batch_size=FLAGS.batch_size, image_size=FLAGS.image_size,
                    patch_size=FLAGS.patch_size, channel=FLAGS.channel,
                    alpha=FLAGS.alpha, beta=FLAGS.beta, gamma=FLAGS.gamma,
                    learning_rate=FLAGS.learning_rate, epoch=FLAGS.epoch,
                    traindata_size=FLAGS.traindata_size,
                    base_image_num=FLAGS.base_image_num,
                    base_patch_num=FLAGS.base_patch_num,
                    target_model_dir= FLAGS.target_model_dir,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    output_dir=FLAGS.output_dir)
    # 2018.5.12 ZhangAnlan
    # test patch
    # model.test_patch(FLAGS.batch_size, 'test_patch')

    if FLAGS.phase == 'train':
        model.train_op()
    else:
        model.test_op()
    sess.close()

if __name__ == '__main__':
    tf.app.run()