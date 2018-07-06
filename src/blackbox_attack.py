from VGG16_Classifier import VGG16_Model
from VGG16_Classifier import load_image
import tensorflow as tf
from keras import backend as K
from keras.models import Model

from GTSRB_Classifier import GTSRB_Model
from utils import OHE_labels
from utils import pre_process_image

import pickle
import numpy as np

def GTSRB_predict():
	# test_data_dir = '/home/zhenxt/zal/GTSRB/data/validation_img_1_8.p'
	# test_data_dir = '/home/zhenxt/zal/AdvPGAN/test/fake_image.p'
	test_data_dir = '/home/zhenxt/zal/AdvPGAN/test/real_image.p'
	weights_dir = '/home/zhenxt/zal/GTSRB/test/GTSRB_best_test'

	img = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
	y = tf.placeholder(tf.float32, shape=[None, 43])
	logits, probs, _ = GTSRB_Model(features=img, keep_prob=1.0, reuse=False)
	predictions = tf.argmax(probs, 1)
	real_labels = tf.argmax(y, 1)
	accuracy = tf.reduce_mean((tf.cast(tf.equal(predictions, real_labels), tf.float32)))

	with open(test_data_dir, 'rb') as f:
		dataset = pickle.load(f)
	images = dataset['data']
	labels = OHE_labels(dataset['labels'], 43)

	#test = np.asarray([item/255. for item in images]).astype(np.float32)
        test = images.astype(np.float32)
	test_labels = labels.astype(np.float32)

	init_op = tf.global_variables_initializer()

	restore_vars = [var for var in tf.global_variables() if var.name.startswith('GTSRB')]
	saver = tf.train.Saver(restore_vars)

	with tf.Session() as sess:
		sess.run(init_op)
		saver.restore(sess=sess, save_path=weights_dir)
		print(predictions.eval({img: test, y: test_labels}))
		print(accuracy.eval({img: test, y: test_labels}))


    

def VGG16_predict():
	# test_data_dir = '/home/zhenxt/zal/GTSRB/data/validation_img_1_8.p'
	test_data_dir = '/home/zhenxt/zal/AdvPGAN/test/fake_image.p'
	# test_data_dir = '/home/zhenxt/zal/AdvPGAN/test/real_image.p'
	weights_dir = '/home/zhenxt/zal/GTSRB/vgg16/vgg16_training10000.h5'
	K.set_learning_phase(False)

	img = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
	y = tf.placeholder(tf.float32, shape=[None, 43])
	model = VGG16_Model(128, 128, False)
	probs = model(img)
	predictions = tf.argmax(probs, 1)
	real_labels = tf.argmax(y, 1)
	accuracy = tf.reduce_mean((tf.cast(tf.equal(predictions, real_labels), tf.float32)))

	test, test_labels = load_image(test_data_dir, 43, True, False)

	init_op = tf.global_variables_initializer()

	with tf.Session() as sess:
		K.set_session(sess)
		sess.run(init_op)
		model.load_weights(weights_dir)
		print(predictions.eval({img: test, y: test_labels}))
		print(accuracy.eval({img: test, y: test_labels}))

if __name__ == '__main__':
	# GTSRB_predict()
	VGG16_predict()
