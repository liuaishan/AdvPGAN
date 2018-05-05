import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import misc

with tf.Session() as sess:
	image = tf.read_file('../data/cat.jpg')
	image = tf.image.decode_jpeg(image, channels=3)	
	image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.image.resize_images(image, (128,128))

	patch = tf.read_file('../data/patch.jpg')
	patch = tf.image.decode_jpeg(patch, channels=3)
	patch = tf.image.convert_image_dtype(patch, tf.float32)
	patch = tf.image.resize_images(patch, (28,28))
	patch_mask = np.ones([patch.shape[0],patch.shape[0],3], dtype=np.float32)
	patch_mask = tf.convert_to_tensor(patch_mask)
	
	patch_size = int(patch.shape[0])*1.5
	patch = tf.image.resize_image_with_crop_or_pad(patch, int(patch_size), int(patch_size))
	patch_mask = tf.image.resize_image_with_crop_or_pad(patch_mask, int(patch_size), int(patch_size))

	angle = np.random.uniform(low=-180.0, high=180.0)

	def random_rotate_image_func(image, angle):
		return misc.imrotate(image, angle, 'bicubic') 

	patch_rotate = tf.py_func(random_rotate_image_func, [patch, angle], tf.uint8)
	patch_mask = tf.py_func(random_rotate_image_func, [patch_mask, angle], tf.uint8)
	patch_rotate = tf.image.convert_image_dtype(patch_rotate, tf.float32)
	patch_mask = tf.image.convert_image_dtype(patch_mask, tf.float32)

	location_x = int(np.random.uniform(low=0, high=int(image.shape[0])-patch_size))
	location_y = int(np.random.uniform(low=0, high=int(image.shape[0])-patch_size))
	patch_rotate = tf.image.pad_to_bounding_box(patch_rotate, location_y, location_x, int(image.shape[0]), int(image.shape[0]))
	patch_mask = tf.image.pad_to_bounding_box(patch_mask, location_y, location_x, int(image.shape[0]), int(image.shape[0]))
	image_with_patch = (1-patch_mask)*image + patch_rotate


	plt.figure()
	plt.subplot(1,3,1)
	plt.imshow(image.eval())
	plt.subplot(1,3,2)
	plt.imshow(patch_rotate.eval())
	plt.subplot(1,3,3)
	plt.imshow(image_with_patch.eval())
	plt.show()