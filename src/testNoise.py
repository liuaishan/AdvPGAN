###################################################
#   Author: Liu Aishan                            #
#   Date: 2018.3.26                               #
#   Create adversarial examples under             #
#   different image transformation distributions  #
###################################################


import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import PIL
import numpy as np
import os
import tarfile
import matplotlib.pyplot as plt
import json


sess = tf.InteractiveSession()

image = tf.Variable(tf.zeros((299, 299, 3)))

log_dir = 'inception_v3_log'
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

# inception_v3 model
def inception(image, reuse=False):
	preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
	arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
	with slim.arg_scope(arg_scope):
		logits,_ = 	nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
		logits = logits[:, 1:]
		probs = tf.nn.softmax(logits)
	return logits, probs

logits, probs = inception(image)

data_dir = '../data'
img_path = os.path.join(data_dir, 'girl.jpeg')

restore_vars =[ var for var in tf.global_variables() if var.name.startswith('InceptionV3/')]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir,'inception_v3.ckpt'))

imagenet_json = os.path.join(data_dir,"imagenet.json")
with open(imagenet_json) as f:
	imagenet_labels = json.load(f)

# classify a image into its class and show result
def classify(img, correct_class=None, target_class=None):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
	fig.sca(ax1)
	p = sess.run(probs, feed_dict={image: img})[0]
	ax1.imshow(img)
	fig.sca(ax1)

	topk = list(p.argsort())[-10:][::-1]
	topprobs = p[topk]
	barlist = ax2.bar(range(10), topprobs)
	if target_class in topk:
		barlist[topk.index(target_class)].set_color('r')
	if correct_class in topk:
		barlist[topk.index(correct_class)].set_color('g')
	plt.sca(ax2)
	plt.ylim([0, 1.1])
	plt.xticks(range(10), [imagenet_labels[i][:15] for i in topk],
				rotation='vertical')
	fig.subplots_adjust(bottom=0.2)
	plt.show()

data_dir = '../data'
img_path = os.path.join(data_dir, 'girl.jpeg')
# image preprocessing
# PIL seems have sth wrong with TF when add tf.image.adjust_brightness()
img_class = 281
img = PIL.Image.open(img_path)
#img.width = img.size[0]
#img.height = img.size[1]
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 /  img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32)

#classify(img, correct_class=img_class)


x = tf.placeholder(tf.float32, (299, 299, 3))

# trainable image to get adversarial example
x_hat = image
assign_op = tf.assign(x_hat, x)

learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())

labels = tf.one_hot(y_hat, 1000)

# get a adversarial example under different image transformation distributions

num_samples = 4 # samples number needed to be increased when GPU is available
average_loss = 0


for i in range(num_samples):
	noise_img = tf.add(image, tf.clip_by_value(tf.random_normal(shape=image.get_shape(), mean=0, stddev=1), 0, 0.25))
	noise_img = tf.clip_by_value(noise_img, 0, 1)
	noise_logits, _ = inception(noise_img, reuse = True)
	average_loss += tf.nn.softmax_cross_entropy_with_logits(logits=noise_logits, labels = labels) / num_samples

# optimizer
optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(average_loss, var_list=[x_hat])

epsilon = tf.placeholder(tf.float32, ())

# clip need to be done to make sure noise 'imperceptible' to human eyes
below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)

# training process
demo_epsilon2 = 8.0/255.0
demo_lr2 = 2e-1
demo_steps2 = 100 # need more than 100 steps
demo_target2 = 924

sess.run(assign_op, feed_dict={x:img})

for i in range(demo_steps2):
	_, loss_value = sess.run([optim_step, average_loss], feed_dict={learning_rate: demo_lr2, y_hat: demo_target2})

	sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon2})
	if (i+1) % 10 == 0 :
		print(" step %d, loss %g" %(i+1, loss_value))

adv_robust = x_hat.eval()

for i in range(0, 10):
	noise_image = tf.add(image, tf.clip_by_value(tf.random_normal(shape=image.get_shape(), mean=0, stddev=1), 0, 0.25))
	noise_image = tf.clip_by_value(noise_image, 0, 1)
	noise_ex = noise_image.eval(feed_dict={image: adv_robust})
	classify(noise_ex, correct_class=img_class, target_class=924)
