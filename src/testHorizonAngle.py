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

def _transform_vector(width, x_shift, y_shift, im_scale, x_scale, y_add):
    """
     If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
     then it maps the output point (x, y) to a transformed input point 
     (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
     where k = c0 x + c1 y + 1.  
     The transforms are inverted compared to the transform mapping input points to output points.
    """
    
    # Standard rotation matrix
    # (use negative rot because tf.contrib.image.transform will do the inverse)
    # fix y
    rot_matrix = np.array(
        [[x_scale, y_add],
        [0, 1]]
    )
    
    # Scale it
    # (use inverse scale because tf.contrib.image.transform will do the inverse)
    inv_scale = 1. / im_scale 
    xform_matrix = rot_matrix * inv_scale
    a0, a1 = xform_matrix[0]
    b0, b1 = xform_matrix[1]
    
    # At this point, the image will have been rotated around the top left corner,
    # rather than around the center of the image. 
    #
    # To fix this, we will see where the center of the image got sent by our transform,
    # and then undo that as part of the translation we apply.
    x_origin = float(width) / 2
    y_origin = float(width) / 2
    
    x_origin_shifted, y_origin_shifted = np.matmul(
        xform_matrix,
        np.array([x_origin, y_origin]),
    )
 
    x_origin_delta = x_origin - x_origin_shifted
    y_origin_delta = y_origin - y_origin_shifted
    
    # Combine our desired shifts with the rotation-induced undesirable shift
    a2 = x_origin_delta - (x_shift/(2*im_scale))
    b2 = y_origin_delta - (y_shift/(2*im_scale))
      
    # Return these values in the order that tf.contrib.image.transform expects
    return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)

def test_random_transform(image_in, min_scale=0.5, max_scale=1.0, max_xscale=4.0, max_yadd=1.0):
    """
    Scales the image between min_scale and max_scale
    """
    img_shape = [299,299,3]
    
    width = img_shape[0]
    
    def _random_transformation():
        im_scale = np.random.uniform(low=min_scale, high=1.0)
        
        padding_after_scaling = (1-im_scale) * width
        x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
        y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
     
        x_scale = np.random.uniform(1, max_xscale)

        y_add = np.random.uniform(-max_yadd, max_yadd)
     
        return _transform_vector(width, 
                                        x_shift=x_delta,
                                        y_shift=y_delta,
                                        im_scale=im_scale, 
                                        x_scale=x_scale,
                                        y_add=y_add)
 
    random_xform_vector = tf.py_func(_random_transformation, [], tf.float32)
    random_xform_vector.set_shape([8])

    output = tf.contrib.image.transform(image_in, random_xform_vector , "BILINEAR")
  
    return output

num_samples = 4 # samples number needed to be increased when GPU is available
average_loss = 0

for i in range(num_samples):
	affine_img = test_random_transform(image_in=image, min_scale=0.5, max_scale=1.0, max_xscale=3.0, max_yadd=1.0)
	affine_logits, _ = inception(affine_img, reuse = True)
	average_loss += tf.nn.softmax_cross_entropy_with_logits(logits=affine_logits, labels = labels) / num_samples

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
demo_steps2 = 100 # small scale picture is hard to be trained into an adversarial sample
demo_target2 = 924

sess.run(assign_op, feed_dict={x:img})

for i in range(demo_steps2):
	_, loss_value = sess.run([optim_step, average_loss], feed_dict={learning_rate: demo_lr2, y_hat: demo_target2})

	sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon2})
	if (i+1) % 10 == 0 :
		print(" step %d, loss %g" %(i+1, loss_value))


adv_robust = x_hat.eval()

for i in range(0, 10):
	affine_out = test_random_transform(image_in=image, min_scale=0.5, max_scale=1.0, max_xscale=3.0, max_yadd=1.0)
	affine_ex = affine_out.eval(feed_dict={image: adv_robust})
	classify(affine_ex, correct_class=img_class, target_class=924)
