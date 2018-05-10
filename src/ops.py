import tensorflow as tf


WEIGHTS_INIT_STDEV = .1

# leakly relu
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

# batch normalization
def batch_norm(x, momentum = 0.95, epsilon = 1e-5, name = "batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                        scale=True, scope=name)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

# convolution layer
def _conv_layer(net, num_filters, filter_size, strides, relu=True, name="conv2d"):
    with tf.variable_scope(name):
        weights_init = _conv_init_vars(net, num_filters, filter_size, name=name)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        net = _instance_norm(net)
        if relu:
            net = tf.nn.relu(net)
        else: # liuas 2018.5.9
            net = lrelu(net)

        return net

# deconvolution layer
def _conv_tranpose_layer(net, num_filters, filter_size, strides, relu=True, name="tansconv"):
    with tf.variable_scope(name):
        weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True, name=name)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]

        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = _instance_norm(net)

        # liuas 2018.5.9
        if relu:
            return tf.nn.relu(net)
        else:
            return lrelu(net)

# residual layer
def _residual_block(net, filter_size=3, name="residual"):
    with tf.variable_scope(name):
        tmp = _conv_layer(net, 128, filter_size, 1, name=name)
        return net + _conv_layer(tmp, 128, filter_size, 1, relu=False, name=name)


def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

# initial weight for current layer
def _conv_init_vars(net, out_channels, filter_size, transpose=False, name="convinit"):

    # get batch_size, image_width, image_height, image_channel from last layer
    with tf.variable_scope(name):
        _, rows, cols, in_channels = [i.value for i in net.get_shape()]

        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]

        weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
        return weights_init