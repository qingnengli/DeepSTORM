import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import common
import math

# ----------------------------------------------------------------------
def densenet(x, depth, num_output, wd, is_training):
    stages = []
    K = 16
    if depth == 121:
        stages = [6, 12, 24, 16]
    elif depth == 169:
        stages = [6, 12, 32, 32]
    elif depth == 201:
        stages = [6, 12, 48, 32]
    elif depth == 161:
        stages = [6, 12, 36, 24]
        K = 48

    return getModel(x, num_output, K, stages, wd, is_training)


def full_conv(x, K, is_training, wd):
    with tf.variable_scope('conv1x1'):
        current = common.batchNormalization(x, is_training=is_training)
        current = tf.nn.relu(current)
        current = common.spatialConvolution(current, 1, 1, 4 * K, wd=wd)
    with tf.variable_scope('conv3x3'):
        current = common.batchNormalization(current, is_training=is_training)
        current = tf.nn.relu(current)
        current = common.spatialConvolution(current, 3, 1, K, wd=wd)
    return current


def block(x, layers, K, is_training, wd):
    current = x
    for idx in xrange(layers):
        with tf.variable_scope('L' + str(idx)):
            tmp = full_conv(current, K, is_training, wd=wd)
            current = tf.concat((current, tmp), 3)
    return current


def transition(x, K, wd, is_training):
    with tf.variable_scope('conv'):
        current = common.batchNormalization(x, is_training=is_training)
        current = tf.nn.relu(current)
        # shape = current.get_shape().as_list()
        # dim = math.floor(shape[3]*0.5)
        dim = K
        current = common.spatialConvolution(current, 1, 1, dim, wd=wd)
        current = common.avgPool(current, 2, 2)
    return current


def getModel(x, num_output, K, stages, wd, is_training):
    with tf.variable_scope('conv1'):
        x = common.spatialConvolution(x, 7, 2, 2*K, wd= wd)
        # x = common.spatialConvolution(x, 7, 1, 2 * K, wd=wd)
        x = common.batchNormalization(x, is_training=is_training)
        x = tf.nn.relu(x)
        pool1 = x
        x = common.maxPool(x, 3, 2)
        pool2 = x

    with tf.variable_scope('block1'):
        x = block(x, stages[0], K, is_training=is_training, wd=wd)
    with tf.variable_scope('trans1'):
        x = transition(x, K, wd=wd, is_training=is_training)
        pool3 = x
    with tf.variable_scope('block2'):
        x = block(x, stages[1], K, is_training=is_training, wd=wd)
    with tf.variable_scope('trans2'):
        x = transition(x, K, wd=wd, is_training=is_training)
        pool4 = x
    with tf.variable_scope('block3'):
        x = block(x, stages[2], K, is_training=is_training, wd=wd)
    with tf.variable_scope('trans3'):
        x = transition(x, K, wd=wd, is_training=is_training)

    x = tf.nn.dropout(x, 0.6)

    with tf.variable_scope('deconv1'):
        x = common.spatialConvolution(x, 3, 1, K, wd=wd)
        x = tf.concat((tf.nn.relu(tf.layers.conv2d_transpose(x, filters=K, kernel_size=2, strides=2)),pool4), axis=-1)
    with tf.variable_scope('deconv2'):
        x = common.spatialConvolution(x, 3, 1, K, wd=wd)
        x = tf.concat((tf.nn.relu(tf.layers.conv2d_transpose(x, filters=K, kernel_size=2, strides=2)),pool3), axis=-1)
    with tf.variable_scope('deconv3'):
        x = common.spatialConvolution(x, 3, 1, K, wd=wd)
        x = tf.concat((tf.nn.relu(tf.layers.conv2d_transpose(x, filters=K, kernel_size=2, strides=2)),pool2), axis=-1)
    with tf.variable_scope('deconv4'):
        x = common.spatialConvolution(x, 3, 1, K, wd=wd)
        x = tf.concat((tf.nn.relu(tf.layers.conv2d_transpose(x, filters=K, kernel_size=2, strides=2)),pool1), axis=-1)
    with tf.variable_scope('deconv5'):
        x = common.spatialConvolution(x, 3, 1, K, wd=wd)
        x = tf.nn.relu(tf.layers.conv2d_transpose(x, filters=K, kernel_size=2, strides=2))

    x = common.spatialConvolution(x, 1, 1, num_output, wd=wd)
    x = common.batchNormalization(x, is_training=is_training)
    x = tf.nn.sigmoid(x)

    return x