import tensorflow as tf


def residual_block(input_img, filter_size, scope_name, model):
    input_depth = int(input_img.get_shape()[3])
    with tf.variable_scope(scope_name):
        conv1 = tf.layers.conv2d(inputs=input_img, filters=filter_size, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 activation=tf.nn.relu, name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, filters=filter_size, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 name='conv2')
        bn_3 = tf.layers.batch_normalization(conv2, name='bn', training=model)
        padding_zeros = tf.pad(input_img, [[0, 0], [0, 0], [0, 0], [int((filter_size - input_depth) / 2),
                                                                    filter_size - input_depth - int(
                                                                   (filter_size - input_depth) / 2)]])
    res_block = padding_zeros + bn_3
    return res_block

"""
ResNet-32
输入图片大小：224*224*3
输出：[batch_size, 512]
"""
class cnn_model(object):
    def __init__(self):
        self.reuse = False

    def __call__(self, features, is_training):
        with tf.variable_scope("cnn") as scope_name:
            if self.reuse:
                scope_name.reuse_variables()
            conv1 = tf.layers.conv2d(inputs=features, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation=tf.nn.relu, name='conv1')
            pool1 = tf.layers.max_pooling2d(conv1, [3, 3], 2)
            # conv2
            bn_1 = tf.layers.batch_normalization(pool1, name='bn1', training=is_training)
            res_block_1 = residual_block(bn_1, 64, "res_block_1", is_training)
            res_block_2 = residual_block(res_block_1, 64, "res_block_2", is_training)
            conv2 = residual_block(res_block_2, 64, "conv2", is_training)
            pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)
            # conv3
            res_block_3 = residual_block(pool2, 128, "res_block_3", is_training)
            res_block_4 = residual_block(res_block_3, 128, "res_block_4", is_training)
            res_block_5 = residual_block(res_block_4, 128, "res_block_5", is_training)
            conv3 = residual_block(res_block_5, 128, "conv3", is_training)
            pool3 = tf.layers.max_pooling2d(conv3, [2, 2], 2)
            # conv4
            res_block_6 = residual_block(pool3, 256, "res_block_6", is_training)
            res_block_7 = residual_block(res_block_6, 256, "res_block_7", is_training)
            res_block_8 = residual_block(res_block_7, 256, "res_block_8", is_training)
            res_block_9 = residual_block(res_block_8, 256, "res_block_9", is_training)
            res_block_10 = residual_block(res_block_9, 256, "res_block_10", is_training)
            conv4 = residual_block(res_block_10, 256, "conv4", is_training)
            pool4 = tf.layers.max_pooling2d(conv4, [2, 2], 2)
            # conv5
            res_block_11 = residual_block(pool4, 512, "res_block_11", is_training)
            res_block_12 = residual_block(res_block_11, 512, "res_block_12", is_training)
            conv5 = residual_block(res_block_12, 512, "conv5", is_training)
            # print("conv5", conv5)
            # bn_in = tf.layers.batch_normalization(conv5, name='bn6', training=C.is_training)  # 这个bn在fc之前是比较重要的
            #
            # fc_shape = int(res_block_8.get_shape()[1] * res_block_8.get_shape()[2] * res_block_8.get_shape()[3])
            # fc_in = tf.reshape(bn_in, [-1, fc_shape])
            # fc = tf.layers.dense(fc_in, FLAGS.feature_size, name='fc')
            fc = tf.layers.average_pooling2d(conv5, pool_size=(6, 6), strides=1)
            fc = tf.squeeze(fc, [1, 2])
            # print(fc)
            fc_bn = tf.layers.batch_normalization(fc,  name='bn_fc1', training=is_training)
            # print(fc_bn)
        self.reuse = True
        return fc_bn   # [batch, 512]
