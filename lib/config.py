import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

# 数据集
tf.app.flags.DEFINE_string('train_data', 'data/kaggle/train', "训练集")
tf.app.flags.DEFINE_string('test_data', './test', "测试集")

# Hyperparameter
tf.app.flags.DEFINE_float('learning_rate', 0.001,  "学习速率")
tf.app.flags.DEFINE_integer('batch_size', 32,  "batch size")
tf.app.flags.DEFINE_integer('cls', 2,  "2分类")
tf.app.flags.DEFINE_list('im_size', [224, 224, 3],  "图片大小")
tf.app.flags.DEFINE_float('mean', 115.63,  "图片均值")
tf.app.flags.DEFINE_float('std', 59.32,  "图片方差")



# ------------------------------------------------------------------------------
