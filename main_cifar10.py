import tensorflow as tf
import time
import numpy as np
from data import read_cifar10
from net import cifar10_model

Batch_size = 64
max_steps = 10000
cifar10_dir = "/home/neo/project/nova_mind/meta/data/cifar_10"

def loss_func(logits,labels):
    labels = tf.cast(labels,tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                           labels=labels,name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(tf.reduce_sum(cross_entropy))
    # tf.add_to_collection("losses",cross_entropy_mean)
    return cross_entropy_mean# tf.add_n(tf.get_collection("losses"),name="total_loss")

def main():
    images_train, labels_train = read_cifar10.distorted_inputs(cifar10_dir, Batch_size)
    images_test, labels_test = read_cifar10.inputs(eval_data=True, data_dir=cifar10_dir
                                                    , batch_size=Batch_size)
    is_training = 1
    image_holder = tf.placeholder(tf.float32,shape=[Batch_size,32,32,3])
    label_holder = tf.placeholder(tf.int32,shape=[Batch_size])
    cnn_model = cifar10_model.cnn_model()
    fn_out = cnn_model(image_holder,is_training)
    logits = tf.layers.dense(fn_out,10,activation=tf.nn.relu, name='fn_out')
    loss = loss_func(logits, label_holder)
    #train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    train_grad = tf.train.AdamOptimizer(1e-3).compute_gradients(loss)
    train_step = tf.train.AdamOptimizer(1e-3).apply_gradients(train_grad)

    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        images_batch, labels_batch = sess.run([images_train, labels_train])
        train_step_value, train_grad_value, loss_value = sess.run([train_step, train_grad, loss], feed_dict={image_holder: images_batch,
                                                                label_holder: labels_batch})
        duration = time.time() - start_time
        if step % 1000 == 0:
            # 计算每秒处理多少张图片
            per_images_second = Batch_size / duration
            # 获取时间
            sec_per_batch = float(duration)
            print("step:%d,duration:%.3f,per_images_second:%.2f,loss:%.3f" % (step, duration
                                                                              , per_images_second, loss_value))
            images_batch, labels_batch = sess.run([images_test, labels_test])
            pred = sess.run([top_k_op], feed_dict={image_holder: images_batch, label_holder: labels_batch})
            precision = np.sum(pred) / Batch_size

            print("test accuracy:%.3f" % precision)
    test = 0

main()