import numpy as np
import tensorflow as tf
from net import cifar10_model
from data import read_cifar10
Batch_size = 4
cifar10_dir = "/home/neo/project/nova_mind/meta/data/cifar_10"
def loss_func(logits,labels):
    labels = tf.cast(labels,tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                           labels=labels,name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(tf.reduce_sum(cross_entropy))
    # tf.add_to_collection("losses",cross_entropy_mean)
    return cross_entropy_mean# tf.add_n(tf.get_collection("losses"),name="total_loss")


class MetaNetFull():
    def __init__(self):
        self.test = 0
    def train_network(self, isTraning, x_set):
        sample_batch = 4
        sample_len = 4
        smaple_batch_size = 1
        is_training = 1

        support_sets_input, support_lbls_input = read_cifar10.distorted_inputs(cifar10_dir, smaple_batch_size)
        support_set_holder = tf.placeholder(tf.float32, shape=[smaple_batch_size, 32, 32, 3])
        support_lbs_holder = tf.placeholder(tf.int32, shape=[smaple_batch_size])

        u_cnn = cifar10_model.cnn_model()

        # for ii in range(0, sample_len):
        fn_out = u_cnn(support_set_holder, isTraning)
        logits = tf.layers.dense(fn_out, 10, activation=tf.nn.relu, name='fn_out')
        loss = loss_func(logits, support_lbs_holder)
        train_grad = tf.train.AdamOptimizer(1e-3).compute_gradients(loss)
        train_step = tf.train.AdamOptimizer(1e-3).apply_gradients(train_grad)


        # grads_convW1 = tf.reduce_mean(train_grad[40][0], 3)
        # grads_convW2 = tf.reduce_mean(train_grad[42][0], 3)
        # grads_convW3 = tf.reduce_mean(train_grad[44][0], 3)
        grads_convW_sec = []
        for layers in range(40,46):
            grads_convW = tf.reshape([train_grad[layers][0]],[1,-1])
            grads_convW_sec.append(grads_convW)
            test = 0

        grads_convW_t = tf.concat(grads_convW_sec,1)
        grads_convW_t = tf.reshape(grads_convW_t,[-1,1])

        # grads_convW_out = tf.layers.dense(grads_convW_t,20,activation=tf.nn.relu, name='gard_out')
        # grads_convW_t_r = tf.reshape(grads_convW_t,[1,-1])

        grads_convW_out = tf.reshape(grads_convW_t,[1,-1,1])

        lstm_in_shape = [110784,sample_len,1]
        lstm_in_holder = tf.placeholder(tf.float32, shape=lstm_in_shape)
        num_units = 20
        cell_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
        initial_state = cell_lstm.zero_state(1,dtype=tf.float32)
        outputs_lstm, states = tf.nn.dynamic_rnn(cell_lstm,lstm_in_holder,dtype=tf.float32)

        meta_out = tf.reshape(tf.layers.dense(outputs_lstm,1,activation=tf.nn.relu, name='meta_out'),[110784,4,1])

        testSet = []
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()
        for ii in range(0, sample_len):

            support_sets, support_lbls = sess.run([support_sets_input, support_lbls_input])
            # train_step_value, train_grad_value, loss_value = sess.run([train_step, train_grad, loss],
            #                                                           feed_dict={support_set_holder: support_sets,
            #                                                                      support_lbs_holder: support_lbls})
            test1 = sess.run(grads_convW_out,feed_dict={support_set_holder: support_sets,
                                                                                 support_lbs_holder: support_lbls})
            testSet.append(test1)

        lstm_in = np.reshape(np.array(testSet, dtype=np.float32),[-1,sample_len,1])
        # outputs_lstm_data,outputs_lstm_state = sess.run([outputs_lstm,states],feed_dict={lstm_in_holder:lstm_in})
        meta_out_data = sess.run([meta_out, states], feed_dict={lstm_in_holder: lstm_in})
        test = 0




    # def meta_lstm_l1(self):
support_sets = tf.placeholder(tf.float32,shape=[Batch_size,32,32,3])
support_lbls = tf.placeholder(tf.int32,shape=[Batch_size])
x_set = 0
isTraning = 1
test_net = MetaNetFull()
test_net.train_network(isTraning, x_set)
