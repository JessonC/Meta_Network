from .cnn import cnn_model
from data.shot_memory import short_memory
from net.read_data import data
from .activate import act

data = data()
cnn = cnn_model()
sm = short_memory()
gan = gan()
act = act()

class meta_net(object):
    def __init__(self):
        self.placeholder()

    def placeholder(self):
        self.im = tf.placeholder(tf.float32, shape=[None, cfg.im_size[0], cfg.im_size[1], cfg.im_size[2]])
        self.label= tf.placeholder(tf.int32, shape=[cfg.batch_size, cfg.cls])
        self.is_training = tf.placeholder(tf.blool)

    def gpu(self):
        pass

    def build_network(self):
        # cnn获取图片特征，输出多个卷机层的feature map
        feature_out = cnn(self.im, self.is_training)

        # 激活网络输入one hot label输出激活特征值
        act_feature = act(self.label)

        # shot memory网络输入feature 通过全连接等网络输出权重值
        memory_weight = sm(feature_out, act_feature)

        # GAN输入特征，输出图片
        out = gan(memory_weight)

    def run(self):
        sess = tf.Sess()
