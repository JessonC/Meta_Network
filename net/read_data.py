from lib.config import FLAGS as cfg
import numpy as np
import random
import glob
import cv2
import os 

class data(object):
    def __init__(self):
        self.cat = glob.glob(cfg.train_data + "/cat*")
        self.dog = glob.glob(cfg.train_data + "/dog*")
        self.train_data = glob.glob(cfg.train_data + "/*.jpg")

        self.index = {"cat":0, "dog":0, "train":0}

        # data numpy
        self.batch_data = np.zeros((cfg.batch_size, cfg.im_size[0], cfg.im_size[1], cfg.im_size[2]))
        self.label = np.zeros((cfg.batch_size, cfg.cls))
    
    def __call__(self):
        '''
        默认只有dog这一个类别训练
        '''
        if (self.index["dog"]+1)*cfg.batch_size > len(self.dog):
            random.shuffle(self.dog)
            self.index["dog"] = 0

        data_list = self.dog[self.index["dog"]*cfg.batch_size:(self.index["dog"]+1)*cfg.batch_size]

        for ind, path in enumerate(data_list):
            im = cv2.imread(path)
            im_arg = self.data_argument(im)
            self.batch_data[ind, ...] = im_arg
            self.label[ind,:] = [1, 0]
        return (self.batch_data - cfg.mean)/cfg.std, self.label
     
    
    def data_argument(self, im):
        # 50%可能性翻转  沿y轴水平旋转
        flip_prop = random.choice([0, 1])
        if flip_prop == 0:
            im = cv2.flip(im, 1)
        im_resize = cv2.resize(im, (cfg.im_size[0], cfg.im_size[1]))
        return im_resize







