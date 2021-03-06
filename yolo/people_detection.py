from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from yolo.util import *
from yolo.darknet import Darknet
from yolo.preprocess import inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl


class PeopleDetection():

    def __init__(self):
        super().__init__()
        self.cfgfile = "yolo/cfg/yolov3.cfg"
        self.weightsfile = "yolo/yolov3.weights"
        self.num_classes = 80
        self.confidence = 0.25
        self.nms_thesh = 0.4
        self.reso = 160

        self.start = 0
        self.CUDA = torch.cuda.is_available()

        self.num_classes = 80
        self.bbox_attrs = 5 + self.num_classes

        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightsfile)

        self.model.net_info["height"] = self.reso
        self.inp_dim = int(self.model.net_info["height"])

        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        if self.CUDA:
            self.model.cuda()

        self.model.eval()

        self.classes = load_classes('yolo/data/coco.names')
        self.colors = pkl.load(open("yolo/pallete", "rb"))

    def prep_image(self, img, inp_dim):
        """
        Prepare image for inputting to the neural network. 

        Returns a Variable 
        """

        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = cv2.resize(orig_im, (inp_dim, inp_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim

    def write(self, x, img, only_person=False, paintings_bounding_boxes=None):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        count_persons = 0
        if (not only_person) or (only_person and self.classes[cls] == 'person'):
            label = "{0}".format(self.classes[cls])
            color = random.choice(self.colors)

            # if person is inside a painting
            if paintings_bounding_boxes is not None:
                # (x_min, x_max, y_min, y_max)
                for pbb in paintings_bounding_boxes:
                    if c1[0] > pbb[0] and c1[1] < pbb[1] and c2[0] > pbb[2] and c2[1] < pbb[3]:
                        return 0

            if c1==(0,0) and c2==(0,0):
                return 0

            cv2.rectangle(img, c1, c2, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
            if only_person and self.classes[cls] == 'person':
                count_persons += 1
        return count_persons

    def run(self, frame, paintings_bounding_boxes=None):
        img, orig_im, dim = self.prep_image(frame, self.inp_dim)

        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = self.model(Variable(img), self.CUDA)
        output = write_results(output, self.confidence, self.num_classes,
                               nms=True, nms_conf=self.nms_thesh)

        if type(output) == int:
            return orig_im, False, 0

        output[:, 1:5] = torch.clamp(
            output[:, 1:5], 0.0, float(self.inp_dim))/self.inp_dim

        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        persons_list = list(map(lambda x: self.write(
            x, orig_im, only_person=True, paintings_bounding_boxes=paintings_bounding_boxes), output))

        persons = np.array(persons_list).sum()

        return orig_im, True, persons
