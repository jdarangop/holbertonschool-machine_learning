#!/usr/bin/env python3
""" Yolo """
import tensorflow.keras as K


class Yolo(object):
    """ Yolo class """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Initializer """
        model = K.models.load_model(model_path)
        self.model = model
        with open(classes_path, 'r') as fp:
            classes = [i.strip() for i in fp.readlines()]
            self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
