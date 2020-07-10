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

    def process_outputs(self, outputs, image_size):
        """ Process the outputs """
        boxes = [np.zeros(i[:, :, :, :4].shape) for i in outputs]
        box_confidence = []
        box_class_probs = []
        image_height, image_width = image_size
        for i in range(len(outputs)):
            grid_height, grid_width, anchor_boxes, _ = outputs[i].shape
            t_x = outputs[i][:, :, :, 0]
            t_y = outputs[i][:, :, :, 1]
            t_w = outputs[i][:, :, :, 2]
            t_h = outputs[i][:, :, :, 3]

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            cx = np.array([np.arange(grid_width) for i in range(grid_height)])
            cx = cx.reshape(grid_width, grid_width, 1)
            cy = np.array([np.arange(grid_width) for i in range(grid_height)])
            cy = cy.reshape(grid_height,
                            grid_height).T.reshape(grid_height, grid_height, 1)

            bx = ((1 / (1 + np.exp(-t_x))) + cx) / grid_width
            by = ((1 / (1 + np.exp(-t_y))) + cy) / grid_height

            bw = p_w * np.exp(t_w)
            bh = p_h * np.exp(t_h)

            bw /= self.model.input.shape[1].value
            bh /= self.model.input.shape[2].value

            boxes[i][:, :, :, 0] = (bx - (bw / 2)) * image_width
            boxes[i][:, :, :, 1] = (by - (bh / 2)) * image_height
            boxes[i][:, :, :, 2] = (bx + (bw / 2)) * image_width
            boxes[i][:, :, :, 3] = (by + (bh / 2)) * image_height

            box_conf = (1 / (1 + np.exp(-outputs[i][:, :, :, 4:5])))
            box_conf.reshape(grid_height, grid_width, anchor_boxes, 1)
            box_confidence.append(box_conf)

            box_class = (1 / (1 + np.exp(-outputs[i][:, :, :, 5:])))
            box_class_probs.append(box_class)

        return boxes, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ Filter for the boxes.
            Args:
                boxes: (numpy.ndarray) containing the processed boundary
                       boxes for each output, respectively.
                box_confidences: (numpy.ndarray) containing the processed
                                 box confidences for each output.
                box_class_probs: (numpy.ndarray) containing the processed
                                 box class probabilities for each output.
            Returns:
                tuple of (filtered_boxes, box_classes, box_scores).
        """
        pass
