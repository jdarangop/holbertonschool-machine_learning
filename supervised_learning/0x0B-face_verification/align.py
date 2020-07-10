#!/usr/bin/env python3
""" FaceAlign """
import dlib
import numpy as np
import cv2


class FaceAlign(object):
    """ FaceAlign class """

    def __init__(self, shape_predictor_path):
        """ Initializer """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """ detects a face in an image.
            Args:
                image: (numpy.ndarray) ank 3 containing an image
                       from which to detect a face.
            Returns:
                a dlib.rectangle containing the boundary box for
                the face in the image, or None on failure.
        """
        area = -1
        detected_faces = self.detector(image, 1)
        for face in detected_faces:
            top = max(0, face.top())
            bottom = min(face.bottom(), image.shape[0])
            left = max(0, face.left())
            right = min(face.right(), image.shape[1])
            if (bottom - top) * (right - left) > area:
                top_final = top
                bottom_final = bottom
                left_final = left
                right_final = right
                area = (bottom_final - top_final) * (right_final - left_final)

        return dlib.rectangle(left=left_final, top=top_final,
                              right=right_final, bottom=bottom_final)

    def find_landmarks(self, image, detection):
        """ finds facial landmarks.
            Args:
                image: (numpy.ndarray) image from which to
                       find facial landmarks.
                detection: (dlib.rectangle) containing the boundary box
                           of the face in the image.
            Returns:
                (numpy.ndarray) containing the landmark points,
                or None on failure.
        """
        shape = self.shape_predictor(image, detection)
        result = np.zeros((shape.num_parts, 2))
        for i in range(shape.num_parts):
            result[i][0] = shape.part(i).x
            result[i][1] = shape.part(i).y
        return result

    def align(self, image, landmark_indices, anchor_points, size=96):
        """ aligns an image for face verification.
            Args:
                image: (numpy.ndarray) rank 3 containing the image
                       to be aligned.
                landmark_indices: (numpy.ndarray)  containing the indices
                                  of the three landmark points that should
                                  be used for the affine transformation.
                anchor_points: (numpy.ndarray) containing the destination
                               points for the affine transformation, scaled
                               to the range [0, 1].
                size: (int) the desired size of the aligned image.
            Returns:
                (numpy.ndarray) containing the aligned image, or None if
                no face is detected.
        """
        face = self.detect(image)
        landmarks = self.find_landmarks(image, face)
        points = landmarks[landmark_indices].astype('float32')
        output = anchor_points * size
        prev = cv2.getAffineTransform(points, output)
        result = cv2.warpAffine(image, prev, (size, size))
        return result
