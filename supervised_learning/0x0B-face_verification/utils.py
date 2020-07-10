#!/usr/bin/env python3
""" Utils """
import numpy as np
from PIL import Image
import glob
import csv


def load_images(images_path, as_array=True):
    """ loads images from a directory or file.
        Args:
            images_path: (str) the path to a directory
                         from which to load images.
            as_array: (bool) a boolean indicating whether
                      the images should be loaded as
                      one numpy.ndarray.
        Returns:
            images: is either a list/numpy.ndarray of all images.
            filenames: a list of the filenames associated
                       with each image in images.
    """
    folder = glob.glob(images_path + '/*')
    images = []
    names = []
    for i in folder:
        images.append(Image.open(i))
        names.append(i.split('\\')[1])

    if as_array:
        result = []
        for j in range(len(images)):
            result.append(np.asarray(images[j]))
        images = np.array(result)
    return images, names


def load_csv(csv_path, params={}):
    """ loads the contents of a csv file as a list of lists.
        Args:
            csv_path: (str) the path to the csv to load.
            params: (dict) the parameters to load the csv with.
        Returns:
            a list of lists representing the contents found in csv_path.
    """
    with open(csv_path, 'r', encoding='utf-8') as fp:
        content = csv.reader(fp, *params)
        return list(content)


def save_images(path, images, filenames):
    """ saves images to a specific path.
        Args:
            path: (str) the path to the directory in
                  which the images should be saved.
            images: (list/numpy.ndarray) images to save.
            filenames: (list) filenames of the images to save
        Returns:
            True on success and False on failure.
    """
    if images is None or filenames is None:
        return False
    if len(images) != len(filenames):
        return False
    for i in range(len(images)):
        cv2.imwrite(path + "/" + filenames[i], images[i])
    return True
