#!/usr/bin/env python3
""" 0-main """
import tensorflow as tf
from tensorflow.keras.datasets import mnist


generator = __import__('0-generator').generator

input_layer = tf.keras.layers.Input(shape=(784,))
output = generator(input_layer)
output.summary()
