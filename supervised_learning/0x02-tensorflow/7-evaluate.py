#!/usr/bin/env python3
""" Evaluate """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ evaluates the output of a neural network.
        X: (np.ndarray) containing the input data to evaluate.
        Y: (np.ndarray) containing the one-hot labels for X.
        save_path: the location to load the model from
        Returns: the networks prediction, accuracy, and loss.
    """
    saver = tf.train.import_meta_graph("{}.meta".format(save_path))
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        accu = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss_mod = sess.run(loss, feed_dict={x: X, y: Y})
        return pred, accu, loss_mod
