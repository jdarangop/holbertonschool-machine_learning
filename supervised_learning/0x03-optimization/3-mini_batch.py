#!/usr/bin/env python3
""" Train """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ builds, trains, and saves a neural network classifier.
        X_train: (np.ndarray) containing the training input data.
        Y_train: (np.ndarray) containing the training labels
        X_valid: (np.ndarray) containing the validation input data.
        Y_valid: (np.ndarray) containing the validation labels.
        batch_sizes: (int) the number of data points in a batch.
        epoch: (int) the number of times the training should pass
               through the whole dataset.
        load_path: (str) the path from which to load the model.
        save_path: (str) path to save the model.
        Returns: the path where the model was saved.
    """
    m = X_train.shape[0]
    loader = tf.train.import_meta_graph("{}.meta".format(load_path))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        loader.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        for i in range(epochs + 1):
            accu_train = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            loss_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accu_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            loss_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(loss_train))
            print("\tTraining Accuracy: {}".format(accu_train))
            print("\tValidation Cost: {}".format(loss_valid))
            print("\tValidation Accuracy: {}".format(accu_valid))
            if i < epochs:
                X_shu, Y_shu = shuffle_data(X_train, Y_train)
                counter = 0
                j = 0
                z = batch_size
                while (z <= m):
                    sess.run(train_op, feed_dict={x: X_shu[j: z],
                                                  y: Y_shu[j: z]})
                    if (counter + 1) % 100 == 0 and counter != 0:
                        step_accu = sess.run(accuracy,
                                             feed_dict={x: X_shu[j: z],
                                                        y: Y_shu[j: z]})
                        step_cost = sess.run(loss, feed_dict={x: X_shu[j: z],
                                                              y: Y_shu[j: z]})
                        print("\tStep {}:".format(counter + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accu))
                    j = z
                    if (z + batch_size <= m):
                        z += batch_size
                    else:
                        z += m % batch_size
                    counter += 1

        return saver.save(sess, save_path)
