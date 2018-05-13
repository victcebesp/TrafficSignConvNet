# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 5


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i),
                                                                 num_classes)  # [one_hot(float(i), num_classes)]
        # image = tf.image.resize_image_with_crop_or_pad(image, 80, 80)
        image = tf.reshape(image, [80, 80, 1])
        image = tf.to_float(image) / 256. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3,
                              activation=tf.nn.relu)  # output -> (x,y,z) = (78, 78, 32)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)  # output -> (x,y,z) = (39, 39, 32)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3,
                              activation=tf.nn.relu)  # output -> (x,y,z) = (37, 37, 64)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)  # output -> (x,y,z) = (18, 18, 64)

        # Change from batch_size * 2 -> batch_size * 3
        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 18 * 64]), units=10, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


test_batch, label_batch_test = dataSource(["data/validation/0/*.ppm", "data/validation/1/*.ppm", "data/validation/2/*.ppm"],
                                          batch_size=batch_size)

example_batch_train_predicted = myModel(test_batch, reuse=False)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_test, dtype=tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./model/model.ckpt")

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    print(sess.run(example_batch_train_predicted))
    coord.request_stop()
    coord.join(threads)
