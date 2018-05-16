from PIL import Image
import glob
import numpy as np
import tensorflow as tf


def get_images_paths(path, maxFiles):
    files = []
    for infile in glob.glob(path):
        if len(files) == maxFiles:
            break
        files.append(np.asarray(Image.open(infile)) / 256. + 0.5)
    return files

batch_size = 117

images0, label0 = get_images_paths("data/test/0/*.ppm", batch_size), [0., 0., 1.]
images1, label1 = get_images_paths("data/test/1/*.ppm", batch_size), [0., 1., 0.]
images2, label2 = get_images_paths("data/test/2/*.ppm", batch_size), [1., 0., 0.]

for i in range(0, len(images0)):
    images0[i] = np.reshape(images0[i], [80, 80, 1])
    images1[i] = np.reshape(images1[i], [80, 80, 1])
    images2[i] = np.reshape(images2[i], [80, 80, 1])

num_classes = 3

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

x = tf.placeholder("float", [None, 80, 80, 1])
y_ = tf.placeholder("float", [None, 3])

with tf.variable_scope('ConvNet'):
    o1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3,
                          activation=tf.nn.relu)  # output -> (x,y,z) = (78, 78, 32)
    o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)  # output -> (x,y,z) = (39, 39, 32)
    o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3,
                          activation=tf.nn.relu)  # output -> (x,y,z) = (37, 37, 64)
    o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)  # output -> (x,y,z) = (18, 18, 64)

    # Change from batch_size * 2 -> batch_size * 3
    h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size, 18 * 18 * 64]), units=10, activation=tf.nn.relu)
    y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./model/model.ckpt")
    output0 = sess.run(y, feed_dict={x: images0})
    output1 = sess.run(y, feed_dict={x: images1})
    output2 = sess.run(y, feed_dict={x: images2})

    output0 = [np.argmax(e) for e in output0]
    output1 = [np.argmax(e) for e in output1]
    output2 = [np.argmax(e) for e in output2]

    confusion_matrix0 = sess.run(tf.confusion_matrix(np.zeros(batch_size), np.array(output0), 3))
    confusion_matrix1 = sess.run(tf.confusion_matrix(np.zeros(batch_size) + 1, np.array(output1), 3))
    confusion_matrix2 = sess.run(tf.confusion_matrix(np.zeros(batch_size) + 2, np.array(output2), 3))

    right_answers = np.diagonal(confusion_matrix0).sum() + \
                    np.diagonal(confusion_matrix1).sum() + \
                    np.diagonal(confusion_matrix2).sum()

    print "Accuracy: " + str(right_answers * 100 / (batch_size * 3)) + "%"