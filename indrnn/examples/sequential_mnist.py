import os
import math
import gzip
import struct
from datetime import datetime

import numpy as np
import tensorflow as tf

from indrnn.ind_rnn_cell import IndRNNCell

TIME_STEPS = 784
NUM_UNITS = 32
NUM_LAYERS = 3
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
NUM_CLASSES = 10

CLIP_GRADIENTS = True
LAST_LAYER_LOWER_BOUND = pow(0.5, 1 / TIME_STEPS)

EPOCHS = 100
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 2000
LEARNING_RATE_INIT = 0.001
LEARNING_RATE_DECAY_STEPS = 600000

OUT_DIR = "out/%s/" % datetime.now().date()
SAVE_PATH = OUT_DIR + "model.ckpt"


def _read(image, label):
    minist_dir = "data/"
    with gzip.open(minist_dir+label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(minist_dir+image, "rb") as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image, label


def get_data():
    train_img, train_label = _read("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    test_img, test_label = _read("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    return train_img, train_label, test_img, test_label


def build(inputs, labels, is_training):
    input_init = tf.random_uniform_initializer(-0.001, 0.001)

    network = inputs

    for layer in range(1, NUM_LAYERS + 1):
        recurrent_init_lower = 0 if layer < NUM_LAYERS else LAST_LAYER_LOWER_BOUND
        recurrent_init = tf.random_uniform_initializer(recurrent_init_lower, RECURRENT_MAX)

        cell = IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX,
                          input_kernel_initializer=input_init,
                          recurrent_kernel_initializer=recurrent_init)
        network, _ = tf.nn.dynamic_rnn(cell, network, dtype=tf.float32, scope="rnn%d" % layer)
        network = tf.layers.batch_normalization(network, training=is_training, momentum=0)

    network = network[:, -1, :]

    logits = tf.layers.dense(network, NUM_CLASSES)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
                                               LEARNING_RATE_DECAY_STEPS, 0.1,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    if CLIP_GRADIENTS:
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimize = optimizer.apply_gradients(zip(gradients, variables))
    else:
        optimize = optimizer.minimize(loss, global_step=global_step)

    correct_pred = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return loss, accuracy, optimize


def run_model():
    inputs = tf.placeholder(tf.float32, [None, TIME_STEPS, 1])
    labels = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool, [])

    loss, accuracy, optimize = build(inputs, labels, is_training)

    X_train, y_train, X_test, y_test = get_data()
    X_train = np.reshape(X_train, [-1, TIME_STEPS, 1])
    X_test = np.reshape(X_test, [-1, TIME_STEPS, 1])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(1, EPOCHS + 1):
            print("======================Epoch {}======================".format(epoch))
            indices = np.arange(y_train.size)
            np.random.shuffle(indices)

            train_losses = []
            train_accuracies = []
            for i in range(int(math.ceil(X_train.shape[0] / BATCH_SIZE_TRAIN))):
                start_idx = (i * BATCH_SIZE_TRAIN) % X_train.shape[0]
                idx = indices[start_idx:start_idx + BATCH_SIZE_TRAIN]

                feed_dict = {inputs: X_train[idx, :], labels: y_train[idx], is_training: True}

                train_loss, train_acc, _ = sess.run([loss, accuracy, optimize], feed_dict=feed_dict)
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)

                if i % 100 == 0:
                    print("Iter {} Loss {} Acc {}".format(i + 1, train_loss, train_acc))

            print("{} Epoch {} Loss {} Acc {}".format(datetime.now(), epoch + 1, np.mean(train_losses),
                                                      np.mean(train_accuracies)))

            if epoch % 10 == 0:
                if not os.path.exists(OUT_DIR):
                    os.makedirs(OUT_DIR)
                save_path = saver.save(sess, SAVE_PATH)
                print("Model saved in path: %s" % save_path)

                feed_dict = {inputs: X_test, labels: y_test, is_training: False}
                test_loss, test_acc, _ = sess.run([loss, accuracy, optimize], feed_dict=feed_dict)
                print("{} Epoch {} valid_loss {} valid_acc {}".format(datetime.utcnow(), epoch + 1, test_loss, test_acc))


if __name__ == "__main__":
    run_model()
