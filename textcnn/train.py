import os
import time
import datetime

import numpy as np
import tensorflow as tf

from textcnn import utils
from textcnn.model import TextCNN

POSITIVE_DATA_FILE = "./data/rt-polaritydata/rt-polarity.pos"
NEGATIVE_DATA_FILE = "./data/rt-polaritydata/rt-polarity.neg"

DEV_SAMPLE_PERCENT = 0.1
NUM_CHECKPOINTS = 5

EMBEDDING_SIZE = 128
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 128
L2_REG_LAMBDA = 0.0
DROPOUT = 0.5

BATCH_SIZE = 64
NUM_EPOCHS = 200
EVALUATE_EVERY = 100
CHECKPOINT_EVERY = 100


def load_data():
    # Load data
    x_text, y = utils.load_data_and_labels(POSITIVE_DATA_FILE, NEGATIVE_DATA_FILE)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(DEV_SAMPLE_PERCENT * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    return x_train, y_train, x_dev, y_dev, vocab_processor


def train(x_train, y_train, x_dev, y_dev, vocab_processor):

    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: DROPOUT
        }
        _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

    def dev_step(x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: 1.0
        }
        step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            cnn = TextCNN(sequence_length=x_train.shape[1],
                          num_classes=y_train.shape[1],
                          vocab_size=len(vocab_processor.vocabulary_),
                          embedding_size=EMBEDDING_SIZE,
                          filter_sizes=FILTER_SIZES,
                          num_filters=NUM_FILTERS,
                          l2_reg_lambda=L2_REG_LAMBDA)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            train_op = optimizer.minimize(cnn.loss, global_step=global_step)

            # Output directory for models
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            batches = utils.batch_iter(list(zip(x_train, y_train)), BATCH_SIZE, NUM_EPOCHS)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % EVALUATE_EVERY == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")
                if current_step % CHECKPOINT_EVERY == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    x_train, y_train, x_dev, y_dev, vocab_processor = load_data()
    train(x_train, y_train, x_dev, y_dev, vocab_processor)
