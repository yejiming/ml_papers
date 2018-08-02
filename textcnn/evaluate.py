import os

import numpy as np
import tensorflow as tf

from textcnn import utils


POSITIVE_DATA_FILE = "./data/rt-polaritydata/rt-polarity.pos"
NEGATIVE_DATA_FILE = "./data/rt-polaritydata/rt-polarity.neg"
CHECKPOINT_DIR = "./runs/1525420982/checkpoints"

BATCH_SIZE = 64


def evaluate():
    x_raw, y_test = utils.load_data_and_labels(POSITIVE_DATA_FILE, NEGATIVE_DATA_FILE)
    y_test = np.argmax(y_test, axis=1)

    vocab_path = os.path.join(CHECKPOINT_DIR, "..", "vocab")
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = utils.batch_iter(list(x_test), BATCH_SIZE, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))


if __name__ == "__main__":
    evaluate()
