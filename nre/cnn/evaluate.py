import os
import warnings

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics.base import UndefinedMetricWarning

from nre.cnn import data_utils

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("eval_dir", "../data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT", "Path of evaluation data")
tf.flags.DEFINE_string("output_dir", "result/prediction.txt", "Path of prediction for evaluation data")
tf.flags.DEFINE_string("target_dir", "result/answer.txt", "Path of target(answer) file for evaluation data")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1528107650/checkpoints", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS


def eval():
    with tf.device("/cpu:0"):
        x_text, pos1, pos2, y = data_utils.load_data_and_labels(FLAGS.eval_dir)

    # Map data into vocabulary
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    text_vec = np.array(list(text_vocab_processor.transform(x_text)))

    # Map data into position
    position_path = os.path.join(FLAGS.checkpoint_dir, "..", "position_vocab")
    position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)
    pos1_vec = np.array(list(position_vocab_processor.transform(pos1)))
    pos2_vec = np.array(list(position_vocab_processor.transform(pos2)))

    x_eval = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])
    y_eval = np.argmax(y, axis=1)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_pos1 = graph.get_operation_by_name("input_pos1").outputs[0]
            input_pos2 = graph.get_operation_by_name("input_pos2").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_utils.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_eval_batch in batches:
                x_batch = np.array(x_eval_batch).transpose((1, 0, 2))
                batch_predictions = sess.run(predictions, {input_text: x_batch[0],
                                                           input_pos1: x_batch[1],
                                                           input_pos2: x_batch[2],
                                                           is_training: False})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            labelsMapping = {0: "Other",
                             1: "Message-Topic(e1,e2)", 2: "Message-Topic(e2,e1)",
                             3: "Product-Producer(e1,e2)", 4: "Product-Producer(e2,e1)",
                             5: "Instrument-Agency(e1,e2)", 6: "Instrument-Agency(e2,e1)",
                             7: "Entity-Destination(e1,e2)", 8: "Entity-Destination(e2,e1)",
                             9: "Cause-Effect(e1,e2)", 10: "Cause-Effect(e2,e1)",
                             11: "Component-Whole(e1,e2)", 12: "Component-Whole(e2,e1)",
                             13: "Entity-Origin(e1,e2)", 14: "Entity-Origin(e2,e1)",
                             15: "Member-Collection(e1,e2)", 16: "Member-Collection(e2,e1)",
                             17: "Content-Container(e1,e2)", 18: "Content-Container(e2,e1)"}
            output_file = open(FLAGS.output_dir, "w")
            target_file = open(FLAGS.target_dir, "w")
            for i in range(len(all_predictions)):
                output_file.write("{}\t{}\n".format(i, labelsMapping[all_predictions[i]]))
                target_file.write("{}\t{}\n".format(i, labelsMapping[y_eval[i]]))
            output_file.close()
            target_file.close()

            correct_predictions = float(sum(all_predictions == y_eval))
            print("\nTotal number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))
            print("(2*9+1)-Way Macro-Average F1 Score (excluding Other): {:g}".format(
                f1_score(y_eval, all_predictions, labels=np.array(range(1, 19)), average="macro")))


def main():
    eval()


if __name__ == "__main__":
    main()
