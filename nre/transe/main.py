import argparse

import tensorflow as tf

from nre.transe.data_utils import KnowledgeGraph
from nre.transe.model import TransE


def main():
    parser = argparse.ArgumentParser(description="TransE")
    parser.add_argument("--data_dir", type=str, default="../data/WN18/")
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--margin_value", type=float, default=4.0)
    parser.add_argument("--score_func", type=str, default="L1")
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt/")
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--eval_freq", type=int, default=100)
    args = parser.parse_args()

    kg = KnowledgeGraph(data_dir=args.data_dir)
    model = TransE(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                   score_func=args.score_func, batch_size=args.batch_size, learning_rate=args.learning_rate)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(args.max_epoch):
            print("=" * 30 + "[EPOCH {}]".format(epoch) + "=" * 30)
            model.launch_training(session=sess)
            if (epoch + 1) % args.eval_freq == 0:
                model.launch_validation(session=sess)


if __name__ == "__main__":
    main()
