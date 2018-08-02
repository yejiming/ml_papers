import math
import timeit

import numpy as np
import tensorflow as tf

from nre.transe.misc_utils import Progbar
from nre.transe.data_utils import KnowledgeGraph


class TransE:

    def __init__(self, kg: KnowledgeGraph, embedding_dim, margin_value,
                 score_func, batch_size, learning_rate):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # ops for training
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")

        # ops for evaluation
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None

        # embeddings
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope("embedding"):
            self.entity_embedding = tf.get_variable(name="entity", shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.relation_embedding = tf.get_variable(name="relation", shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope("normalization"):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope("training"):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_eval_graph(self):
        with tf.name_scope("evaluation"):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope("lookup"):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope("link"):
            distance_pos = head_pos + relation_pos - tail_pos
            distance_neg = head_neg + relation_neg - tail_neg
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope("loss"):
            if self.score_func == "L1":  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name="max_margin_loss")
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope("lookup"):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope("link"):
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope("rank"):
            if self.score_func == "L1":  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session):
        nbatches = (self.kg.n_training_triple + self.batch_size - 1) // self.batch_size
        prog = Progbar(target=nbatches)
        for i, raw_batch in enumerate(self.kg.next_train_batch(self.batch_size)):
            batch_pos, batch_neg = self.kg.generate_training_batch(raw_batch)
            batch_loss, _ = session.run(fetches=[self.loss, self.train_op],
                                        feed_dict={self.triple_pos: batch_pos, self.triple_neg: batch_neg,
                                                   self.margin: [self.margin_value] * len(batch_pos)})
            prog.update(i + 1, [("train loss", batch_loss)], [], [])

    def launch_validation(self, session):
        print("-----Start evaluation-----")
        start = timeit.default_timer()
        prog = Progbar(target=self.kg.n_validation_triple)

        eval_results = []
        for i, eval_triple in enumerate(self.kg.validation_triples):
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_results.append((eval_triple, idx_head_prediction, idx_tail_prediction))
            prog.update(i + 1, [], [], [])

        # Raw
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0

        # Filter
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0

        for eval_result in eval_results:
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = self.calculate_rank(eval_result)
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1

        print("-----Raw-----")
        head_meanrank_raw /= len(eval_results)
        head_hits10_raw /= len(eval_results)
        tail_meanrank_raw /= len(eval_results)
        tail_hits10_raw /= len(eval_results)
        print("-----Head prediction-----")
        print("MeanRank: {:.3f}, Hits@10: {:.3f}".format(head_meanrank_raw, head_hits10_raw))
        print("-----Tail prediction-----")
        print("MeanRank: {:.3f}, Hits@10: {:.3f}".format(tail_meanrank_raw, tail_hits10_raw))
        print("------Average------")
        print("MeanRank: {:.3f}, Hits@10: {:.3f}".format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print("-----Filter-----")
        head_meanrank_filter /= len(eval_results)
        head_hits10_filter /= len(eval_results)
        tail_meanrank_filter /= len(eval_results)
        tail_hits10_filter /= len(eval_results)
        print("-----Head prediction-----")
        print("MeanRank: {:.3f}, Hits@10: {:.3f}".format(head_meanrank_filter, head_hits10_filter))
        print("-----Tail prediction-----")
        print("MeanRank: {:.3f}, Hits@10: {:.3f}".format(tail_meanrank_filter, tail_hits10_filter))
        print("-----Average-----")
        print("MeanRank: {:.3f}, Hits@10: {:.3f}".format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print("cost time: {:.3f}s".format(timeit.default_timer() - start))
        print("-----Finish evaluation-----")

    def calculate_rank(self, eval_result):
        eval_triple, idx_head_prediction, idx_tail_prediction = eval_result
        head, tail, relation = eval_triple
        head_rank_raw = 0
        tail_rank_raw = 0
        head_rank_filter = 0
        tail_rank_filter = 0
        for candidate in idx_head_prediction[::-1]:
            if candidate == head:
                break
            else:
                head_rank_raw += 1
                if (candidate, tail, relation) in self.kg.golden_triple_pool:
                    continue
                else:
                    head_rank_filter += 1
        for candidate in idx_tail_prediction[::-1]:
            if candidate == tail:
                break
            else:
                tail_rank_raw += 1
                if (head, candidate, relation) in self.kg.golden_triple_pool:
                    continue
                else:
                    tail_rank_filter += 1
        return head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter

    def check_norm(self, session):
        print("-----Check norm-----")
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print("entity norm: {} relation norm: {}".format(entity_norm, relation_norm))
