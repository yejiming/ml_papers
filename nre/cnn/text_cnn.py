import tensorflow as tf


class TextCNN:
    
    def __init__(self, sequence_length, num_classes, text_vocab_size, 
                 text_embedding_size, pos_vocab_size, pos_embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name="input_text")
        self.input_pos1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name="input_pos1")
        self.input_pos2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name="input_pos2")
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name="input_y")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Embedding layer
        with tf.device("/cpu:0"), tf.name_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([text_vocab_size, text_embedding_size], -1.0, 1.0), name="W_text")
            self.text_embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)
            self.text_embedded_chars_expanded = tf.expand_dims(self.text_embedded_chars, -1)
        with tf.device("/cpu:0"), tf.name_scope("position-embedding"):
            self.W_position = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embedding_size], -1.0, 1.0), name="W_position")
            self.pos1_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos1)
            self.pos1_embedded_chars_expanded = tf.expand_dims(self.pos1_embedded_chars, -1)
            self.pos2_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos2)
            self.pos2_embedded_chars_expanded = tf.expand_dims(self.pos2_embedded_chars, -1)

        self.embedded_chars_expanded = tf.concat([self.text_embedded_chars_expanded,
                                                  self.pos1_embedded_chars_expanded,
                                                  self.pos2_embedded_chars_expanded], 2)

        embedding_size = text_embedding_size + 2 * pos_embedding_size

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                h = tf.layers.conv2d(self.embedded_chars_expanded, filters=num_filters,
                                     kernel_size=(filter_size, embedding_size),
                                     strides=(1, 1), activation=tf.nn.relu)
                pooled = tf.layers.max_pooling2d(h, pool_size=(sequence_length - filter_size + 1, 1),
                                                 strides=(1, 1), name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.layers.dropout(self.h_pool_flat, 0.5, training=self.is_training)

        # Final scores and predictions
        with tf.name_scope("output"):
            self.logits = tf.layers.dense(inputs=self.h_drop, units=num_classes,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
                                          bias_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda))
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + tf.reduce_sum(reg_losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
