import os

import numpy as np
import tensorflow as tf
from scipy.misc import imsave as ims

from vae import input_data
from vae import utils


class LatentAttention:

    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.n_z = 20
        self.batchsize = 100

        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images, [-1, 28, 28, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) +
                                              (1-self.images) * tf.log(1e-8 + 1 - generated_flat), 1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) -
                                               tf.log(tf.square(z_stddev)) - 1, 1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = tf.nn.leaky_relu(tf.layers.conv2d(input_images, 16, (5, 5), strides=(2, 2), padding="same", name="d_h1")) # 28x28x1 -> 14x14x16
            h2 = tf.nn.leaky_relu(tf.layers.conv2d(h1, 32, (5, 5), strides=(2, 2), padding="same", name="d_h2")) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2, [self.batchsize, 7*7*32])

            w_mean = tf.layers.dense(h2_flat, self.n_z, name="w_mean")
            w_stddev = tf.layers.dense(h2_flat, self.n_z, name="w_stddev")
        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = tf.layers.dense(z, 7*7*32, name="z_matrix")
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            h1 = tf.nn.relu(tf.layers.conv2d_transpose(z_matrix, 16, (5, 5), strides=(2, 2), padding="same", name="g_h1"))
            h2 = tf.layers.conv2d_transpose(h1, 1, (5, 5), strides=(2, 2), padding="same", name="g_h2")
            h2 = tf.nn.sigmoid(h2)
        return h2

    def train(self):
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        reshaped_vis = visualization.reshape(self.batchsize, 28, 28)
        ims("results/base.jpg", utils.merge(reshaped_vis[:64], [8, 8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss),
                                                     feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        saver.save(sess, os.getcwd()+"/training/train", global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize, 28, 28)
                        ims("results/"+str(epoch)+".jpg", utils.merge(generated_test[:64], [8, 8]))


if __name__ == "__main__":
    model = LatentAttention()
    model.train()
