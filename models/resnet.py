import tensorflow as tf
import numpy as np
import os

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.dropout_list = []

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, self.num_x])
            X_img = tf.reshape(self.X, [-1,
             self.ixsize,
             self.iysize,
             self.channels])
            self.Y = tf.placeholder(tf.int32, shape=[None, self.num_y])
            self.Y_one_hot = tf.one_hot(self.Y, self.num_classes)
            self.Y_one_hot = tf.reshape(self.Y_one_hot, [-1, self.num_classes])
            conv1 = self.conv_layer(X_img, 3, 64, 'conv1', 7)
            pool0 = self.max_pool(conv1, 'pool0')
            block1 = self.resnet_block(pool0, 64, 64, 'rblock_1_1')
            block2 = self.resnet_block(block1, 64, 64, 'rblock_1_2')
            block3 = self.resnet_block(block2, 64, 64, 'rblock_1_3')
            block4 = self.resnet_block(block3, 64, 64, 'rblock_2_1')
            block5 = self.resnet_block(block4, 64, 64, 'rblock_2_2')
            block6 = self.resnet_block(block5, 64, 64, 'rblock_2_3')
            block7 = self.resnet_block(block6, 64, 64, 'rblock_2_4')
            self.pool3 = self.avg_pool(block7, 'pool7')
            self.fc = self.fc_layer(self.pool3, 409600, 1000, 'fc')
            flat = tf.reshape(self.fc, [100, 10])
            self.logits = tf.layers.dense(inputs=flat, units=self.num_classes)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y_one_hot))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=1e-06).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def resnet_block(self, bottom, bottom_filt_size, top_filt_size, name):
        with tf.variable_scope(name):
            x = bottom
            conv1 = self.conv_layer(bottom, bottom_filt_size, top_filt_size, 'conv' + name)
            if bottom_filt_size == top_filt_size:
                filt, conv_biases = self.get_conv_var(3, top_filt_size, top_filt_size, name)
                tmp_conv = tf.nn.conv2d(conv1, filt, [1,
                 1,
                 1,
                 1], padding='SAME')
                tmp_conv2 = tf.nn.bias_add(tmp_conv, conv_biases)
                conv2 = tmp_conv2 + x
            else:
                filt, conv_biases = self.get_conv_var(3, top_filt_size, top_filt_size, name)
                conv2 = tf.nn.conv2d(conv1, filt, [1,
                 1,
                 1,
                 1], padding='SAME')
            relu = tf.nn.relu(conv2)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name + '_weights')
        initial_value = tf.truncated_normal([out_size], 0.0, 0.001)
        biases = self.get_var(initial_value, name + '_biases')
        return (weights, biases)

    def conv_layer(self, bottom, in_channels, out_channels, name, filter_size = 3):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1,
             1,
             1,
             1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size,
         filter_size,
         in_channels,
         out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name + '_filters')
        initial_value = tf.truncated_normal([out_channels], 0.0, 0.001)
        biases = self.get_var(initial_value, name + '_biases')
        return (filters, biases)

    def get_var(self, initial_value, var_name):
        var = tf.Variable(initial_value, name=var_name)
        return var

    def max_pool(self, bottom, name, kenel_size = 3, stride = 2):
        return tf.nn.max_pool(bottom, ksize=[1,
         kenel_size,
         kenel_size,
         1], strides=[1,
         stride,
         stride,
         1], padding='SAME', name=name)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2,1], strides=[1,2,2,1], padding='SAME', name=name)

    def set_size(self, num_x, num_y, num_classes):
        self.num_x = num_x
        self.num_y = num_y
        self.num_classes = num_classes

    def set_dropout(self, dropout):
        self.dropout = dropout

    def set_model(self, learning_rate = 0.001):
        self.learning_rate = learning_rate

    def set_img(self, ixsize, iysize, channels):
        self.ixsize = ixsize
        self.iysize = iysize
        self.channels = channels

    def predict(self, x_test, training = False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test,
         self.training: training})

    def get_accuracy(self, x_test, y_test, training = False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test,
         self.Y: y_test,
         self.training: training})

    def train(self, x_data, y_data, training = True):
        return self.sess.run([self.cost, self.accuracy, self.optimizer], feed_dict={self.X: x_data,self.Y: y_data,self.training: training})
