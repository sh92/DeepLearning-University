import tensorflow as tf
import os

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.dropout_list = []

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
	    # input place holders
	    self.X = tf.placeholder(tf.float32, [None, self.num_x])
	    X_img = tf.reshape(self.X, [-1, self.ixsize, self.iysize, self.channels])
            self.Y = tf.placeholder(tf.int32, shape=[None, self.num_y])
            self.Y_one_hot = tf.one_hot(self.Y, self.num_classes) # one hot
            self.Y_one_hot = tf.reshape(self.Y_one_hot, [-1, self.num_classes])

	    self.conv1_1 = self.conv_layer(X_img, 3, 32, "conv1_1")
	    self.conv1_2 = self.conv_layer(self.conv1_1, 32, 32, "conv1_2")
	    #self.conv1_3 = self.conv_layer(self.conv1_2, 32, 32, "conv1_3")
	    #self.conv1_4 = self.conv_layer(self.conv1_3, 32, 32, "conv1_4")
	    #self.conv1_5 = self.conv_layer(self.conv1_4, 32, 32, "conv1_5")
	    self.pool1 = self.max_pool(self.conv1_2, 'pool1')
	    self.pool1= tf.nn.dropout(self.pool1, self.dropout)

            '''
	    self.conv2_1 = self.conv_layer(self.pool1, 32, 64, "conv2_1")
	    self.conv2_2 = self.conv_layer(self.conv2_1, 64, 64, "conv2_2")
	    self.conv2_3 = self.conv_layer(self.conv2_2, 64, 64, "conv2_3")
	    self.pool2 = self.max_pool(self.conv2_2, 'pool2')
	    self.pool2= tf.nn.dropout(self.pool2, self.dropout)


	    self.conv3_1 = self.conv_layer(self.pool2, 64, 128, "conv3_1")
	    self.conv3_2 = self.conv_layer(self.conv3_1, 128, 128, "conv3_2")
	    #self.conv3_3 = self.conv_layer(self.conv3_2, 128, 128, "conv3_3")
	    #self.conv3_4 = self.conv_layer(self.conv3_3, 128, 128, "conv3_4")
	    self.pool3 = self.max_pool(self.conv3_2, 'pool3')
	    self.pool3= tf.nn.dropout(self.pool3, self.dropout)

	    self.conv4_1 = self.conv_layer(self.pool3, 128, 256, "conv4_1")
	    self.conv4_2 = self.conv_layer(self.conv4_1, 256, 256, "conv4_2")
	    #self.conv4_4 = self.conv_layer(self.conv4_3, 256, 256, "conv4_4")
	    self.pool4 = self.max_pool(self.conv4_2, 'pool4')
	    self.pool4 = tf.nn.dropout(self.pool4, 0.25)

	    self.conv5_1 = self.conv_layer(self.pool4, 256, 512, "conv5_1")
	    self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
	    #self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
	    #self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
	    self.pool5 = self.max_pool(self.conv5_2, 'pool5')
	    self.pool5 = tf.nn.dropout(self.pool5, 0.5)
            '''

            fc1 = tf.contrib.layers.flatten(self.pool1)
            fc1 = tf.layers.dense(fc1, 512)
            fc1 = tf.layers.dropout(fc1, rate=0.5, training=self.training)
            self.logits = tf.layers.dense(fc1, self.num_classes)

            '''
	    self.fc6 = self.fc_layer(self.pool5, 51200, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
            self.relu6 = tf.nn.relu(self.fc6)
            if self.training==True:
                self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

            self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)
            if self.training==True:
                self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

            self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")
            #self.prob = tf.nn.softmax(self.fc8, name="prob")
            '''

            #self.flat2 = tf.reshape(self.fc8, [100,10])
            #self.logits = tf.layers.dense(inputs=self.flat2, units=self.num_classes)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y_one_hot))
        #self.optimizer = tf.train.AdamOptimizer(
        #    learning_rate=self.learning_rate).minimize(self.cost)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5, decay=1e-7).minimize(self.cost)
        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	pass

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name + "_biases")

        return weights, biases

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name + "_biases")

        return filters, biases

    def get_var(self, initial_value, var_name):
        var = tf.Variable(initial_value, name=var_name)
        #var = tf.constant(initial_value, dtype=tf.float32, name=var_name)
        return var

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def set_size(self, num_x, num_y, num_classes):
        self.num_x = num_x 
        self.num_y = num_y
        self.num_classes= num_classes 

    def set_dropout(self, dropout):
        self.dropout = dropout

    def set_model(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def set_img(self, ixsize, iysize, channels):
        self.ixsize = ixsize
        self.iysize = iysize
        self.channels = channels 

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.accuracy, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})
