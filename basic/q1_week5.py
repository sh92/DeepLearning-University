import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets("./", one_hot=True)

def train_layer1(learning_rate, epoch_count, batch_count, training_count):
    with tf.name_scope("layer1"):
        X = tf.placeholder(tf.float32, [None, 784], name='X')
        Y = tf.placeholder(tf.float32, [None, 10], name='Y')
        W = tf.Variable(tf.random_normal([784, 10]), name= 'Weight1')
        bias = tf.Variable(tf.random_normal([10]), name = 'Bias1')
        W_histo = tf.summary.histogram("weight1", W)
        bias_histo = tf.summary.histogram("bias1", bias)
        hypo = tf.matmul(X, W) + bias
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypo, labels=Y))
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        cost_scalar = tf.summary.scalar('cost',cost)
        prediction = tf.argmax(hypo, 1)
        correct = tf.equal(prediction, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_scalar = tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter("./layer1", graph=sess.graph)
        for epoch in range(epoch_count):
            #for i in range(training_count):
            #    x, y = MNIST.train.next_batch(batch_count)
            sess.run([train], feed_dict={X: MNIST.train.images, Y: MNIST.train.labels})
            s, a = sess.run([merged, accuracy], feed_dict={X: MNIST.train.images, Y: MNIST.train.labels})
            writer.add_summary(s, global_step=epoch)

train_layer1(0.05, 2000, 10, 10)
