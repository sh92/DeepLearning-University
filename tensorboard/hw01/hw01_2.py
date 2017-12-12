import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
training_epochs = 5
batch_size = 100
l1_class = 50
l2_class = 20
nb_class = 10

max_test_acc = 0
batch_size = 100

train_acc = []
test_acc = []
train_cost_list = []

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('Layer1'):
   W1 = tf.Variable(tf.random_normal([784,l1_class], name='weight1'))
   b1 = tf.Variable(tf.random_normal([l1_class]), name='bias1')
   layer1=  tf.sigmoid(tf.matmul(X,W1)+b1)

with tf.name_scope('Layer2'):
   W2 = tf.Variable(tf.random_normal([l1_class, l2_class]), name='weight2')
   b2 = tf.Variable(tf.random_normal([l2_class]), name='bias2')
   layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

with tf.name_scope('Layer3'):
   W3 = tf.Variable(tf.random_normal([l2_class, nb_class]), name='weight2')
   b3 = tf.Variable(tf.random_normal([nb_class]), name='bias2')
   hypothesis = tf.matmul(layer2, W3) + b3

   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=hypothesis, labels=Y))
   cost_scalar = tf.summary.scalar('cost',cost)

   optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
   prediction = tf.argmax(hypothesis, 1)

   correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   accuracy_scalar = tf.summary.scalar('accuracy',accuracy)

   summary = tf.summary.merge_all()

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   writer = tf.summary.FileWriter("./board2");
   total_batch = int(mnist.train.num_examples / batch_size)
   for epoch in range(training_epochs):
      avg_train_cost = 0
      avg_train_acc = 0
      for i in range(total_batch):
         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
         feed_dict = {X: batch_xs, Y: batch_ys}
         s, c, a, _ = sess.run([summary, cost, accuracy, optimizer], feed_dict=feed_dict)
         writer.add_summary(s, global_step=epoch)

         train_cost_list.append(c)
         avg_train_cost += c / total_batch
         avg_train_acc +=  a/total_batch

      print('Epoch :', '%04d' % (epoch +1 ))
      train_a = avg_train_acc
      train_acc.append(train_a)
      print('Learning Finished!')

      test_a = sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})
      if max_test_acc <  test_a:
         max_test_acc = test_a
         max_epoch = epoch
      test_acc.append(test_a)
      print('#Train accruracy:',train_a, '#Test accuracy', test_a )
   print("MAX epoch:", max_epoch+1, "MAX Accuracy", max_test_acc)

   writer.add_graph(sess.graph)
