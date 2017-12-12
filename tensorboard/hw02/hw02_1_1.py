import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np


data = np.loadtxt("./image.csv", delimiter=",", dtype = np.float32)
idx = int(len(data)*4/5)

train_data = data[:idx]
test_data = data[idx:]

train_data_x = train_data[:, 0:-1]
train_data_y = train_data[:, [-1]]

test_data_x = test_data[:, 0:-1]
test_data_y = test_data[:, [-1]]

learning_rate = 0.1
training_epochs = 2000
batch_size = 100
l1_class = 10
l2_class = 9
nb_class = 7

max_test_acc = 0
batch_size = 100


X = tf.placeholder(tf.float32, shape=[None, 19])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_class) # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_class])

W = tf.Variable(tf.random_normal([19, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

train_acc = []
test_acc = []
train_cost_list = []


with tf.name_scope('Layer1'):
   W1 = tf.Variable(tf.random_normal([19, l1_class], name='weight1'))
   b1 = tf.Variable(tf.random_normal([l1_class]), name='bias1')
   layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

   W1_hist = tf.summary.histogram('weight1', W1)
   b1_hist = tf.summary.histogram('biases1', b1)
   layer1_hist = tf.summary.histogram('layer1', layer1)

with tf.name_scope('Layer2'):
   W2 = tf.Variable(tf.random_normal([l1_class, l2_class]), name='weight2')
   b2 = tf.Variable(tf.random_normal([l2_class]), name='bias2')
   layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

   W2_hist = tf.summary.histogram('weight2', W2)
   b2_hist = tf.summary.histogram('biases2', b2)
   layer2_hist = tf.summary.histogram('layer2', layer2)


with tf.name_scope('Layer3'):
   W3 = tf.Variable(tf.random_normal([l2_class, nb_class]), name='weight2')
   b3 = tf.Variable(tf.random_normal([nb_class]), name='bias2')
   hypothesis = tf.matmul(layer2, W3) + b3

   W3_hist = tf.summary.histogram('weight3', W3)
   b3_hist = tf.summary.histogram('biases3', b3)
   layer3_hist = tf.summary.histogram('layer3', hypothesis)

   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=hypothesis, labels=Y_one_hot))

   optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
   prediction = tf.argmax(hypothesis, 1)

   correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   cost_scalar = tf.summary.scalar('cost',cost)
   accuracy_scalar = tf.summary.scalar('accuracy',accuracy)

   summary = tf.summary.merge_all()

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   writer = tf.summary.FileWriter("./board2_1_1");
   total_batch = int( len(train_data)/ batch_size)
   for epoch in range(training_epochs):
      avg_train_cost = 0
      avg_train_acc = 0
      for i in range(total_batch):
         feed_dict = {X: train_data_x, Y: train_data_y}
         s, c, a, _ = sess.run([summary, cost, accuracy, optimizer], feed_dict=feed_dict)
         writer.add_summary(s, global_step=epoch)

         train_cost_list.append(c)
         avg_train_cost += c/ total_batch
         avg_train_acc +=  a/total_batch

      print('Epoch :', '%04d' % (epoch +1 ))
      train_a = avg_train_acc
      train_acc.append(train_a)
      #print('Learning Finished!')

      test_a = sess.run(accuracy, feed_dict={X:test_data_x, Y:test_data_y})
      if max_test_acc <  test_a:
         max_test_acc = test_a
         max_epoch = epoch
      test_acc.append(test_a)
      print('#Train accruracy:',train_a, '#Test accuracy', test_a )
   print("MAX epoch:", max_epoch+1, "MAX Accuracy", max_test_acc)

   writer.add_graph(sess.graph)
