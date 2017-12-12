import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.5
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))


hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hypothesis, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_acc = []
valid_acc = []
test_acc = []
train_cost_list = []

max_epoch = 0
max_test_acc = 0


for epoch in range(training_epochs):
    avg_train_cost = 0
    avg_train_acc = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, a, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)
        train_cost_list.append(c)
        avg_train_cost += c / total_batch
        avg_train_acc +=  a/total_batch

    print('Epoch :', '%04d' % (epoch +1 ))
    train_a = avg_train_acc
    train_acc.append(train_a)

    valid_a = sess.run(accuracy, feed_dict={X:mnist.validation.images, Y:mnist.validation.labels})
    valid_acc.append(valid_a)

    test_a = sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})
    if max_test_acc <  test_a:
        max_test_acc = test_a
        max_epoch = epoch
    test_acc.append(test_a)
    print('#Train accruracy:',train_a, '#Validation accuracy:', valid_a, '#Test accuracy', test_a )

print("MAX epoch:", max_epoch+1, "MAX Accuracy", max_test_acc)


fig = plt.figure()
plt.plot(train_acc, 'k--', label='train')
plt.plot(valid_acc, 'o--', label='valid')
plt.plot(test_acc, 'r--', label='test')
plt.legend()
plt.show()
