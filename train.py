import tensorflow as tf
import numpy as np
import models.mymodel as model
import batch
import sys
import os

tf.set_random_seed(0)
with tf.device("gpu:0"):

	model_name = "mymodel"

	print('Loading data...')
	_DATA_DIR = "np_data"
	trainX = np.load(_DATA_DIR+'/trainX.npy')
	trainY = np.load(_DATA_DIR+'/trainY.npy')

	print(np.shape(trainX))
	print(np.shape(trainY))
	testX = np.load(_DATA_DIR+'/testX.npy')
	testY = np.load(_DATA_DIR+'/testY.npy')

	print(np.shape(testX))
	print(np.shape(testY))


	train_size = np.shape(trainX)[0]
	test_size = np.shape(testX)[0]

	num_x_features = np.shape(trainX)[1]
	num_y_features = np.shape(trainY)[1]

	learning_rate = 1e-4
	training_epochs = 200
	batch_size = 100
	num_classes = 10
	total_batch = (int)(train_size/batch_size)

	keep_prob = 0.25

	sess = tf.Session()
	m1 = model.Model(sess, "m1")
	m1.set_model(learning_rate)
	m1.set_dropout(keep_prob)
	m1.set_size(num_x_features, num_y_features, num_classes)
	m1.set_img(32,32,3)
	m1._build_net()
	bat = batch.Batch(trainX, trainY)

	sess.run(tf.global_variables_initializer())
	avg_cost = 0
	sum_cost = 0
	avg_accuracy = 0
	sum_accuracy = 0

	print('Learning Started!')

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
   for epoch in range(training_epochs):
       sum_accuracy = 0
       sum_cost = 0
       for step in range(total_batch):
           batch_xs, batch_ys = bat.next_batch(batch_size)
           c,a, _ = m1.train(batch_xs, batch_ys, True)
           sum_cost += c 
           sum_accuracy += a
           #if step % 10 == 0:
           os.system('cls' if os.name == 'nt' else 'clear')
           print('Epoch:', '%04d' % (epoch + 1),'step:','%d' %step,'cost =', '{:.4f}'.format(sum_cost/(step+1)),"accuracy=", '{:.4f}'.format(sum_accuracy/(step+1)))
       avg_accuracy = sum_accuracy/(total_batch)
       avg_cost = sum_cost/(total_batch)
       print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.4f}'.format(avg_cost), '{:.9f}'.format(c),"accuracy=", avg_accuracy)

print('Learning Finished!')
print('Accuracy:', m1.get_accuracy(testX, testY, False))
