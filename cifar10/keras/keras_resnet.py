import numpy as np
import os
import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import numpy as np
import resnet

tf.set_random_seed(0)

with tf.device('/gpu:0'):
	lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
	early_stopper = EarlyStopping(min_delta=0.001, patience=10)
	csv_logger = CSVLogger('resnet18_cifar10.csv')

	#save_dir = 'save_model'

	data_augmentation = True
	tf.set_random_seed(0)
	print('Loading data...')
	_DATA_DIR='np_data'
	trainX = np.load('../'+_DATA_DIR+'/trainX.npy')
	trainY = np.load('../'+_DATA_DIR+'/trainY.npy')

	print(np.shape(trainX))
	print(np.shape(trainY))
	testX = np.load('../'+_DATA_DIR+'/testX.npy')
	testY = np.load('../'+_DATA_DIR+'/testY.npy')

	print(np.shape(testX))
	print(np.shape(testY))

	train_size = np.shape(trainX)[0]
	test_size= np.shape(testX)[0]

	learning_rate = 0.0001
	nb_epoch = 200
	batch_size = 100
	nb_classes = 10
	img_rows, img_cols, img_channels = 32, 32, 3

	data_augmentation = True

	x_train = trainX.astype('float32')
	y_train = trainY

	x_test = testX.astype('float32') 
	y_test = testY


	mean_image = np.mean(x_train, axis=0)
	x_train -= mean_image
	x_test -= mean_image
	x_train /= 128.
	x_test /= 128.

	x_train = x_train.reshape(-1,img_rows, img_cols, img_channels)
	x_test = x_test.reshape(-1, img_rows, img_cols, img_channels)
	y_train = keras.utils.to_categorical(y_train, nb_classes)
	y_test = keras.utils.to_categorical(y_test, nb_classes)


	model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
	model.compile(loss='categorical_crossentropy',
		      optimizer='adam',
		      metrics=['accuracy'])

	if not data_augmentation:
	    print('Not using data augmentation.')
	    model.fit(x_train, y_train,
		      batch_size=batch_size,
		      nb_epoch=nb_epoch,
		      validation_data=(x_test, y_test),
		      shuffle=True,
		      callbacks=[lr_reducer, early_stopper, csv_logger])
	else:
	    print('Using real-time data augmentation.')
	    datagen = ImageDataGenerator(
		featurewise_center=False, 
		samplewise_center=False,
		featurewise_std_normalization=False, 
		samplewise_std_normalization=False, 
		zca_whitening=False,  
		rotation_range=0,  
		width_shift_range=0.1, 
		height_shift_range=0.1,  
		horizontal_flip=True, 
		vertical_flip=False) 

	    datagen.fit(x_train)

	    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
				steps_per_epoch=x_train.shape[0] // batch_size,
				validation_data=(x_test, y_test),
				epochs=nb_epoch, verbose=1, max_q_size=100,
				callbacks=[lr_reducer, early_stopper, csv_logger])
	scores = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
