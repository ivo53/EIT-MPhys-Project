# new conv neural net - UNet

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from datetime import datetime
import h5py
import os
from os.path import exists

'''inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
'''
class Dataset(object):
	def __init__(self, filePath, batchSize):
		f = h5py.File(filePath, 'r')
		print(f.get('train_data').shape[0])
		if f.get('train_data').shape[0] == 100040:
			self._num_examples = 100000
		elif f.get('train_data').shape[0] % batchSize == 0:
			self._num_examples = f.get('train_data').shape[0]
		else:
			raise ValueError('Number of examples has to be divisibble by batch size.')
		self._images = f.get('train_data')[0:int(self._num_examples/10)]
		self._true = f.get('truth_data')[0:int(self._num_examples/10)]
		self._images = self._images.reshape(self._images.shape[0],
											self._images.shape[1],
											self._images.shape[2])
		self._true = self._true.reshape(self._true.shape[0],
										self._true.shape[1],
										self._true.shape[2])
		f.close()
		'''#This is somewhat redundant
		images = images.reshape(images.shape[0],
								images.shape[1],images.shape[2])
		true = true.reshape(true.shape[0],
							true.shape[1],true.shape[2])'''
		self._start = 0
		self._end = batchSize
		self._epochs_completed= 0
		self._index_in_epoch = 0
		self._shuff = np.arange(int(self._num_examples/10))
		self._count = 1
		self._next_batch_count = 0

	@property
	def images(self):
		return self._images

	@property
	def true(self):
		return self._true

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	@property
	def shuff(self):
		return self._shuff

	@property
	def end(self):
		return self._end

	@property
	def start(self):
		return self._start

	def next_batch(self, batch_size, fileName):
		"""Return the next `batch_size` examples from this data set."""

		self._start = self._index_in_epoch
		self._index_in_epoch += batch_size

		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# set new epoch indices
			self._start = 0
			self._index_in_epoch = batch_size
			self._count = 0
			# Start next epoch
			assert batch_size <= self._num_examples

		if self._index_in_epoch > self._count * self._num_examples / 10:
			fData = h5py.File(fileName,'r')
			dataTrain = fData.get('train_data')
			dataTrue = fData.get('truth_data')

			self._images = dataTrain[int((self._count) * self._num_examples / 10):int((self._count + 1) * self._num_examples / 10)]
			self._true = dataTrue[int((self._count) * self._num_examples / 10):int((self._count + 1) * self._num_examples / 10)]
			self._count += 1
			fData.close()
			np.random.shuffle(self._shuff)
			self._images = self._images.reshape(self._images.shape[0],
			                    self._images.shape[1],self._images.shape[2])
			self._true = self._true.reshape(self._true.shape[0],
			                    self._true.shape[1],self._true.shape[2])

		shuffle = self._shuff[int(self._start - (self._count - 1) * self._num_examples / 10):int(self._index_in_epoch - (self._count - 1) * self._num_examples / 10)]
		shuffle = np.sort(shuffle)

		images = self._images[list(shuffle)]
		true = self._true[list(shuffle)]

		self._end = self._index_in_epoch
		self._next_batch_count = 1

		return images, true

def train(dataSet, fileTrain, fileTest, filePath, experimentName, bSize=10, useTensorboard=True, trainIter=1000, checkpointInterval=8000):
	model = UNet()
	#loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
	opt = keras.optimizers.Adam(learning_rate=0.00001)

	model.compile(optimizer=opt, loss=loss_fn, metrics=[loss_fn, 'acc'])
	print(model.summary())
	if(useTensorboard):
		datetime_str = ('{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.now()))
		callbacks = [
			#ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True, verbose=0),
			keras.callbacks.TensorBoard(log_dir='./logs/'+datetime_str, histogram_freq=0, write_graph=True, write_images=True),
		]
	else:
		callbacks = None

	for i in range(trainIter):
		batch = dataSet.train.next_batch(bSize, fileTrain)
		batch = np.array(batch, dtype='f4')
		batch = batch.reshape((2, bSize, batch.shape[2], batch.shape[3], 1))
		model.train_on_batch(batch[0], batch[1])

		if i % 50 == 0:
			testBatch = dataSet.test.next_batch(bSize, fileTest)
			testBatch = np.array(testBatch, dtype='f4')
			testBatch = testBatch.reshape((2, bSize, testBatch.shape[2], testBatch.shape[3], 1))

			print("Iteration %d:" % i, model.test_on_batch(testBatch[0], testBatch[1]))

		if i % checkpointInterval == 0:
			'''checkPointName = filePath + experimentName + '_' + str(i) + '.ckpt'
			checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
			checkpoint.save(checkPointName)
			#saved_model_obj = tf.saved_model.save(sess, checkPointName)'''
			if not exists(filePath):
				os.makedirs(filePath)
			checkPointName = filePath + experimentName + '_' + str(i) + '.h5'
			model.save(checkPointName)
			print("Model saved in file: %s" % checkPointName)

	checkPointName = filePath + 'final/' + experimentName + '_final.h5'
	try:
		model.save(checkPointName)
	except:
		newFilePath = filePath + 'final/'
		if not exists(newFilePath):
			os.makedirs(newFilePath)
		model.save(checkPointName)
	#saved_model_obj = tf.saved_model.save(sess, checkPointName)
	print("Model saved in file: %s" % checkPointName)

def test(fileTest, filePath, experimentName, index=None):
	'''opt = keras.optimizers.Adam(learning_rate=0.0001)
	model = UNet()
	checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
	checkpoint.restore(tf.train.latest_checkpoint(filePath))
	model = checkpoint.model
	model.compile(optimizer=opt, loss=loss_fn, metrics=[loss_fn])'''
	model = keras.models.load_model(filePath, custom_objects={'loss_fn': loss_fn})
	print(model.summary())
	if index is None:
		h = h5py.File(fileTest, 'r')
		reconstructions = h['train_data'][()]
		true = h['truth_data'][()]
		h.close()
	else:
		h = h5py.File(fileTest, 'r')
		reconstructions = h['train_data'][index][()]
		#true = h['truth_data'][index][()]
		h.close()
	reconstructions = reconstructions.reshape(-1, 64, 64, 1)
	#true = true.reshape(-1, 64, 64, 1)
	print(reconstructions.shape)
	return model.evaluate(reconstructions, true, verbose=1)

def process_image(inputs, filePath):
	try:
		inputs = inputs.reshape((-1, 64, 64, 1))
	except:
		raise ValueError("inputs must be an array with dims (x, 64, 64)")
	'''opt = keras.optimizers.Adam(learning_rate=0.0001)
	model = UNet()
	model.compile(optimizer=opt, loss=loss_fn, metrics=[loss_fn])
	'''
	model = keras.models.load_model(filePath, custom_objects={'loss_fn': loss_fn})
	print(model.summary())
	'''
	opt = keras.optimizers.Adam(learning_rate=0.0001)
	checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
	checkpoint.restore(tf.train.latest_checkpoint(filePath))
	model = checkpoint.model
	model.compile(optimizer=opt, loss=loss_fn, metrics=[loss_fn])
	'''
	guess = model.predict(inputs, verbose=1)

	del model
	return guess

def get_rec_from_file(fileTest, index=None):
	if index is None:
		h = h5py.File(fileTest, 'r')
		reconstructions = h['train_data'][()]
		#true = h['truth_data'][()]
		h.close()
	else:
		h = h5py.File(fileTest, 'r')
		reconstructions = h['train_data'][index][()]
		#true = h['truth_data'][index][()]
		h.close()
	return reconstructions.reshape((-1, 64, 64, 1))


def loss_fn(x_out, true):
	return tf.nn.l2_loss(x_out-true)

def read_data_sets(FileNameTrain, FileNameTest, batchSize):
	class DataSets(object):
		pass
	data_sets = DataSets()

	TRAIN_SET = FileNameTrain
	TEST_SET  = FileNameTest
	IMAGE_NAME = 'train_data'
	TRUE_NAME  = 'truth_data'

	print('loading data') 

	data_sets.train = Dataset(FileNameTrain, batchSize)
	data_sets.test = Dataset(FileNameTest, batchSize)

	return data_sets


def lower_part(image, filters, kernel=(3, 3), padding="same", strides=1):
	x = keras.layers.Conv2D(filters, kernel, padding=padding, activation="relu", strides=strides)(image)
	x = keras.layers.Conv2D(filters, kernel, padding=padding, activation="relu", strides=strides)(x)
	return x
def max_pool(image, filters, kernel=(3, 3), padding="same", strides=1):
	x = keras.layers.Conv2D(filters, kernel, padding=padding, activation="relu", strides=strides)(image)
	y = keras.layers.Conv2D(filters, kernel, padding=padding, activation="relu", strides=strides)(x)
	x = keras.layers.MaxPool2D((2,2),(2,2), padding="same")(y)
	return x, y
def upsampleConv(image, old_image, filters, kernel=(3, 3), padding="same", strides=(1, 1)):
	#x = keras.layers.UpSampling2D((2, 2))(image)
	x = keras.layers.Conv2DTranspose(filters, kernel, padding=padding, activation="relu", strides=(2, 2))(image)
	x = keras.layers.Concatenate()([old_image, x])
	x = keras.layers.Conv2D(filters, kernel, padding=padding, activation="relu", strides=strides)(x)
	x = keras.layers.Conv2D(filters, kernel, padding=padding, activation="relu", strides=strides)(x)
	return x
def upsample(image, old_image, filters, kernel=(3, 3), padding="same", strides=(1, 1)):
	x = keras.layers.UpSampling2D(size=(2, 2))(image)
	x = keras.layers.Concatenate()([old_image, x])
	x = keras.layers.Conv2D(filters, kernel, padding=padding, activation="relu", strides=strides)(x)
	x = keras.layers.Conv2D(filters, kernel, padding=padding, activation="relu", strides=strides)(x)

def UNet():
	#model architecture
	inputs = keras.Input(shape=(64, 64, 1), name="reconstructions")
	filters = [64, 128, 256, 512, 1024]
	image = keras.layers.Conv2D(64, 5, padding="same", activation="relu")(inputs)
	#down stage
	image, im_1 = max_pool(image, filters[0])
	image, im_2 = max_pool(image, filters[1])
	image, im_3 = max_pool(image, filters[2])
	image, im_4 = max_pool(image, filters[3])
	image = lower_part(image, filters[4])
	#up stage
	image = upsampleConv(image, im_4, filters[3])
	image = upsampleConv(image, im_3, filters[2])
	image = upsampleConv(image, im_2, filters[1])
	image = upsampleConv(image, im_1, filters[0])
	outputs = keras.layers.Conv2D(1, 1, padding="same", activation="relu")(image)
	model = keras.models.Model(inputs, outputs)
	return model

def UnetUpSample():
	#model architecture with upsample instead of conv transpose
	inputs = keras.Input(shape=(64, 64, 1), name="reconstructions")
	filters = [64, 128, 256, 512, 1024]
	image = keras.layers.Conv2D(64, 5, padding="same", activation="relu")(inputs)
	#down stage
	image, im_1 = max_pool(image, filters[0])
	image, im_2 = max_pool(image, filters[1])
	image, im_3 = max_pool(image, filters[2])
	image, im_4 = max_pool(image, filters[3])
	image = lower_part(image, filters[4])
	#up stage
	image = upsample(image, im_4, filters[3])
	image = upsample(image, im_3, filters[2])
	image = upsample(image, im_2, filters[1])
	image = upsample(image, im_1, filters[0])
	outputs = keras.layers.Conv2D(1, 1, padding="same", activation="relu")(image)
	model = keras.models.Model(inputs, outputs)
	return model
	'''
	x = keras.layers.Conv2D(64, 5, padding="same")(x)
	x = keras.layers.Conv2D(64, 5, padding="same")(x)

	x2 = keras.layers.MaxPool2D(padding="same")(x)

	x2 = keras.layers.Conv2D(128, 5, padding="same")(x2)
	x2 = keras.layers.Conv2D(128, 5, padding="same")(x2)

	x3 = keras.layers.MaxPool2D(padding="same")(x2)

	x3 = keras.layers.Conv2D(256, 5, padding="same")(x3)
	x3 = keras.layers.Conv2D(256, 5, padding="same")(x3)

	x4 = keras.layers.MaxPool2D(padding="same")(x3)

	x4 = keras.layers.Conv2D(512, 5, padding="same")(x4)
	x4 = keras.layers.Conv2D(512, 5, padding="same")(x4)

	x5 = keras.layers.MaxPool2D(padding="same")(x4)

	x5 = keras.layers.Conv2D(1024, 5, padding="same")(x5)
	x5 = keras.layers.Conv2D(1024, 5, padding="same")(x5)

	x5 = keras.layers.UpSampling2D((2, 2))(x5)
	x5 = keras.layers.Conv2DTranspose(512, 5, padding="same")(x5)

	x4_ = keras.layers.Concatenate()([x4, x5])
	x4_ = keras.layers.Conv2D(512, 5, padding="same")(x4_)
	x4_ = keras.layers.Conv2D(512, 5, padding="same")(x4_)

	x4_ = keras.layers.UpSampling2D((2, 2))(x4_)
	x4_ = keras.layers.Conv2DTranspose(256, 5, padding="same")(x4_)

	x3_ = keras.layers.Concatenate()([x3, x4_])
	x3_ = keras.layers.Conv2D(256, 5, padding="same")(x3_)
	x3_ = keras.layers.Conv2D(256, 5, padding="same")(x3_)

	x3_ = keras.layers.UpSampling2D((2, 2))(x3_)
	x3_ = keras.layers.Conv2DTranspose(256, 5, padding="same")(x3_)

	x2_ = keras.layers.Concatenate()([x2, x3_])
	x2_ = keras.layers.Conv2D(128, 5, padding="same")(x2_)
	x2_ = keras.layers.Conv2D(128, 5, padding="same")(x2_)

	x2_ = keras.layers.UpSampling2D((2, 2))(x2_)
	x2_ = keras.layers.Conv2DTranspose(256, 5, padding="same")(x2_)

	x_ = keras.layers.Concatenate()([x, x2_])
	x_ = keras.layers.Conv2D(64, 5, padding="same")(x_)
	x_ = keras.layers.Conv2D(64, 5, padding="same")(x_)
	x_out = keras.layers.Conv2D(1, 5, padding="same")(x_)
	'''

