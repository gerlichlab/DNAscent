import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json, Model
from tensorflow.python.keras.layers import Dense, Dropout, ZeroPadding1D, Layer, LSTM, Bidirectional, Concatenate
from tensorflow.python.keras.layers import Embedding, Flatten, MaxPooling1D,AveragePooling1D, Input
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Activation, LSTM, SeparableConv1D, Add
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional, Reshape, BatchNormalization, Reshape, Lambda
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.python.keras import Input
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.utils import normalize, to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.backend import ctc_batch_cost

import itertools
from itertools import groupby
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
from scipy.stats import halfnorm

tf.keras.backend.set_learning_phase(1)  # set inference phase

#as compared with train13, this disregards insertion events (mostly in order to get 12 features so we can divide by 4

logPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/modelTraining/trainingLog_CTC_9.csv'
trainingReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/modelTraining/trainingReads_CTC_9.txt'
valReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/modelTraining/validationReads_CTC_9.txt'
checkpointPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/modelTraining/checkpoints_CTC_9'
validationSplit = 0.2

f_checkpoint = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/modelTraining/checkpoints41/weights.19-0.39.h5'

maxLen = 200

#static params
truePositive = 0.9
trueNegative = 0.9
falsePositive = 1. - trueNegative
llThreshold = 1.25

#-------------------------------------------------
#
class trainingRead:

	def __init__(self, augmented_sixMers, augmented_eventMean, augmented_eventLength, augmented_modelMeans, augmented_modelStd, logLikelihood, readID, analogueConc):


		self.sixMers = augmented_sixMers
		self.eventMean = augmented_eventMean
		self.eventLength = augmented_eventLength
		self.modelMeans = augmented_modelMeans
		self.modelStd = augmented_modelStd
		self.logLikelihood = logLikelihood
		self.readID = readID
		self.analogueConc = analogueConc

		if not len(self.sixMers) == len(self.eventMean) == len(self.eventLength) == len(self.modelMeans) == len(self.modelStd):
			print(len(self.sixMers), len(self.logLikelihood))
			print("Length Mismatch")
			sys.exit()

#-------------------------------------------------
# from https://keras.io/examples/vision/captcha_ocr/
class CTCLayer(Layer):
	def __init__(self, name=None):
		super().__init__(name=name)
		self.loss_fn = ctc_batch_cost

	def call(self, input_tuple):
		y_true = input_tuple[0]
		y_pred = input_tuple[1]

		print(input_tuple)

		# Compute the training-time loss value and add it
		# to the layer using `self.add_loss()`.
		batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
		input_length = input_tuple[2]
		label_length = input_tuple[3]

		input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
		label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

		loss = self.loss_fn(y_true, y_pred, input_length, label_length)
		self.add_loss(loss)

		# At test time, just return the computed predictions
		return y_pred


#-------------------------------------------------
#
def convolutional_block(X, f, filters, stage, block, s=1):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3, F4, F5, F6 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = SeparableConv1D(filters=F1, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2a')(X)
    X = Activation('tanh')(X)

    # Second component of main path
    X = SeparableConv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2b')(X)
    X = Activation('tanh')(X)

    # Third component of main path
    X = SeparableConv1D(filters=F3, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2c')(X)
    X = Activation('tanh')(X)

    # Fourth component of main path
    X = SeparableConv1D(filters=F4, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2d', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2d')(X)
    X = Activation('tanh')(X)

    # Fifth component of main path
    X = SeparableConv1D(filters=F5, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2e', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2e')(X)
    X = Activation('tanh')(X)

    # Sixth component of main path
    X = SeparableConv1D(filters=F6, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2f', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2f')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv1D(filters=F6, kernel_size=f, strides=1, padding='same', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('tanh')(X)

    return X


#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}


#-------------------------------------------------
#
def trainingReadToTensor(t):

	wholeRead = []
	for i, s in enumerate(t.sixMers[0:maxLen]):

		for j in range(len(t.eventMean[i])):

			#base
			oneHot = [0]*4
			index = baseToInt[s[0]]
			oneHot[index] = 1

			#other features
			oneHot.append(t.eventMean[i][j])
			oneHot.append(t.modelMeans[i])

			wholeRead.append(np.array(oneHot))
		wholeRead.append(np.zeros(6))

	return np.array((wholeRead))


#-------------------------------------------------
#
def trainingReadToLabel(t,whichSet):

	label = []


	if whichSet == 0.:
		for i, s in enumerate(t.modelMeans[0:maxLen]):

			label.append(1)

	elif whichSet == 1: #a BrdU augmented read
		for i, s in enumerate(t.logLikelihood[0:maxLen]):

			if s == '-': #not thymidine
				label.append(1)
			else:
				if s == '-X': #not thymidine
					label.append(1)
				elif s[-1] == 'X': #in a swapped thymidine region
					label.append(1)
				else:
					score = float(s[:-1])
					if score > llThreshold and np.random.uniform() <= truePositive:
						label.append(2)
					else:
						label.append(1)
	elif whichSet == 2: #a EdU augmented read
		for i, s in enumerate(t.logLikelihood[0:maxLen]):

			if s == '-': #not thymidine
				label.append(1)
			else:
				if s == '-X': #not thymidine
					label.append(1)
				elif s[-1] == 'X': #in a swapped thymidine region
					label.append(1)
				else:
					score = float(s[:-1])
					if score > llThreshold and np.random.uniform() <= truePositive:
						label.append(3)
					else:
						label.append(1)
	elif whichSet == 3: #a EdU augmented read
		for i, s in enumerate(t.logLikelihood[0:maxLen]):

			if s == '-': #not thymidine
				label.append(1)
			else:
				if s == '-X': #not thymidine
					label.append(1)
				elif s[-1] == 'X': #in a swapped BrdU region
					score = float(s[:-1])
					if score > llThreshold and np.random.uniform() <= truePositive:
						label.append(2)
					else:
						label.append(1)
				else:
					score = float(s[:-1])
					if score > llThreshold and np.random.uniform() <= truePositive:
						label.append(3)
					else: #in an EdU region
						label.append(1)
	return np.array(label)


#-------------------------------------------------
#
class DataGenerator(Sequence):
	#'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1, n_classes=10, shuffle=True):
		#'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		#'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		#'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		inputs,outputs = self.__data_generation(list_IDs_temp)

		return inputs,outputs

	def on_epoch_end(self):
		#'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):

		# Generate data
		X = []
		y = []
		input_lengths = []
		label_lengths = []

		for i, ID in enumerate(list_IDs_temp):

			whichSet = -1
			if 'Thym_trainingData' in ID:
				whichSet = 0
			elif 'BrdU_augmentedTrainingData' in ID:
				whichSet = 1
			elif 'EdU_augmentedTrainingData' in ID:
				whichSet = 2
			elif 'EdUinBrdU_trainingData' in ID:
				whichSet = 3
			if whichSet == -1:
				print('setting analogue failed')
				sys.exit()

			#pull data for this 6mer from the appropriate pickled read
			trainingRead = pickle.load(open(ID, "rb"))

			tensor = trainingReadToTensor(trainingRead)
			label = trainingReadToLabel(trainingRead,whichSet)

			X.append(tensor)
			y.append(label)

			input_lengths.append(len(tensor))
			label_lengths.append(len(label))			

		X = pad_sequences(X, dtype='float32', value=0, padding='post')
		X = X.reshape(X.shape[0],X.shape[1],6)
		y = pad_sequences(y, dtype='float32', value=0, padding='post', maxlen=max(label_lengths))
		input_lengths = np.array(input_lengths)
		label_lengths = np.array(label_lengths)
		inputs = {'signal_input': X, 'the_labels': y, 'input_length': input_lengths,'label_length': label_lengths}
		outputs = {'ctc': np.zeros([len(input_lengths)])}  # dummy data for dummy loss function

		return inputs,outputs


#-------------------------------------------------
#
def buildModel():

	input_shape = (None,6)
	classes = 4 #no analogue, BrdU, EdU, plus blank

	# Define the input as a tensor with shape input_shape
	X_input = Input(name='signal_input',shape=input_shape)
	
	# Stage 1
	X = Conv1D(64, 6, strides = 1, padding='same', name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
	X = BatchNormalization(name = 'bn_conv1')(X)
	X = Activation('tanh')(X)

	# Stage 2
	X = convolutional_block(X, f = 6, filters = [64, 64, 64, 64, 64, 64], stage = 2, block='a', s = 1)

	# Stage 3
	X = convolutional_block(X, f=6, filters=[64, 64, 64, 64, 64, 64], stage=3, block='a', s=2)

	# Stage 4
	X = convolutional_block(X, f=15, filters=[128, 128, 128, 128, 128, 128], stage=4, block='a', s=2)

	# Stage 5
	X = convolutional_block(X, f=15, filters=[128, 128, 128, 128, 128, 128], stage=5, block='a', s=2)

	# Stage 6
	X = convolutional_block(X, f=30, filters=[256, 256, 256, 256, 256, 256], stage=6, block='a', s=2)

	X = Conv1D(64, 6, strides = 1, padding='same', name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(name = 'bn_conv2')(X)
	X = Activation('tanh')(X)

	X = Conv1D(64, 6, strides = 1, padding='same', name = 'conv3', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(name = 'bn_conv3')(X)
	X = Activation('tanh')(X)

	X = Conv1D(64, 6, strides = 1, padding='same', name = 'conv4', kernel_initializer = glorot_uniform(seed=0))(X)

	# Output layer
	X = TimeDistributed(Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0)))(X)

	# Add CTC layer for calculating CTC loss at each step
	labels = Input(name="the_labels", shape=(None,), dtype="float32")
	input_length = Input(name="input_length", shape=(None,), dtype="int64")
	label_length = Input(name="label_length", shape=(None,), dtype="int64")
	output = CTCLayer(name="ctc_loss")((labels, X, input_length, label_length))

	# Create model
	model = Model(inputs=[X_input, labels, input_length, label_length], outputs=output, name="BrdU_model")
	op = Adam()
	model.compile(optimizer=op)
	return model


#-------------------------------------------------
#MAIN

#uncomment to train from scratch

filepaths = ['/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdU_trainingData/EdU_augmentedTrainingData_slices_CNN_raw_gap20',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdU_trainingData/EdU_augmentedTrainingData_slices_CNN_raw_gap30',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdU_trainingData/EdU_augmentedTrainingData_slices_CNN_raw_gap40',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/BrdU_trainingData/BrdU_augmentedTrainingData_slices_CNN_raw_gap20',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/BrdU_trainingData/BrdU_augmentedTrainingData_slices_CNN_raw_gap30',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/BrdU_trainingData/BrdU_augmentedTrainingData_slices_CNN_raw_gap40',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdUinBrdU_trainingData_raw/gap20',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdUinBrdU_trainingData_raw/gap30', 
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdUinBrdU_trainingData_raw/gap40',  
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/Thym_trainingData/trainingFiles_raw']

maxReads = [500,
500,
500,
500,
500,
500,
500,
500,
500,
1000]

trainPaths = []
valPaths = []

for index, directory in enumerate(filepaths):

	allPaths = []
	for fcount, fname in enumerate(os.listdir(directory)):
		allPaths.append(directory + '/' + fname)
		if fcount > maxReads[index]:
			break

	random.shuffle(allPaths)
	divideIndex = int(validationSplit*len(allPaths))
	trainPaths += allPaths[divideIndex+1:]
	valPaths += allPaths[0:divideIndex]
	
#record the reads we're using for training and val
f_trainingReads = open(trainingReadLogPath,'w')
for ri in trainPaths:
	f_trainingReads.write(ri+'\n')
f_trainingReads.close()

f_valReads = open(valReadLogPath,'w')
for ri in valPaths:
	f_valReads.write(ri+'\n')
f_valReads.close()

partition = {'training':trainPaths, 'validation':valPaths}

#uncommment to resume from a checkpoint
'''
val_readIDs = []
f_readIDs = open(valReadLogPath,'r')
for line in f_readIDs:
	val_readIDs.append(line.rstrip())
f_readIDs.close()

train_readIDs = []
f_readIDs = open(trainingReadLogPath,'r')
for line in f_readIDs:
	train_readIDs.append(line.rstrip())
f_readIDs.close()

partition = {'training':train_readIDs, 'validation':val_readIDs}
'''
labels = {}

# Parameters
params = {'dim': (None,6),
          'batch_size': 32,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['training'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

#-------------------------------------------------
#CNN architecture

model = buildModel()


print(model.summary())
plot_model(model, to_file='model.png')


#uncomment to load weights from a trainign checkpoint
#model.load_weights(f_checkpoint)

#callbacks
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None)
chk = ModelCheckpoint(checkpointPath + '/weights.{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
csv = CSVLogger(logPath, separator=',', append=False)

#generator fit
model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=100, verbose=1, callbacks=[chk,csv])
print('made it to end')
