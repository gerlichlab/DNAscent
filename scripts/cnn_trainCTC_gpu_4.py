import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json, Model
from tensorflow.python.keras.layers import Dense, Dropout, ZeroPadding1D, Layer
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
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
from scipy.stats import halfnorm

tf.keras.backend.set_learning_phase(1)  # set inference phase

#as compared with train13, this disregards insertion events (mostly in order to get 12 features so we can divide by 4

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_CTC'
logPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingLogCTC_4.csv'
trainingReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingReadsUsedCTC_4.txt'
valReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/valReadsUsedCTC_4.txt'
checkpointPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/checkpointsCTC_4'
validationSplit = 0.2

#f_checkpoint = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/checkpoints19/weights.09-0.41.h5'

maxLen = 2001
maxReads = 250#100000 #

#static params
truePositive = 0.5
trueNegative = 0.9
falsePositive = 1. - trueNegative
llThreshold = 1.25

#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sequence, read_raw, read_modelMeans, read_modelStdvs, seqIdx2LL, readID, analogueConc, ll):
		
		self.sequence = read_sequence
		self.raw = read_raw
		self.modelMeans = read_modelMeans
		self.modelStdvs = read_modelStdvs
		self.readID = readID
		self.analogueConc = analogueConc
		self.logLikelihood = seqIdx2LL
		self.labelLength = ll

#-------------------------------------------------
# from https://keras.io/examples/vision/captcha_ocr/
class CTCLayer(Layer):
	def __init__(self, name=None):
		super().__init__(name=name)
		self.loss_fn = ctc_batch_cost

	def call(self, input_tuple):
		y_true = input_tuple[0]
		y_pred = input_tuple[1]
		# Compute the training-time loss value and add it
		# to the layer using `self.add_loss()`.
		batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
		input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
		label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

		input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
		label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

		loss = self.loss_fn(y_true, y_pred, input_length, label_length)
		self.add_loss(loss)

		# At test time, just return the computed predictions
		return y_pred


#-------------------------------------------------
#
def identity_block(X, f, filters, stage, block):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = SeparableConv1D(filters = F1, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('tanh')(X)
    
    # Second component of main path
    X = SeparableConv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)
    X = Activation('tanh')(X)

    # Third component of main path 
    X = SeparableConv1D(filters = F3, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('tanh')(X)
    
    return X


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

	return np.array(t.raw)

#-------------------------------------------------
#
def trainingReadToLabel(t):
	label =[]
	#blank, B, C, G, T, A, N 

	if t.analogueConc != -1: #not a data augmented read
		for i, s in enumerate(t.sequence):
			if s == '-':
				label.append(1)
			elif s == 'C':
				label.append(3)
			elif s == 'G':
				label.append(4)
			elif s == 'T':
				label.append(5)
			elif s == 'A':
				label.append(6)
			elif s == 'N':
				label.append(7)
		return np.array(label)
	else: #data augmented read
		for i, s in enumerate(t.sequence):
			if s == '-':
				label.append(1)
			elif s == 'C':
				label.append(3)
			elif s == 'G':
				label.append(4)
			elif s == 'T':
				if s == '-X':
					label.append(5)
				elif s[-1] == 'X':
					score = float(s[:-1])
					if score > llThreshold and np.random.uniform() <= 0.9:
						label.append(2)
					else:
						label.append(5)
				else:	
					label.append(5)					
			elif s == 'A':
				label.append(6)
			elif s == 'N':
				label.append(7)
		return np.array(label)	

#-------------------------------------------------         
#
def trainingReadToWeights(t):

	weightOnBrdU = 10.

	weights =[]

	#weight augmented data more because we have less of it
	scaling = 1.
	#if t.analogueConc == -1:
	#	scaling = 2.

	#weight thymidine positions 3x more than A,C,G positions
	#also underweight positions where the DNAscent HMM aborted making a call
	if t.analogueConc != -1: #not a data augmented read
		for s in t.logLikelihood:
			if s in ['-', '-10000.000000', '-20000.000000']:
				weights.append(1.*scaling)
			else:
				weights.append(3.*scaling)

	else: #is a data augmented read
		for s in t.logLikelihood:

			if s == '-X':
				weights.append(1.*scaling)
			elif s[-1] == 'X': #in a swapped 80% BrdU region
				score = float(s[:-1])
				if score > llThreshold:
					weights.append(3.*scaling*weightOnBrdU)
				else:
					weights.append(3.*scaling)
			elif s in ['-', '-10000.000000', '-20000.000000']:
				weights.append(1.*scaling)
			else:
				weights.append(3.*scaling)

	return np.array(weights)


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

			#pull data for this 6mer from the appropriate pickled read
			trainingRead = pickle.load(open('/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_CTC/'+ID+'.p', "rb"))

			tensor = trainingReadToTensor(trainingRead)

			X.append(trainingReadToTensor(trainingRead))
			y.append(trainingReadToLabel(trainingRead))

			input_lengths.append(len(tensor))
			label_lengths.append(trainingRead.labelLength)			

		X = pad_sequences(X, dtype='float32', value=0.0)
		X = X.reshape(X.shape[0],X.shape[1],1)
		y = pad_sequences(y, dtype='float32', value=0.0,maxlen=max(label_lengths))
		input_lengths = np.array(input_lengths)
		label_lengths = np.array(label_lengths)
		inputs = {'the_input': X, 'the_labels': y, 'input_length': input_lengths,'label_length': label_lengths}
		outputs = {'ctc': np.zeros([len(input_lengths)])}  # dummy data for dummy loss function

		return inputs,outputs


#-------------------------------------------------
#
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    #y_pred = y_pred[:, 2:, :]
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


#-------------------------------------------------
#
def buildModel(input_shape = (64, 64, 3), classes = 6):

	input_shape = (None,1)
	classes = 9 #6 bases, one blank, one padding

	# Define the input as a tensor with shape input_shape
	X_input = Input(name='the_input',shape=input_shape)

	# Stage 1
	X = Conv1D(256, 4, strides = 1, padding='same', name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
	X = BatchNormalization(name = 'bn_conv1')(X)
	X = Activation('tanh')(X)

	# Stage 2
	X = convolutional_block(X, f = 4, filters = [256, 256, 256, 256, 256, 256], stage = 2, block='a', s = 1)

	# Stage 3
	X = convolutional_block(X, f=4, filters=[256, 256, 256, 256, 256, 256], stage=3, block='a', s=2)

	# Stage 4
	X = convolutional_block(X, f=8, filters=[512, 512, 512, 512, 512, 512], stage=4, block='a', s=2)

	# Stage 5
	X = convolutional_block(X, f=8, filters=[512, 512, 512, 512, 512, 512], stage=5, block='a', s=2)

	# Stage 6
	X = convolutional_block(X, f=16, filters=[512, 512, 512, 512, 512, 512], stage=6, block='a', s=2)

	X = Conv1D(512, 4, strides = 1, padding='same', name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(name = 'bn_conv2')(X)
	X = Activation('tanh')(X)

	X = Conv1D(1024, 4, strides = 1, padding='same', name = 'conv3', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(name = 'bn_conv3')(X)
	X = Activation('tanh')(X)

	X = Conv1D(classes, 4, strides = 1, padding='same', name = 'conv4', kernel_initializer = glorot_uniform(seed=0))(X)

	# Output layer
	X = TimeDistributed(Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0)))(X)

	# Add CTC layer for calculating CTC loss at each step
	labels = Input(name="the_labels", shape=(None,), dtype="float32")
	output = CTCLayer(name="ctc_loss")((labels, X))

	# Create model
	model = Model(inputs=[X_input, labels], outputs=output, name="BrdU_model")
	op = Adam()
	model.compile(optimizer=op)
	return model


#-------------------------------------------------
#MAIN

#uncomment to train from scratch

readIDs = []
f_readIDs = open('/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_CTC.IDs','r')
for i, line in enumerate(f_readIDs):
	if i == 250:
		break
	readIDs.append(line.rstrip())
f_readIDs.close()

random.shuffle(readIDs)

#truncate reads
readIDs = readIDs[0:maxReads]
f_trainingReads = open(trainingReadLogPath,'w')
for ri in readIDs:
	f_trainingReads.write(ri+'\n')
f_trainingReads.close()

divideIndex = int(validationSplit*len(readIDs))

#record the reads we're using for training and val
f_trainingReads = open(trainingReadLogPath,'w')
for ri in readIDs[divideIndex+1:]:
	f_trainingReads.write(ri+'\n')
f_trainingReads.close()

f_valReads = open(valReadLogPath,'w')
for ri in readIDs[0:divideIndex]:
	f_valReads.write(ri+'\n')
f_valReads.close()

partition = {'training':readIDs[divideIndex+1:], 'validation':readIDs[0:divideIndex]}


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
params = {'dim': (None,1),
          'batch_size': 32,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['training'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

#-------------------------------------------------
#CNN architecture

model = buildModel((None,10), 6)


print(model.summary())
plot_model(model, to_file='model.png')


#uncomment to load weights from a trainign checkpoint
#model.load_weights(f_checkpoint)

#callbacks
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None)#, restore_best_weights=True)
chk = ModelCheckpoint(checkpointPath + '/weights.{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
csv = CSVLogger(logPath, separator=',', append=False)

#generator fit
model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=1000, verbose=1, callbacks=[chk,csv])
print('made it to end')
