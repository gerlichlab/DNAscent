import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json, Model
from tensorflow.python.keras.layers import Dense, Dropout, ZeroPadding1D
from tensorflow.python.keras.layers import Embedding, Flatten, MaxPooling1D,AveragePooling1D, Input
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Activation, LSTM, SeparableConv1D, Add
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional, Reshape, BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.python.keras import Input
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import normalize, to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import plot_model
import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
from scipy.stats import halfnorm

tf.keras.backend.set_learning_phase(1)  # set inference phase

#was 38

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_DNAscentTrainingData_highIns_noBrdUScaling_wellMixed_230k'
logPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingLog51pt3.csv'
trainingReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingReadsUsed51.txt'
valReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/valReadsUsed51.txt'
checkpointPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/checkpoints51pt3'
validationSplit = 0.2

f_checkpoint = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/checkpoints51pt2/weights.03-0.41.h5'

maxLen = 3000
maxReads = 230000

#static params
truePositive = 0.5
trueNegative = 0.9
falsePositive = 1. - trueNegative
llThreshold = 1.25


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sixMers, read_eventMeans, read_eventStd, read_stutter, read_lengthMeans, read_lengthStd, read_modelMeans, read_modelStd, read_positions, readID, analogueConc, logLikelihood):
		self.sixMers = read_sixMers[0:maxLen]
		self.eventMean = read_eventMeans[0:maxLen]
		self.eventStd = read_eventStd[0:maxLen]
		self.stutter = read_stutter[0:maxLen]
		self.lengthMean = read_lengthMeans[0:maxLen]
		self.lengthStd = read_lengthStd[0:maxLen]
		self.modelMeans = read_modelMeans[0:maxLen]
		self.modelStd = read_modelStd[0:maxLen]
		self.logLikelihood = logLikelihood[0:maxLen]
		self.readID = readID
		self.analogueConc = analogueConc

		gaps = [1]
		for i in range(1, maxLen):
			gaps.append( abs(read_positions[i] - read_positions[i-1]) )
		self.gaps = gaps

		'''
		print(self.sixMers[0:5])
		print(self.eventMean[0:5])
		print(self.eventStd[0:5])
		print(self.stutter[0:5])
		print(self.lengthMean[0:5])
		print(self.lengthStd[0:5])
		print(self.modelMeans[0:5])
		print(self.modelStd[0:5])
		print(self.gaps[0:5])
		print(read_positions[0:5])
		print(logLikelihood[0:5])
		print('-----------------------')
		'''

		if not len(self.sixMers) == len(self.eventMean) == len(self.eventStd) == len(self.stutter) == len(self.lengthMean) == len(self.lengthStd) == len(self.modelMeans) == len(self.modelStd) == len(self.gaps) == len(self.logLikelihood):
			print(len(self.sixMers), len(self.logLikelihood))
			print("Length Mismatch")
			sys.exit()


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
#
def buildModel(input_shape = (64, 64, 3), classes = 6):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Stage 1
    X = Conv1D(64, 4, strides = 1, padding='same', name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(name = 'bn_conv1')(X)
    X = Activation('tanh')(X)
    #X = MaxPooling1D(4, strides=1, padding='same')(X)

    # Stage 2
    X = convolutional_block(X, f = 4, filters = [64, 64, 64, 64, 64, 64], stage = 2, block='a', s = 1)
    #X = identity_block(X, 4, [64, 64, 256], stage=2, block='b')
    #X = identity_block(X, 4, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=4, filters=[64, 64, 64, 64, 64, 64], stage=3, block='a', s=2)
    #X = identity_block(X, 4, [128, 128, 512], stage=3, block='b')
    #X = identity_block(X, 4, [128, 128, 512], stage=3, block='c')
    #X = identity_block(X, 4, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=8, filters=[128, 128, 128, 128, 128, 128], stage=4, block='a', s=2)
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='b')
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='c')
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='d')
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='e')
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=8, filters=[128, 128, 128, 128, 128, 128], stage=5, block='a', s=2)
    #X = identity_block(X, 4, [512, 512, 2048], stage=5, block='b')
    #X = identity_block(X, 4, [512, 512, 2048], stage=5, block='c')

    # Stage 6
    X = convolutional_block(X, f=16, filters=[256, 256, 256, 256, 256, 256], stage=6, block='a', s=2)

    X = Conv1D(64, 4, strides = 1, padding='same', name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv2')(X)
    X = Activation('tanh')(X)

    X = Conv1D(64, 4, strides = 1, padding='same', name = 'conv3', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv3')(X)
    X = Activation('tanh')(X)

    X = Conv1D(64, 4, strides = 1, padding='same', name = 'conv4', kernel_initializer = glorot_uniform(seed=0))(X)

    # Output layer
    #X = Flatten()(X)
    X = TimeDistributed(Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0)))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='BrdUDetect')

    return model



#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}


#-------------------------------------------------
#
def trainingReadToTensor(t):

	oneSet = []
	for i, s in enumerate(t.sixMers):

		#base
		oneHot = [0]*4
		index = baseToInt[s[0]]
		oneHot[index] = 1

		#other features
		oneHot.append(t.eventMean[i])
		oneHot.append(t.eventStd[i])
		oneHot.append(t.stutter[i])
		oneHot.append(t.lengthMean[i])
		oneHot.append(t.lengthStd[i])
		oneHot.append(t.modelMeans[i])
		oneHot.append(t.modelStd[i])
		oneHot.append(t.gaps[i])
		oneSet.append(oneHot)

	return np.array(oneSet)

#-------------------------------------------------
#
def trainingReadToLabel(t):
	label =[]
	for s in t.logLikelihood:
		if s == '-':
			label.append([1., 0.])
		else:
			score = float(s)
			if score > llThreshold:
				l = (truePositive*t.analogueConc)/(truePositive*t.analogueConc + falsePositive*(1-t.analogueConc))
				label.append([1.-l,l])
			else:
				l = ((1-truePositive)*t.analogueConc)/((1-truePositive)*t.analogueConc + (1-falsePositive)*(1-t.analogueConc))
				label.append([1.-l,l])

	return np.array(label)


#-------------------------------------------------         
#
def trainingReadToWeights(t):
	weights =[]

	#weight thymidine positions 3x more than A,C,G positions
	#also underweight positions where the DNAscent HMM aborted making a call
	for s in t.logLikelihood:
		if s in ['-', '-10000.000000', '-20000.000000']:
			weights.append(1.)
		else:
			weights.append(3.)

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
		X, y, w = self.__data_generation(list_IDs_temp)

		return X, y, w

	def on_epoch_end(self):
		#'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		#'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim))
		y = np.empty((self.batch_size, maxLen, 2), dtype=float)
		w = np.empty((self.batch_size, maxLen), dtype=float)
		#print(list_IDs_temp)
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			splitID = ID.split(';')

			#pull data for this 6mer from the appropriate pickled read
			trainingRead = pickle.load(open(folderPath + '/' + ID + '.p', "rb"))
			tensor = trainingReadToTensor(trainingRead)
			
			X[i,] = tensor

			# Store class
			y[i] = trainingReadToLabel(trainingRead)
			w[i] = trainingReadToWeights(trainingRead)
			
		y = y.reshape(y.shape[0],y.shape[1],2)
		w = w.reshape(w.shape[0],w.shape[1])
		return X, y, w


#-------------------------------------------------
#MAIN

#uncomment to train from scratch
'''
readIDs = []
f_readIDs = open('/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_DNAscentTrainingData_highIns_noBrdUScaling_wellMixed_230k.IDs','r')
for line in f_readIDs:
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
'''

#uncommment to resume from a checkpoint

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

labels = {}

# Parameters
params = {'dim': (maxLen,12),
          'batch_size': 32,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['training'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

#-------------------------------------------------
#CNN architecture

model = buildModel((None,12), 2)
op = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy', sample_weight_mode="temporal")
print(model.summary())
plot_model(model, to_file='model.png')


#uncomment to load weights from a trainign checkpoint
model.load_weights(f_checkpoint)

#callbacks
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
chk = ModelCheckpoint(checkpointPath + '/weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
csv = CSVLogger(logPath, separator=',', append=False)

#generator fit
model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=1000, verbose=1, callbacks=[es,chk,csv])

# serialize model to JSON
model_json = model.to_json()
with open(checkpointPath + "/model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(checkpointPath + "/model.h5")
print("Saved model to disk")
