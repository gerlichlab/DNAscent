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

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_DNAscentTrainingData_highIns_noBrdUScaling_CTC_noGap'
logPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingLog41.csv'
trainingReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingReadsUsed41.txt'
valReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/valReadsUsed41.txt'
checkpointPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/checkpoints41'
validationSplit = 0.2

#f_checkpoint = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/checkpoints19/weights.09-0.41.h5'

maxLen = 3000
maxReads = 100000

#static params
truePositive = 0.5
trueNegative = 0.9
falsePositive = 1. - trueNegative
llThreshold = 1.25

#hack - global variable for batch lengths
lengths_training = []
lengths_labels = []


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sequence, read_eventMeans, read_eventLengths, read_modelMeans, read_modelStdvs, seqIdx2LL, readID, analogueConc):
		
		#reshape log likelihood calls to sequence
		logLikelihood = []
		for i in range(0, len(read_sequence)):
			if i in seqIdx2LL:
				logLikelihood.append(seqIdx2LL[i])
			else:
				logLikelihood.append('-')


		self.sequence = read_sequence
		self.eventMeans = read_eventMeans
		self.eventLengths = read_eventLengths
		self.modelMeans = read_modelMeans
		self.modelStdvs = read_modelStdvs
		self.readID = readID
		self.analogueConc = analogueConc
		self.logLikelihood = logLikelihood

		'''
		print(self.sequence[0:40])
		print(self.eventMeans[0:5])
		print(self.eventLengths[0:5])
		print(self.modelMeans[0:5])
		print(self.modelStdvs[0:5])
		print(self.readID)
		print(self.analogueConc)
		print(self.logLikelihood[0:5])
		print('-----------------------')
		'''


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
    X = convolutional_block(X, f=8, filters=[64, 64, 64, 64, 64, 64], stage=3, block='a', s=2)
    #X = identity_block(X, 4, [128, 128, 512], stage=3, block='b')
    #X = identity_block(X, 4, [128, 128, 512], stage=3, block='c')
    #X = identity_block(X, 4, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=8, filters=[64, 64, 64, 64, 64, 64], stage=4, block='a', s=2)
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='b')
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='c')
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='d')
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='e')
    #X = identity_block(X, 4, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=16, filters=[64, 64, 64, 64, 64, 64], stage=5, block='a', s=2)
    #X = identity_block(X, 4, [512, 512, 2048], stage=5, block='b')
    #X = identity_block(X, 4, [512, 512, 2048], stage=5, block='c')

    # Stage 6
    X = convolutional_block(X, f=16, filters=[64, 64, 64, 64, 64, 64], stage=6, block='a', s=2)

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
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3, 'B':4, '-':5}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C', 4:'B', 5:'-'}


#-------------------------------------------------
#
def trainingReadToTensor(t):

	oneSet = []
	for i, s in enumerate(t.sequence):

		#base
		oneHot = [0]*6
		index = baseToInt[s[0]]
		oneHot[index] = 1

		#other features
		oneHot.append(t.eventMeans[i])
		oneHot.append(t.eventLengths[i])
		oneHot.append(t.modelMeans[i])
		oneHot.append(t.modelStdvs[i])

		oneSet.append(oneHot)

	return np.array(oneSet)

#-------------------------------------------------
#
'''
def trainingReadToLabel(t):
	label =[]
	for i, s in enumerate(t.sequence):

		oneHot = [0]*6

		if s != "T":
			oneHot[baseToInt[s]] = 1
		elif t.logLikelihood[i] == '-':
			oneHot[baseToInt['T']] = 0.5
			oneHot[baseToInt['B']] = 0.5
		else:
			score = float(t.logLikelihood[i])
			if score > llThreshold:
				l = (truePositive*t.analogueConc)/(truePositive*t.analogueConc + falsePositive*(1-t.analogueConc))
				oneHot[baseToInt['T']] = 1. - l
				oneHot[baseToInt['B']] = l
			else:
				l = ((1-truePositive)*t.analogueConc)/((1-truePositive)*t.analogueConc + (1-falsePositive)*(1-t.analogueConc))
				oneHot[baseToInt['T']] = 1. - l
				oneHot[baseToInt['B']] = l

		label.append(np.array(oneHot))

	return np.array(label)
'''

#-------------------------------------------------
#hard labels
def trainingReadToLabel(t):
	label =[]
	for i, s in enumerate(t.sequence):

		if s != "T":
			label.append(baseToInt[s])
		elif t.logLikelihood[i] == '-':
			label.append(baseToInt['T'])
		else:
			score = float(t.logLikelihood[i])
			if score > llThreshold:
				label.append(baseToInt['B'])
			else:
				label.append(baseToInt['T'])


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
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		#'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		#'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		#X = np.empty((self.batch_size, *self.dim))
		#y = np.empty((self.batch_size, maxLen, 2), dtype=float)

		X = []
		y = []
		global lengths_training
		lengths_training = []
		global lengths_labels
		lengths_labels = []

		#print(list_IDs_temp)
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			splitID = ID.split(';')

			#pull data for this 6mer from the appropriate pickled read
			trainingRead = pickle.load(open(folderPath + '/' + ID + '.p', "rb"))

			tensor_training = trainingReadToTensor(trainingRead)
			X.append(tensor_training)
			lengths_training.append(len(tensor_training))

			tensor_label = trainingReadToLabel(trainingRead)
			y.append(tensor_label)
			lengths_labels.append(len(tensor_label))



		#-1 padding
		X = np.array(pad_sequences(X, value=0, padding='post'))
		y = np.array(pad_sequences(y, value=-1, padding='post'))
		print(y.shape)

		lengths_training = np.array(lengths_training).reshape(32,1)
		lengths_labels = np.array(lengths_labels).reshape(32,1)

		#y = y.reshape(y.shape[0],y.shape[1],6)

		return X, y


def variableLength_CTC_loss(y_true, y_pred):

	#global lengths_labels
	#global lengths_training

	lengths_training = np.ones((32,1))*100
	#lengths_labels = np.ones((32,1))*100

	result = ctc_batch_cost(y_true, y_pred, lengths_training, lengths_labels)
	return result

#-------------------------------------------------
#MAIN

#uncomment to train from scratch

readIDs = []
f_readIDs = open('/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_DNAscentTrainingData_highIns_noBrdUScaling_CTC_noGap.IDs','r')
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
params = {'dim': (None,10),
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
op = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=op, metrics=['accuracy'], loss=variableLength_CTC_loss)
print(model.summary())
plot_model(model, to_file='model.png')


#uncomment to load weights from a trainign checkpoint
#model.load_weights(f_checkpoint)

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
