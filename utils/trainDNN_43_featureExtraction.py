import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Attention, Conv1D, Activation, BatchNormalization, SeparableConv1D, TimeDistributed, MultiHeadAttention, AdditiveAttention, Attention, Concatenate, Masking, LSTM, GRU, Bidirectional, Add, SeparableConv1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError, BinaryFocalCrossentropy
from tensorflow.keras.utils import pad_sequences, Sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import numpy as np
import random
import os
import pickle

print('Tensorflow version:',tf.__version__)

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

logPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/logs/trainingLog_DNN_43_transfer.csv'
trainingReadLogPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/logs/trainingReads_DNN_43_transfer.txt'
valReadLogPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/logs/validationReads_DNN_43_transfer.txt'
checkpointPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/checkpoints/43'
validationSplit = 0.1

f_checkpoint = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/checkpoints/43/weights.01-0.20.h5'

maxLen = 2000

#static params
truePositive_BrdU = 0.8
truePositive_EdU = 0.8
trueNegative = 0.9
falsePositive = 1. - trueNegative
llThreshold = 0.5
incorporationEstimate = 0.8

maxEvents = 20

labelSmoothing = 0.01


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sixMers, read_signals, read_modelMeans, read_positions, read_analogueCalls, readID, label):
		self.sixMers = read_sixMers[0:maxLen]
		self.signal = read_signals[0:maxLen]
		self.modelMeans = read_modelMeans[0:maxLen]
		self.analogueCalls = read_analogueCalls[0:maxLen]
		self.readID = readID
		self.label = label
		
		#make proto sequence training tensor
		sequence_tensor = []
		for i, s in enumerate(self.sixMers):		

			#base
			oneHot = [0]*4
			index = baseToInt[s[0]]
			oneHot[index] = 1
			oneHot.append(self.modelMeans[i])
			oneHot.append(float(len(self.signal[i])))
			sequence_tensor.append(oneHot)
		self.sequence_tensor = np.array(sequence_tensor)

		#make proto signal training tensor
		signal_tensor = []
		for i in range(len(self.signal)):
			padded_signals = []
			for j in range(len(self.signal[i])):			
				padded_signals.append(self.signal[i][j])		
			
			#pad/truncate to uniform maxRaw length
			if len(padded_signals) < maxRaw:		
				for b in range(maxRaw - len(padded_signals)):		
					padded_signals.append(0.)
			padded_signals = padded_signals[0:maxRaw]
			signal_tensor.append(np.array(padded_signals))			
		self.signal_tensor = np.array(signal_tensor)


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

    # Final step: Add shortcut value to main path, and pass it through a tanh activation
    X = Add()([X, X_shortcut])
    X = Activation('tanh')(X)

    return X


#-------------------------------------------------
#
def buildModel(input_shape = (64, 64, 3), classes = 6):
    
    # Define the input as a tensor with shape input_shape
    sequence_input = Input(shape=(None,6))
    
    signal_input = Input(shape=(None,maxEvents, 1))
    X = TimeDistributed(Masking(mask_value=0.0))(signal_input)
    X = TimeDistributed(GRU(16,return_sequences=True))(X)
    X = TimeDistributed(GRU(10,return_sequences=False))(X)

    X = Concatenate(axis=-1)([sequence_input,X])

    # Stage 1
    X = Conv1D(64, 3, strides = 1, padding='same', name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv1')(X)
    X = Activation('tanh')(X)

    # Stage 2
    X = convolutional_block(X, f = 5, filters = [64, 64, 64, 64, 64, 64], stage = 2, block='a', s = 1)

    # Stage 3
    X = convolutional_block(X, f=5, filters=[64, 64, 64, 64, 64, 64], stage=3, block='a', s=2)

    # Stage 4
    X = convolutional_block(X, f=9, filters=[128, 128, 128, 128, 128, 128], stage=4, block='a', s=2)

    # Stage 5
    X = convolutional_block(X, f=9, filters=[128, 128, 128, 128, 128, 128], stage=5, block='a', s=2)

    # Stage 6
    X = convolutional_block(X, f=17, filters=[256, 256, 256, 256, 256, 256], stage=6, block='a', s=2)

    X = Conv1D(256, 3, strides = 1, padding='same', name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv2')(X)
    X = Activation('tanh')(X)

    X = Conv1D(128, 3, strides = 1, padding='same', name = 'conv3', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv3')(X)
    X = Activation('tanh')(X)

    X = Conv1D(64, 3, strides = 1, padding='same', name = 'conv4', kernel_initializer = glorot_uniform(seed=0))(X)

    # Output layer
    X = TimeDistributed(Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0)))(X)
  
    # Create model
    model = Model(inputs = [sequence_input, signal_input] , outputs = X, name='R10_BrdU_EdU')

    return model


#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}


#-------------------------------------------------
#
def trainingReadToTensor(t):

	return t.sequence_tensor, t.signal_tensor


#-------------------------------------------------
#
def trainingReadToLabel(t,whichSet):
	label =[]

	if whichSet == 0.:
		for s in t.modelMeans:

			label.append([0.98, 0.01, 0.01])

	elif whichSet == 1: # BrdU read
		for i, s in enumerate(t.analogueCalls):
			if s == '-': #not thymidine
				label.append([0.98, 0.01, 0.01])
			else:
				score = float(t.analogueCalls[i])

				if score > llThreshold:
					l = (truePositive_BrdU*incorporationEstimate)/(truePositive_BrdU*incorporationEstimate + falsePositive*(1-incorporationEstimate))
					label.append([1.-l-0.005, l-0.005, 0.01])
				else:
					l = ((1-truePositive_BrdU)*incorporationEstimate)/((1-truePositive_BrdU)*incorporationEstimate + (1-falsePositive)*(1-incorporationEstimate))
					label.append([1.-l-0.005, l-0.005, 0.01])

	elif whichSet == 2: # EdU read
		for i, s in enumerate(t.analogueCalls):
			if s == '-': #not thymidine
				label.append([0.98, 0.01, 0.01])
			else:

				score = float(t.analogueCalls[i])

				if score > llThreshold:
					l = (truePositive_EdU*incorporationEstimate)/(truePositive_EdU*incorporationEstimate + falsePositive*(1-incorporationEstimate))
					label.append([1.-l-0.005, 0.01, l-0.005])
				else:
					l = ((1-truePositive_EdU)*incorporationEstimate)/((1-truePositive_EdU)*incorporationEstimate + (1-falsePositive)*(1-incorporationEstimate))
					label.append([1.-l-0.005, 0.01, l-0.005])

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
		X1 = np.empty((self.batch_size, *self.dim[0]))
		X2 = np.empty((self.batch_size, *self.dim[1]))
		y = np.empty((self.batch_size, maxLen, 3), dtype=float)
		w = np.empty((self.batch_size, maxLen), dtype=float)

		# Generate data
		for i, ID in enumerate(list_IDs_temp):

			whichSet = -1
			if 'barcode03' in ID:
				whichSet = 0
			elif 'barcode01' in ID:
				whichSet = 1
			elif 'barcode02' in ID:
				whichSet = 2
			if whichSet == -1:
				print('setting analogue failed')
				sys.exit()

			trainingRead = pickle.load(open(ID, "rb"))

			# input tensors
			sequence_tensor, signal_tensor = trainingReadToTensor(trainingRead)
			X1[i,] = sequence_tensor
			X2[i,] = signal_tensor.reshape(signal_tensor.shape[0], signal_tensor.shape[1], 1)

			# labels
			y[i] = trainingReadToLabel(trainingRead,whichSet)

			# weights
			w[i] = np.ones(maxLen)

		y = y.reshape(y.shape[0],y.shape[1],3)
		w = w.reshape(w.shape[0],w.shape[1])
		return [X1, X2], y, w


#-------------------------------------------------
#MAIN

#uncomment to train from scratch

filepaths = ['/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_09_08_FT_ONT_Pfal_R10_trainingset/barcode03/trainingReads_slices_raw',
'/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_09_08_FT_ONT_Pfal_R10_trainingset/barcode02/trainingReads_slices_raw',
'/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_09_08_FT_ONT_Pfal_R10_trainingset/barcode01/trainingReads_slices_raw']

maxReads = [2900,
1450,
1450]

trainPaths = []

for index, directory in enumerate(filepaths):

	count = 0
	allPaths = []
	for fcount, fname in enumerate(os.listdir(directory)):
		allPaths.append(directory + '/' + fname)
		count += 1
		if count > maxReads[index]:
			break

	random.shuffle(allPaths)
	trainPaths += allPaths
	
#record the reads we're using for training and val
f_trainingReads = open(trainingReadLogPath,'w')
for ri in trainPaths:
	f_trainingReads.write(ri+'\n')
f_trainingReads.close()

f_valReads = open(valReadLogPath,'w')
f_valReads.close()

partition = {'training':trainPaths}


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
params = {'dim': [(maxLen,6), (maxLen,maxEvents,1)],
          'batch_size': 16,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['training'], labels, **params)

#-------------------------------------------------
#CNN architecture

with strategy.scope():

	model = buildModel((None,15,6), 3)

	#uncomment to load weights from a trainign checkpoint
	model.load_weights(f_checkpoint)
	
	for l in model.layers:
		l.trainable = False

	#unfreeze last 5 layers
	for i in range(1,9):
	
		#don't unfreeze batch norm
		if 'bn' in model.layers[-1*i].name:
			model.layers[-1*i].trainable = False
		else: 		
			model.layers[-1*i].trainable = True 
			
	op = Adam(learning_rate=0.0001)
	model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy', sample_weight_mode="temporal")
	print(model.summary(show_trainable=True))

	#callbacks
	chk = ModelCheckpoint(checkpointPath + '/feature_extraction/weights.{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	csv = CSVLogger(logPath, separator=',', append=False)

	#generator fit
	model.fit_generator(generator=training_generator, epochs=20, verbose=1, callbacks=[chk,csv])
