import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Embedding, Flatten, MaxPooling1D,AveragePooling1D
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Activation, LSTM
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional, Reshape, Activation, BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.python.keras import Input
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import normalize, to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras import backend
import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
from scipy.stats import halfnorm

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_DNAscentTrainingData'
logPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingLog13.csv'
trainingReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingReadsUsed13.txt'
valReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/valReadsUsed13.txt'
checkpointPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/checkpoints13'
validationSplit = 0.2

maxLen = 3000
maxReads = 100000

#static params
truePositive = 0.7
trueNegative = 0.95
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
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3, 'N':4}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C', 4:'N'}


#-------------------------------------------------
#
def trainingReadToTensor(t):

	oneSet = []
	for i, s in enumerate(t.sixMers):

		#base
		oneHot = [0]*5
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
		X = np.empty((self.batch_size, *self.dim))
		y = np.empty((self.batch_size, maxLen, 2), dtype=float)
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
			
		y = y.reshape(y.shape[0],y.shape[1],2)
		return X, y


#-------------------------------------------------
#MAIN

#uncomment to train from scratch
readIDs = []
f_readIDs = open('/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_DNAscentTrainingData.IDs','r')
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
params = {'dim': (maxLen,13),
          'batch_size': 32,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['training'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)


#-------------------------------------------------
#CNN architecture
model = Sequential()
model.add(Conv1D(64,(4),padding='same',input_shape=(None,13)))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(Conv1D(64,(4),padding='same'))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(MaxPooling1D(pool_size=4, strides=1, padding='same'))

model.add(Conv1D(64,(8),padding='same'))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(Conv1D(64,(8),padding='same'))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(MaxPooling1D(pool_size=4, strides=1, padding='same'))

model.add(Conv1D(64,(16),padding='same'))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(Conv1D(64,(16),padding='same'))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(MaxPooling1D(pool_size=4, strides=1, padding='same'))

model.add(TimeDistributed(Dense(100)))
model.add(Activation("tanh"))
model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(100)))
model.add(Activation("tanh"))
model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(2,activation='softmax')))
op = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy')
print(model.summary())

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
