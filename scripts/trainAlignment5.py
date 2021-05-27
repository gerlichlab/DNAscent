import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json, Model
from tensorflow.python.keras.layers import Dense, Dropout, ZeroPadding1D, Concatenate
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
from tensorflow.python.keras.losses import CategoricalCrossentropy
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

logPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingLog_alignment5.csv'
trainingReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/trainingReadsUsed_alignment5.txt'
valReadLogPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/valReadsUsed_alignment5.txt'
checkpointPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/alignment5'
validationSplit = 0.2

#f_checkpoint = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/alignment4/weights.01-0.13.h5'

maxLen = 4000

#static params
truePositive = 0.5
trueNegative = 0.9
falsePositive = 1. - trueNegative
llThreshold = 1.25


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sixMers, read_eventMeans, read_eventLength, read_positions, readID, analogueConc, logLikelihood, matchingReadDic):

		self.sixMers = read_sixMers[0:maxLen]
		self.eventMean = read_eventMeans[0:maxLen]
		self.eventLength = read_eventLength[0:maxLen]
		self.logLikelihood = logLikelihood[0:maxLen]
		self.readID = readID
		self.positions = read_positions
		self.analogueConc = -1

		if not len(self.sixMers) == len(self.eventMean) == len(self.eventLength) == len(self.logLikelihood):
			print(len(self.sixMers), len(self.logLikelihood))
			print("Length Mismatch")
			sys.exit()

#-------------------------------------------------
#one-hot encodings

#bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}

#alignment actions
actionToInt = {'NewMatch':0,'Dwell':1,'Insertion':2,'Deletion':3}
intToAction = {0:'NewMatch',1:'Dwell',2:'Insertion',3:'Deletion'}

#-------------------------------------------------
#
def trainingReadToTensor(t):
	
	#make event tensor
	eventTensor = []
	for i, s in enumerate(t.sixMers[0:1000]):
		for j in range(len(t.eventMean[i])):	
			eventTensor.append([t.eventMean[i][j], t.eventLength[i][j]])
	
	eventTensor = np.array(eventTensor[0:1000])			

	#make sequence tensor 
	sequenceTensor = []
	for i, s in enumerate(t.sixMers[0:2000]):

		if s == 'NNNNNN':
			continue

		oneHot = [0]*4
		index = baseToInt[s[0]]
		oneHot[index] = 1
		sequenceTensor.append(oneHot)
		
	sequenceTensor = np.array(sequenceTensor[0:1000])			

	return eventTensor, sequenceTensor

#-------------------------------------------------
#
def trainingReadToLabel(t):

	label = []
	weights = []
	for i, s in enumerate(t.sixMers[0:1000]):

		i_fix = i

		if s == 'NNNNNN':
			oneHot = [0]*4
			index = actionToInt['Insertion']
			oneHot[index] = 1
			label.append(oneHot)
			weights.append(100)
		else:

			if t.positions[i_fix] != t.positions[i_fix-1] + 1: #we've moved

				#moved to a new match
				if t.positions[i_fix] != t.positions[i_fix-1]:
					oneHot = [0]*4
					index = actionToInt['NewMatch']
					oneHot[index] = 1
					label.append(oneHot)
					weights.append(2.5)

				#there were intervening deletions
				else:
					for j in range(t.positions[i_fix] - t.positions[i_fix-1] - 1):
						oneHot = [0]*4
						index = actionToInt['Deletion']
						oneHot[index] = 1
						label.append(oneHot)
						weights.append(100)
			else:


				oneHot = [0]*4
				index = actionToInt['Dwell']
				oneHot[index] = 1
				label.append(oneHot)
				weights.append(1)
	label = label[0:1000]
	return np.array(label),np.array(weights)


#-------------------------------------------------
#
def buildModel(input_shape = (64, 64, 3), classes = 6):

	# Define the input as a tensor with shape input_shape
	input_events = Input(name='events_input',shape=(None,2))
	input_baseSequence = Input(name='bases_input',shape=(None,4))

	#events stack
	E = Bidirectional(LSTM(16, return_sequences=True))(input_events)
	E = Bidirectional(LSTM(16, return_sequences=True))(E)
	E = Bidirectional(LSTM(16, return_sequences=True))(E)
	
	#base sequence stack
	S = Bidirectional(LSTM(16, return_sequences=True))(input_baseSequence)
	S = Bidirectional(LSTM(16, return_sequences=True))(S)
	S = Bidirectional(LSTM(16, return_sequences=True))(S)

	D = Concatenate()([E,S])
	D = Bidirectional(LSTM(16, return_sequences=True))(D)
	D = Bidirectional(LSTM(16, return_sequences=True))(D)
	D = Bidirectional(LSTM(16, return_sequences=True))(D)

	D = Dense(1000, activation="tanh")(D)
	D = Dropout(0.5)(D)
	D = TimeDistributed(Dense(4,activation='softmax'))(D)

	# Create model
	model = Model(inputs = [input_events,input_baseSequence], outputs = D, name='Alignment')

	return model


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

		# Generate data
		E = []
		S = []
		y = []
		w = []

		for i, ID in enumerate(list_IDs_temp):

			#pull data for this 6mer from the appropriate pickled read
			trainingRead = pickle.load(open(ID, "rb"))

			eventTensor, sequenceTensor = trainingReadToTensor(trainingRead)
			label, weights = trainingReadToLabel(trainingRead)

			E.append(eventTensor)
			S.append(sequenceTensor)
			y.append(label)
			w.append(weights)

		E = np.array(E).reshape(self.batch_size,1000,2)
		S = np.array(S).reshape(self.batch_size,1000,4)
		y = np.array(y).reshape(self.batch_size,1000,4)
		w = np.array(w).reshape(self.batch_size,1000)
		inputs = {'events_input': E, 'bases_input': S}

		return inputs,y,w

#-------------------------------------------------
#MAIN

#uncomment to train from scratch

filepaths = ['/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_alignment_bc08']

maxReads = [10000]

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
params = {'dim': (None,7),
          'batch_size': 32,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['training'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

#-------------------------------------------------
#CNN architecture

model = buildModel((None,7), 2)
op = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
loss = CategoricalCrossentropy(label_smoothing=0.15)
model.compile(optimizer=op, metrics=['accuracy'], loss=loss, sample_weight_mode="temporal")
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
