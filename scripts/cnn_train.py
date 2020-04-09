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
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import Input
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import normalize, to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import Sequence
import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
from scipy.stats import halfnorm


folderPath = 'data'
validationSplit = 0.2
inputFiles = [(0., 'trainingData.barcode08.dnascent'),(0.26, 'trainingData.barcode10.dnascent'),(0.5, 'trainingData.barcode11.dnascent'),(0.8, 'trainingData.barcode12.dnascent')]
#inputFiles = [(0., 'trainingData.barcode08.dnascent'),(0.8, 'trainingData.barcode12.dnascent')]
maxLen = 1000


#static params
truePositive = 0.7
trueNegative = 0.95
falsePositive = 1. - trueNegative
llThreshold = 1.25

#for data scaling
maxEvent = 0.
minEvent = 1000.
minEventLength = 1000.
maxEventLength = 0.

#-------------------------------------------------
#
class trainingRead:
	def __init__(self, sixMers, eventMags, eventLengths, llScores, modelMeans, readID, analogueConc):
		self.sixMers = sixMers[0:maxLen]
		self.eventMags = eventMags[0:maxLen]
		self.eventLengths = eventLengths[0:maxLen]
		self.length = len(sixMers)
		self.llScores = llScores[0:maxLen]
		self.modelMeans = modelMeans
		self.readID = readID
		self.analogueConc = analogueConc



#-------------------------------------------------
#one-hot for 6mers
bases = ['A','T','G','C']
k = 6
all6mers = [''.join(i) for i in itertools.product(bases,repeat=k)]
sixMerToInt = {}
IntTosixMer = {}
for i,s in enumerate(all6mers):
	sixMerToInt[s] = i
	IntTosixMer[i] = s


#-------------------------------------------------
#
def trainingReadToTensor(t):

	global maxEvent, minEvent, minEventLength, maxEventLength
	oneSet = []
	for i, s in enumerate(t.sixMers):
		oneHot = [0]*4**6
		index = sixMerToInt[s]
		oneHot[index] = 1
		oneHot.append((t.eventMags[i] - minEvent)/(maxEvent - minEvent) )
		oneHot.append((t.eventLengths[i] - minEventLength)/(maxEventLength - minEventLength))
		oneHot.append((t.modelMeans[i] - minEvent)/(maxEvent - minEvent))# - (t.eventMags[i] - minEvent)/(maxEvent - minEvent) )
		oneSet.append(oneHot)

	return np.array(oneSet)

#-------------------------------------------------
#
def trainingReadToLabel(t):
	label =[]
	for s in t.llScores:
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
#reshape into tensors
def saveRead(trainingRead, readID):

	f = open(folderPath + '/' + readID + '.p', 'wb')
	pickle.dump(trainingRead, f)
	f.close()


#-------------------------------------------------
#pull from training data file
def importFromFile(fname, analogueConc, readIDs):
	print('Parsing training data file...')
	global maxEvent, minEvent, minEventLength, maxEventLength
	sixMers = []
	eventMags = []
	eventLengths = []
	llScores = []
	modelMeans = []
	f = open(fname,'r')
	ctr = 0
	first = True
	readsLoaded = 0
	for line in f:
		if line[0] == '>':

			if not first and len(sixMers) > maxLen:
				readIDs.append(readID)
				tr = trainingRead(sixMers, eventMags, eventLengths, llScores, modelMeans, readID, analogueConc)
				saveRead(tr, readID)
			first = False

			splitLine = line.rstrip().split()
			readID = splitLine[0][1:]

			sixMers = []
			eventMags = []
			eventLengths = []
			llScores = []
			modelMeans = []

			readsLoaded += 1
			print('Reads loaded: ',readsLoaded)
		else:
			splitLine = line.rstrip().split('\t')
			sixMers.append(splitLine[0])
			eventMags.append(float(splitLine[1]))
			eventLengths.append(float(splitLine[2]))
			modelMeans.append(float(splitLine[3]))
			if len(splitLine) == 5:
				llScores.append(splitLine[4])
			else:
				llScores.append('-')

			#sort out min/max
			minEvent = min([minEvent, float(splitLine[1]), float(splitLine[3])])
			maxEvent = max([maxEvent, float(splitLine[1]), float(splitLine[3])])
			minEventLength = min([minEventLength, float(splitLine[2])])
			maxEventLength = max([maxEventLength, float(splitLine[2])])

	f.close()
	return readIDs
	print('Done.')


#-------------------------------------------------
#MAIN
os.system('mkdir '+folderPath)

readIDs = []
for tup in inputFiles:
	readIDs = importFromFile(tup[1], tup[0], readIDs)

print('Scalings:')
print('MinEvent: ',minEvent)
print('MaxEvent: ',maxEvent)
print('MinEventLength: ',minEventLength)
print('MaxEventLength: ',maxEventLength)

random.shuffle(readIDs)
divideIndex = int(validationSplit*len(readIDs))

partition = {'training':readIDs[divideIndex+1:], 'validation':readIDs[0:divideIndex]}
labels = {}

# Parameters
params = {'dim': (maxLen,4**6+3),
          'batch_size': 16,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['training'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)


#-------------------------------------------------
#CNN architecture
model = Sequential()

'''
model.add(Bidirectional(LSTM(16,return_sequences=True),input_shape=(None,4**6+3)))
model.add(TimeDistributed(Dense(100,activation='tanh')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(2,activation='softmax')))
'''
model.add(Conv1D(16,(16),padding='same',activation='tanh',input_shape=(maxLen,4**6+3)))
#model.add(BatchNormalization())
model.add(Conv1D(16,(16),padding='same',activation='tanh'))
#model.add(BatchNormalization())
model.add(Conv1D(16,(16),padding='same',activation='tanh'))
#model.add(BatchNormalization())
model.add(Conv1D(16,(16),padding='same',activation='tanh'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(100,activation='tanh')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(100,activation='tanh')))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(2,activation='softmax')))


model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True) 
model.fit_generator(generator=training_generator, validation_data=validation_generator,epochs=100,verbose=1,callbacks=[es])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
