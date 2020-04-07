import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Embedding, Flatten, MaxPooling1D,AveragePooling1D
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Activation
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
from scipy.stats import halfnorm


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
		y = np.empty((self.batch_size), dtype=float)
		#print(list_IDs_temp)
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			splitID = ID.split(';')
			
			#TODO: think about writing something here that transforms 6mers into one-hot encodings on the fly (may not have to load this from disk)
			X[i,] = np.load('data/' + splitID[0] + '/' + splitID[1] + '.npy')

			# Store class
			y[i] = self.labels[ID]
			#print(i)

		return X, y

input_f = sys.argv[1]
folderPath = 'data'
validationSplit = 0.2

os.system('mkdir '+folderPath)
partition = {'training':[], 'validation':[]}
labels = {}

#-------------------------------------------------
class trainingSet:
	def __init__(self, sixMers, eventMags, eventLengths, name):
		self.sixMers = sixMers
		self.eventMags = eventMags
		self.eventLengths = eventLengths
		self.length = len(sixMers)
		self.name = name


#-------------------------------------------------
#one-hot for 6mers
bases = ['A','T','G','C']
k = 6
all6mers = [''.join(i) for i in itertools.product(bases,repeat=k)]
sixMerToInt = {}
IntTosixMer = {}
for i,s in enumerate(all6mers):
	sixMerToInt[s]=i
	IntTosixMer[i]=s


#-------------------------------------------------
#pull from training data file
print('Parsing training data file...')
trainingSets = []
sixMers = []
eventMags = []
eventLengths = []
maxLen = 0
f = open(input_f,'r')
ctr = 0
readsLoaded = 0
for line in f:
	if line[0] == '>':

		splitLine = line.rstrip().split()
		readID = splitLine[0][1:]
		ctr = 0
		os.system('mkdir ' + folderPath + '/' + readID)
		readsLoaded += 1
	elif line[0] == '-':
		ctr += 1

	if line[0] == '>' or line[0] == '-':
		if len(sixMers) > 0:
			ctr += 1
			trainingSets.append(trainingSet(sixMers,eventMags,eventLengths,readID + ';'+str(ctr)))
			if len(sixMers) > maxLen:
				maxLen = len(sixMers)

		if ctr > 500:
			break

		sixMers = []
		eventMags = []
		eventLengths = []
	else:
		splitLine = line.rstrip().split()
		sixMers.append(splitLine[0])
		eventMags.append(float(splitLine[1]))
		eventLengths.append(float(splitLine[2]))
f.close()
print('Done.')


#-------------------------------------------------
#reshape into tensors
print('Reshaping into tensors...')
for t in trainingSets:
	oneSet = []
	for i, s in enumerate(t.sixMers):
		oneHot = [0]*4**6
		index = sixMerToInt[s]
		oneHot[index] = 1
		oneHot.append(t.eventMags[i])
		oneHot.append(t.eventLengths[i])
		oneSet.append(oneHot)

	#padding
	if len(t.sixMers) < maxLen:
		for k in range(0, maxLen - len(t.sixMers)):
			oneHot = [0]*4**6
			oneHot += [0, 0]
			oneSet.append(oneHot)

	#labels
	if random.random() < validationSplit:
		partition['validation'].append(t.name)
	else:
		partition['training'].append(t.name)

	if random.random() < 0.5:

		if t.name not in labels:
			labels[t.name] = 0.			
		else:
			labels[t.name].append(0.)
	else:	
		if t.name not in labels:
			labels[t.name] = 1.			
		else:
			labels[t.name].append(1.)		

	name = t.name.split(';')
	np.save(folderPath + '/' + name[0] + '/' + name[1], np.array(oneSet))


print(partition)
print(labels)
# Parameters
params = {'dim': (maxLen,4**6+2),
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
model.add(Conv1D(16,(4),padding='same',activation='relu',input_shape=(maxLen,4**6+2)))
model.add(MaxPooling1D(pool_size=4,strides=2,padding='same'))
model.add(Conv1D(16,(4),padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=4,strides=2,padding='same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='softmax'))
op = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=op, metrics=['accuracy'], loss='binary_crossentropy')
print(model.summary())

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True) 
model.fit_generator(generator=training_generator, validation_data=validation_generator)#,epochs=1000,callbacks=[es], verbose=2)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
