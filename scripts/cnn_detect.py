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

maxLen = 3000
maxReads = 100000
f_checkpoint = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/checkpoints9/weights.07-0.08.h5'
folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data8'


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sixMers, read_eventMeans, read_eventStd, read_stutter, read_lengthMeans, read_lengthStd, read_modelMeans, read_modelStd, read_positions, readID, analogueConc, readID2calls):
		self.sixMers = read_sixMers[0:maxLen]
		self.eventMean = read_eventMeans[0:maxLen]
		self.eventStd = read_eventStd[0:maxLen]
		self.stutter = read_stutter[0:maxLen]
		self.lengthMean = read_lengthMeans[0:maxLen]
		self.lengthStd = read_lengthStd[0:maxLen]
		self.modelMeans = read_modelMeans[0:maxLen]
		self.modelStd = read_modelStd[0:maxLen]
		self.readID = readID
		self.analogueConc = analogueConc

		#get log likelihoods from DNAscent detect
		positiveCallsOnRef = readID2calls[readID]
		positiveCallsOnRef = dict(positiveCallsOnRef)
		logLikelihood = []
		for p in read_positions[0:maxLen]:
			if p in positiveCallsOnRef:
				logLikelihood.append(positiveCallsOnRef[p])
			else:
				logLikelihood.append('-')
		self.logLikelihood = logLikelihood

		gaps = [1]
		for i in range(1, maxLen):
			gaps.append(read_positions[i] - read_positions[i-1])
		self.gaps = gaps

		if not len(self.sixMers) == len(self.eventMean) == len(self.eventStd) == len(self.stutter) == len(self.lengthMean) == len(self.lengthStd) == len(self.modelMeans) == len(self.modelStd) == len(self.gaps):
			print("Length Mismatch")
			sys.exit()


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
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
print(model.summary())

model.load_weights(f_checkpoint)


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
#main

print('Loading read IDs...')
readIDs = []
f_readIDs = open('/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data8.IDs','r')
for line in f_readIDs:
	readIDs.append(line.rstrip())
f_readIDs.close()

readIDs = readIDs[maxReads+1:]

thresholds = np.array(range(1,10))/10.
thresholds2calls_zero = {}
thresholds2attempts_zero = {}
thresholds2calls_twenty = {}
thresholds2attempts_twenty = {}
thresholds2calls_fifty = {}
thresholds2attempts_fifty = {}
thresholds2calls_eighty = {}
thresholds2attempts_eighty = {}

for t in thresholds:
	thresholds2calls_zero[t] = 0
	thresholds2attempts_zero[t] = 0
	thresholds2calls_twenty[t] = 0
	thresholds2attempts_twenty[t] = 0
	thresholds2calls_fifty[t] = 0
	thresholds2attempts_fifty[t] = 0
	thresholds2calls_eighty[t] = 0
	thresholds2attempts_eighty[t] = 0

count = 0
for ID in readIDs:

	#attempts = 0
	#calls = 0

	f = open(folderPath + '/' + ID + '.p', "rb")
	tr = pickle.load(f)
	f.close()
	
	tensor = trainingReadToTensor(tr)
	tensor = tensor.reshape(1,tensor.shape[0],tensor.shape[1])
	pred = model.predict(tensor)
	for i, s in enumerate(tr.sixMers):
		if s[0] != 'T':
			continue
		#print(pred[0,i,1], s, tr.analogueConc )

		
		if tr.analogueConc == 0.8:
			for t in thresholds:
				if pred[0,i,1] > t:
					thresholds2calls_eighty[t] += 1
				thresholds2attempts_eighty[t] += 1

		if tr.analogueConc == 0.5:
			for t in thresholds:
				if pred[0,i,1] > t:
					thresholds2calls_fifty[t] += 1
				thresholds2attempts_fifty[t] += 1

		if tr.analogueConc == 0.26:
			for t in thresholds:
				if pred[0,i,1] > t:
					thresholds2calls_twenty[t] += 1
				thresholds2attempts_twenty[t] += 1

		if tr.analogueConc == 0.0:
			for t in thresholds:
				if pred[0,i,1] > t:
					thresholds2calls_zero[t] += 1
				thresholds2attempts_zero[t] += 1



	count += 1
	print(count)
	if count == 10000:
		break

		
x = []
y = []
y2 = []
y3 = []
for t in thresholds:
	falsePositives = float(thresholds2calls_zero[t])/thresholds2attempts_zero[t]
	truePositives = (float(thresholds2calls_fifty[t])/thresholds2attempts_fifty[t])#/0.49
	truePositives2 = (float(thresholds2calls_eighty[t])/thresholds2attempts_eighty[t])#/0.79
	truePositives3 = (float(thresholds2calls_twenty[t])/thresholds2attempts_twenty[t])#/0.26
	x.append(falsePositives)
	y.append(truePositives)
	y2.append(truePositives2)
	y3.append(truePositives3)
fig, ax = plt.subplots()
plt.plot(x,y3,label='26% by Mass Spec')
plt.plot(x,y,label='49% by Mass Spec')
plt.plot(x,y2,label='79% by Mass Spec')
for i,txt in enumerate(thresholds):
	ax.annotate(str(txt),(x[i],y[i]))
	ax.annotate(str(txt),(x[i],y2[i]))
	ax.annotate(str(txt),(x[i],y3[i]))

plt.legend()
plt.ylim(-0.1,1.1)
plt.ylabel('Calls/Attempts')
plt.xlabel('False Positives')
plt.savefig('benchmark.pdf')
