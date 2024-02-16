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
import time
import pickle
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import Convolution1DReparameterization, DenseReparameterization
from tensorflow_probability.python.distributions import Categorical
from sklearn.utils import shuffle

print('Tensorflow version:',tf.__version__)

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

logPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/logs/trainingLog_DNN_64.csv'
trainingReadLogPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/logs/trainingReads_DNN_64.txt'
valReadLogPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/logs/validationReads_DNN_64.txt'
checkpointPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/checkpoints/64'
validationSplit = 0.1

f_checkpoint = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/checkpoints/20_HMM_partII/weights.03-0.23.h5'

maxLen = 2000

analogueProb = 0.8

#-------------------------------------------------
#
def elbo_loss(labels, logits):
	loss_en = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
	loss_kl = tf.keras.losses.KLD(labels, logits)
	loss = tf.reduce_mean(tf.add(loss_en, loss_kl))
	return loss


#-------------------------------------------------
#
def accuracy(preds, labels):
	return np.mean(np.argmax(preds, axis=-1) == np.argmax(labels, axis=-1))


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
			sequence_tensor.append(oneHot)
		self.sequence_tensor = np.array(sequence_tensor)

		#make proto signal training tensor
		signal_tensor = []
		for i in range(len(self.signal)):
			padded_signals = []

			padded_signals.append( np.mean(self.signal[i]) )
			padded_signals.append( np.std(self.signal[i]) )			
			padded_signals.append( float(len(self.signal[i])) )						

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
    X = Convolution1DReparameterization(filters=F1, kernel_size=f, strides=1, padding='same', activation='relu', name=conv_name_base + '2a')(X)

    # Second component of main path
    X = Convolution1DReparameterization(filters=F2, kernel_size=f, strides=1, padding='same', activation='relu', name=conv_name_base + '2b')(X)

    # Third component of main path
    X = Convolution1DReparameterization(filters=F3, kernel_size=f, strides=1, padding='same', activation='relu', name=conv_name_base + '2c')(X)

    # Fourth component of main path
    X = Convolution1DReparameterization(filters=F4, kernel_size=f, strides=1, padding='same', activation='relu', name=conv_name_base + '2d')(X)

    # Fifth component of main path
    X = Convolution1DReparameterization(filters=F5, kernel_size=f, strides=1, padding='same', activation='relu', name=conv_name_base + '2e')(X)

    # Sixth component of main path
    X = Convolution1DReparameterization(filters=F6, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2f')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Convolution1DReparameterization(filters=F6, kernel_size=f, strides=1, padding='same', name=conv_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a relu activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)


    return X


#-------------------------------------------------
#
def buildModel():
    
    # Define the input as a tensor with shape input_shape
    sequence_input = Input(shape=(None,5))
    
    signal_input = Input(shape=(None,3))

    X = Concatenate(axis=-1)([sequence_input,signal_input])

    # Stage 1
    X = Convolution1DReparameterization(64, 3, strides = 1, padding='same', activation='relu', name = 'conv1')(X)

    X = Convolution1DReparameterization(128, 5, strides = 1, padding='same', activation='relu', name = 'conv5')(X)

    X = Convolution1DReparameterization(256, 9, strides = 1, padding='same', activation='relu', name = 'conv6')(X)

    '''
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
    '''


    X = Convolution1DReparameterization(64, 3, strides = 1, padding='same', activation='relu', name = 'conv2')(X)

    X = Convolution1DReparameterization(64, 3, strides = 1, padding='same', activation='relu', name = 'conv3')(X)

    X = Convolution1DReparameterization(64, 3, strides = 1, padding='same', activation='relu', name = 'conv4')(X)

    # Output layer
    X = TimeDistributed(DenseReparameterization(3, name='fc_final'))(X)
    
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

			label.append([1., 0., 0.])

	elif whichSet == 1: # BrdU read
		for i, s in enumerate(t.analogueCalls):
			if s == '-': #not thymidine
				label.append([1., 0., 0.])
			else:
				label.append([0.2, 0.8, 0.])


	elif whichSet == 2: # EdU read
		for i, s in enumerate(t.analogueCalls):
			if s == '-': #not thymidine
				label.append([1., 0., 0.])
			else:
				label.append([0.2, 0., 0.8])
	return np.array(label)	


#-------------------------------------------------
#BUILD MODEL
model = buildModel()
op = tf.keras.optimizers.Adam(lr=0.001)


#-------------------------------------------------
#UPDATE GRADIENTS
def train_step(batchLogits, batchLabels):
	with tf.GradientTape() as tape:
		logits = model(batchLogits)
		loss = elbo_loss(batchLabels, logits)
	gradients = tape.gradient(loss, model.trainable_variables)
	op.apply_gradients(zip(gradients, model.trainable_variables))
	return loss


#-------------------------------------------------
#IMPORT DATA

#uncomment to train from scratch

filepaths = ['/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode24/trainingReads_slices',
'/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_11_MJ_ONT_Brdu_48hr_training_v14_5khz_fast5/trainingReads_slices',
'/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode22/trainingReads_slices']

maxReads = [2000,
2000,
2000]

trainPaths = []
valPaths = []

for index, directory in enumerate(filepaths):

	count = 0
	allPaths = []
	for fcount, fname in enumerate(os.listdir(directory)):
		allPaths.append(directory + '/' + fname)
		count += 1
		if count > maxReads[index]:
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
'''

#-------------------------------------------------
#TRAINING LOOP
times = []
accs = []
val_accs = []
losses = []
val_losses = []
batch_size=32
epochs=15

def train(model, trainPaths):
	for i in range(epochs):
		tic = time.time()
		num_batches = int(len(trainPaths)/batch_size)
		for b in range(num_batches):
		
			#index bounds for the batch
			top = batch_size*(b+1)
			bottom = top-batch_size
			
			X1 = np.empty((batch_size, maxLen, 5))
			X2 = np.empty((batch_size, maxLen, 3))
			y = np.empty((batch_size, maxLen, 3), dtype=float)

			#spin up training data and labels
			for i, ID in enumerate(trainPaths[bottom:top]):

				whichSet = -1
				if 'barcode22' in ID:
					whichSet = 0
				elif '2023_07_11_MJ_ONT_Brdu_48hr_training_v14_5khz_fast5' in ID:
					whichSet = 1
				elif 'barcode24' in ID:
					whichSet = 2
				if whichSet == -1:
					print('setting analogue failed')
					sys.exit()
				
				trainingRead = pickle.load(open(ID, "rb"))

				# input tensors
				sequence_tensor, signal_tensor = trainingReadToTensor(trainingRead)
				X1[i,] = sequence_tensor
				X2[i,] = signal_tensor.reshape(signal_tensor.shape[0], signal_tensor.shape[1])

				# labels
				y[i] = trainingReadToLabel(trainingRead,whichSet)

			#randomly sample on label probability distributions
			sampled_labels = tf.one_hot(Categorical(probs=y).sample(), depth=3)	
		
			#update gradients, calculate batch loss and accuracy
			loss = train_step([X1, X2], sampled_labels)
			preds = model([X1, X2])
			acc = accuracy(preds, sampled_labels)

			print("Batch: {}: loss = {:7.3f} , accuracy = {:7.3f}".format(b, loss, acc), end='\r')

		'''
		#compute on whole dataset
		loss = train_step(dat_train, tf.keras.utils.to_categorical(lab_train, num_classes=2))
		preds = model(dat_train)
		acc = accuracy(preds, tf.keras.utils.to_categorical(lab_train, num_classes=2))
		val_preds = model(dat_test)
		val_loss = elbo_loss(tf.keras.utils.to_categorical(lab_test, num_classes=2), val_preds)
		val_acc = accuracy(tf.keras.utils.to_categorical(lab_test, num_classes=2), val_preds)

		accs.append(acc)
		losses.append(loss)
		val_accs.append(val_acc)
		val_losses.append(val_loss)
		tac = time.time()
		train_time = tac-tic
		times.append(train_time)
		'''
		
		#shuffle between epochs
		trainPaths = shuffle(trainPaths)

		print("Epoch: {}: loss = {:7.3f} , accuracy = {:7.3f}, val_loss = {:7.3f}, val_acc={:7.3f} time: {:7.3f}".format(i, loss, acc, val_loss, val_acc, train_time))
		
train(model, trainPaths)
