import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import SpatialDropout1D, Input, Dense, Attention, Conv1D, Activation, BatchNormalization, SeparableConv1D, TimeDistributed, MultiHeadAttention, AdditiveAttention, Attention, Concatenate, Masking, LSTM, GRU, Bidirectional, Add, SeparableConv1D, Dropout, Softmax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError, BinaryFocalCrossentropy
from tensorflow.keras.utils import pad_sequences, Sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import numpy as np
import random
import os
import pickle
import sys

import tensorflow_probability as tfp
from tensorflow_probability.python.layers import Convolution1DReparameterization, DenseReparameterization

maxReads = 200

dir_BrdUtest = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_11_MJ_ONT_Brdu_48hr_training_v14_5khz_fast5/testReads_raw'
dir_EdUtest = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode24/testReads_raw'
dir_Thymtest = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode22/testReads_raw'

probThresholds = np.array(range(0,21))/20.

checkpoint_fn = sys.argv[1]
maxEvents = 20

dropout_prob = 0.5

baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}

#-------------------------------------------------
#
def kmer2index(kmer,kmer_len):

	p = 1;
	r = 0;
	for i in range(kmer_len):

		r += baseToInt[kmer[kmer_len-i-1]] * p;
		p *= 4;
	return r+1
	

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
def buildModel(classes, core_embedding_layer, residual_embedding_layer):
    
    #core base sequence input
    core_sequence_input = Input(shape=(2000))
    X_sequence_core = core_embedding_layer(core_sequence_input)
    
    #residual base sequence input
    residual_sequence_input = Input(shape=(2000))
    X_sequence_residual = residual_embedding_layer(residual_sequence_input)    
    
    signal_input = Input(shape=(None,maxEvents, 1))
    X_signal = TimeDistributed(Masking(mask_value=0.0))(signal_input)
    X_signal = TimeDistributed(GRU(16,return_sequences=True))(X_signal)
    X_signal = TimeDistributed(GRU(16,return_sequences=False))(X_signal)

    X = Concatenate(axis=-1)([X_sequence_core, X_sequence_residual, X_signal])

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
    model = Model(inputs = [core_sequence_input, residual_sequence_input, signal_input] , outputs = X, name='R10_BrdU_EdU')

    return model


#-------------------------------------------------
#
def trainingReadToTensor(t):

	core_seq = []
	residual_seq = []
	for kmer in t.kmers:
		core = kmer[2:7]
		residual = kmer[0:2] + kmer[7:]
		core_seq.append(kmer2index(core,5))
		residual_seq.append(kmer2index(residual,4))		

	return np.array(core_seq).reshape(2000), np.array(residual_seq).reshape(2000), np.array(t.modelMeans).reshape(2000,1), t.signal_tensor


#-------------------------------------------------
#
def checkTestDirectory(dirName, model):

	print(dirName)

	#initialise call and attempts dictionaries
	dict_calls_brdu = {}
	dict_calls_edu = {}
	dict_attempts = {}
	for p in probThresholds:
		dict_calls_brdu[p] = 0
		dict_calls_edu[p] = 0
		dict_attempts[p] = 0

	#iterate on pickled test reads
	for fcount, fname in enumerate(os.listdir(dirName)):
	
		if fcount > maxReads:
			break
			
		if fcount % 10 == 0:
			print('Read: ',fcount)
	
		pickledRead = dirName + '/' + fname
		testRead = pickle.load(open(pickledRead, "rb"))
		
		#build and shape input tensors
		core_sequence_tensor, residual_sequence_tensor, model_tensor, signal_tensor = trainingReadToTensor(testRead)
		core_sequence_tensor = core_sequence_tensor.reshape(1,core_sequence_tensor.shape[0])
		residual_sequence_tensor = residual_sequence_tensor.reshape(1,residual_sequence_tensor.shape[0])
		model_tensor = model_tensor.reshape(1,model_tensor.shape[0],model_tensor.shape[1])
		#print(signal_tensor.shape)
		signal_tensor = signal_tensor.reshape(1,signal_tensor.shape[0],signal_tensor.shape[1],1)		

		pred = model.predict((core_sequence_tensor, residual_sequence_tensor, signal_tensor))
		pred = pred.reshape(pred.shape[1], 3)
		for i in range(pred.shape[0]):
		
			#skip non-thymidine positions
			if testRead.kmers[i][4] != 'T':
				continue

			thymProb = pred[i,0]
			brduProb = pred[i,1]
			eduProb = pred[i,2]
			

			for p in probThresholds:
			
				if brduProb > p:
					dict_calls_brdu[p] += 1
				if eduProb > p:
					dict_calls_edu[p] += 1
				dict_attempts[p] += 1
		#testRead.close()
	return dict_attempts, dict_calls_brdu, dict_calls_edu


#-------------------------------------------------
#
# core embedding model
core_embedding_model = tf.keras.models.load_model('/home/mb915/rds/hpc-work/development/DNAscent_R10align/DNAscent_dev/utils/kmer_core_embedding')
core_embedding_weights = core_embedding_model.get_layer('embedding')
core_embedding_weights._name = 'embedding_core'
core_embedding_weights.trainable = False

residual_embedding_model = tf.keras.models.load_model('/home/mb915/rds/hpc-work/development/DNAscent_R10align/DNAscent_dev/utils/kmer_residual_embedding')
residual_embedding_weights = residual_embedding_model.get_layer('embedding')
residual_embedding_weights._name = 'embedding_residual'
residual_embedding_weights.trainable = False

model = buildModel(3, core_embedding_weights, residual_embedding_weights)
op = Adam(learning_rate=0.0001)
model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy', sample_weight_mode="temporal")
print(model.summary())
model.load_weights(checkpoint_fn)
print('weights loaded')

final_model = model
#pop noisy channel layers
#final_layer = model.layers[-4].output 
#final_model = Model(inputs = model.input, outputs = final_layer, name='final')
#final_model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy', sample_weight_mode="temporal")
#print(final_model.summary())

attempts_inThym, brduCalls_inThym, eduCalls_inThym = checkTestDirectory(dir_Thymtest, final_model)
attempts_inEdU, brduCalls_inEdU, eduCalls_inEdU = checkTestDirectory(dir_EdUtest, final_model)
attempts_inBrdU, brduCalls_inBrdU, eduCalls_inBrdU = checkTestDirectory(dir_BrdUtest, final_model)

#plotting
fig,ax = plt.subplots()

#EdU and thymidine
falsePositives = []
for p in probThresholds:
	falsePositives.append( eduCalls_inThym[p]/float(attempts_inThym[p]) )
truePositives = []
for i, p in enumerate(probThresholds):
	truePositives.append( eduCalls_inEdU[p]/float(attempts_inEdU[p]) )
	ax.annotate(str(p),(falsePositives[i],truePositives[i]),fontsize=6)
plt.plot(falsePositives,truePositives,'b',alpha=0.5,label='EdU/Thym')


#BrdU and thymidine
falsePositives = []
for p in probThresholds:
	falsePositives.append( brduCalls_inThym[p]/float(attempts_inThym[p]) )
truePositives = []
for i, p in enumerate(probThresholds):
	truePositives.append( brduCalls_inBrdU[p]/float(attempts_inBrdU[p]) )
	ax.annotate(str(p),(falsePositives[i],truePositives[i]),fontsize=6)
plt.plot(falsePositives,truePositives,'r',alpha=0.5,label='BrdU/Thym')

#EdU and BrdU
falsePositives = []
for p in probThresholds:
	falsePositives.append( eduCalls_inBrdU[p]/float(attempts_inBrdU[p]) )
truePositives = []
for i, p in enumerate(probThresholds):
	truePositives.append( eduCalls_inEdU[p]/float(attempts_inEdU[p]) )
	ax.annotate(str(p),(falsePositives[i],truePositives[i]),fontsize=6)
plt.plot(falsePositives,truePositives,'b--',alpha=0.5,label='EdU/BrdU')


#BrdU and EdU
falsePositives = []
for p in probThresholds:
	falsePositives.append( brduCalls_inEdU[p]/float(attempts_inEdU[p]) )
truePositives = []
for i, p in enumerate(probThresholds):
	truePositives.append( brduCalls_inBrdU[p]/float(attempts_inBrdU[p]) )
	ax.annotate(str(p),(falsePositives[i],truePositives[i]),fontsize=6)
plt.plot(falsePositives,truePositives,'r--',alpha=0.5,label='BrdU/EdU')

checkpoint_fn_split = checkpoint_fn.split('.')

plt.legend(framealpha=0.5)
plt.xlim(0,1.0)
plt.xlabel('False Positive Rate')
plt.ylabel('(Positive Calls)/Attempts')
plt.ylim(0,1)
plt.savefig("".join(checkpoint_fn_split[0:-1])+'.pdf')
plt.xlim(0,0.2)
plt.savefig("".join(checkpoint_fn_split[0:-1])+'_zoom.pdf')
plt.close()

