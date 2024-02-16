import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, Sequential
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

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from tensorflow_probability.python.layers import Convolution1DReparameterization, DenseReparameterization, OneHotCategorical, KLDivergenceRegularizer

print('Tensorflow version:',tf.__version__)

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

logPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/logs/trainingLog_DNN_70.csv'
trainingReadLogPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/logs/trainingReads_DNN_70.txt'
valReadLogPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/logs/validationReads_DNN_70.txt'
checkpointPath = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/checkpoints/70'
f_checkpoint = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/checkpoints/20_HMM_partII/weights.03-0.23.h5'
validationSplit = 0.1

maxLen = 2000

analogueInc = 0.8


#-------------------------------------------------
#
class DenseVariationalCustom(tfp.layers.DenseVariational):
	def __init__(self,
			units,
			make_posterior_fn,
			make_prior_fn,
			kl_weight=None,
			kl_use_exact=False,
			activation=None,
			use_bias=True,
			activity_regularizer=None,
			**kwargs):
		super(DenseVariationalCustom, self).__init__(
			units=units,
			make_posterior_fn=make_posterior_fn,
			make_prior_fn=make_prior_fn,
			kl_weight=kl_weight,
			kl_use_exact=kl_use_exact,
			activation=activation,
			use_bias=use_bias,
			activity_regularizer=activity_regularizer,
			**kwargs)
		self._kl_weight = kl_weight
		self._kl_use_exact = kl_use_exact

	def get_config(self):
		config ={
		'units': self.units,
		'make_posterior_fn': self._make_posterior_fn,
		'make_prior_fn': self._make_prior_fn,
		'kl_weight': self._kl_weight,
		'kl_use_exact': self._kl_use_exact,
		'activation': self.activation,
		'use_bias': self.use_bias,
       		}
		return config


#-------------------------------------------------
#
def nll(y_true, y_pred):

	return -tf.reduce_mean(tf.reduce_mean(tf.reduce_sum( tf.math.multiply(y_true,tf.math.log(tf.math.add(y_pred,1e-12))), axis=-1), axis=-1))
	#return -y_pred.log_prob(y_true)


#-------------------------------------------------
#
def spike_and_slab(event_shape, dtype):
	distribution = tfd.Mixture(
		cat=tfd.Categorical(probs=[0.5, 0.5]),
		components=[
		tfd.Independent(tfd.Normal(
		loc=tf.zeros(event_shape, dtype=dtype), 
		scale=1.0*tf.ones(event_shape, dtype=dtype)),
		reinterpreted_batch_ndims=1),
		tfd.Independent(tfd.Normal(
		loc=tf.zeros(event_shape, dtype=dtype), 
		scale=10.0*tf.ones(event_shape, dtype=dtype)),
		reinterpreted_batch_ndims=1)],
		name='spike_and_slab'
	)
	return distribution

#-------------------------------------------------
#
def get_posterior(kernel_size, bias_size, dtype=None):
	"""
	This function should create the posterior distribution as specified above.
	The distribution should be created using the kernel_size, bias_size and dtype
	function arguments above.
	The function should then return a callable, that returns the posterior distribution.
	"""
	n = kernel_size + bias_size
	return Sequential([
	tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
	tfpl.IndependentNormal(n)
	])


#-------------------------------------------------
#
def get_prior(kernel_size, bias_size, dtype=None):
	"""
	This function should create the prior distribution, consisting of the 
	"spike and slab" distribution that is described above. 
	The distribution should be created using the kernel_size, bias_size and dtype
	function arguments above.
	The function should then return a callable, that returns the prior distribution.
	"""
	n = kernel_size+bias_size  
	prior_model = Sequential([tfpl.DistributionLambda(lambda t : spike_and_slab(n, dtype))])
	return prior_model


#-------------------------------------------------
#
def get_dense_variational_layer(prior_fn, posterior_fn, kl_weight):
	"""
	This function should create an instance of a DenseVariational layer according 
	to the above specification. 
	The function takes the prior_fn, posterior_fn and kl_weight as arguments, which should 
	be used to define the layer.
	Your function should then return the layer instance.
	"""
	#return DenseVariationalCustom(units=3, make_posterior_fn=posterior_fn, make_prior_fn=prior_fn, kl_weight=kl_weight)
	return Sequential([DenseVariationalCustom(units=3, make_posterior_fn=posterior_fn, make_prior_fn=prior_fn, kl_weight=kl_weight), tfpl.OneHotCategorical(3, convert_to_tensor_fn=tfd.Distribution.mode)])


#-------------------------------------------------
#
def get_convolutional_reparameterization_layer(f, k, divergence_fn):
	"""
	This function should create an instance of a Convolution2DReparameterization 
	layer according to the above specification. 
	The function takes the input_shape and divergence_fn as arguments, which should 
	be used to define the layer.
	Your function should then return the layer instance.
	"""
    
	layer = Convolution1DReparameterization(
		filters=f, kernel_size=k,
		activation='relu', padding='same',
		kernel_prior_fn=tfpl.default_multivariate_normal_fn,
		kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
		kernel_divergence_fn=divergence_fn,
		bias_prior_fn=tfpl.default_multivariate_normal_fn,
		bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
		bias_divergence_fn=divergence_fn
	)
	return layer


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
def convolutional_block(X, divergence_fn, f, filters, stage, block, s=1):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3, F4, F5, F6 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = get_convolutional_reparameterization_layer(F1, f, divergence_fn)(X) 

    # Second component of main path
    X = get_convolutional_reparameterization_layer(F2, f, divergence_fn)(X) 

    # Third component of main path
    X = get_convolutional_reparameterization_layer(F3, f, divergence_fn)(X) 

    # Fourth component of main path
    X = get_convolutional_reparameterization_layer(F4, f, divergence_fn)(X) 

    # Fifth component of main path
    X = get_convolutional_reparameterization_layer(F5, f, divergence_fn)(X) 

    # Sixth component of main path
    X = get_convolutional_reparameterization_layer(F6, f, divergence_fn)(X) 

    ##### SHORTCUT PATH #### 
    X_shortcut = get_convolutional_reparameterization_layer(F6, f, divergence_fn)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a relu activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


#-------------------------------------------------
#
def buildModel(classes = 3, numTrainingReads=100000):

    divergence_fn = lambda q, p, _ : tfd.kl_divergence(q, p) / numTrainingReads
    
    # Define the input as a tensor with shape input_shape
    sequence_input = Input(shape=(None,5))
    
    signal_input = Input(shape=(None,3))

    X = Concatenate(axis=-1)([sequence_input,signal_input])

    # Stage 1
    X = get_convolutional_reparameterization_layer(64, 3, divergence_fn)(X)
    
    # Stage 2
    X = convolutional_block(X, divergence_fn, f=5, filters = [64, 64, 64, 64, 64, 64], stage = 2, block='a', s = 1)
    
    # Stage 3
    X = convolutional_block(X, divergence_fn, f=5, filters=[64, 64, 64, 64, 64, 64], stage=3, block='a', s=2)
    
    # Stage 4
    X = convolutional_block(X, divergence_fn, f=9, filters=[128, 128, 128, 128, 128, 128], stage=4, block='a', s=2)

    # Stage 5
    X = convolutional_block(X, divergence_fn, f=9, filters=[128, 128, 128, 128, 128, 128], stage=5, block='a', s=2)

    # Stage 6
    X = convolutional_block(X, divergence_fn, f=17, filters=[256, 256, 256, 256, 256, 256], stage=6, block='a', s=2)
 
    X = get_convolutional_reparameterization_layer(256, 3, divergence_fn)(X)   
    X = get_convolutional_reparameterization_layer(128, 3, divergence_fn)(X)   
    X = get_convolutional_reparameterization_layer(64, 3, divergence_fn)(X)

    # Output layer
    X = TimeDistributed(get_dense_variational_layer(get_prior, get_posterior, kl_weight=1/numTrainingReads))(X)
    #X = tfpl.OneHotCategorical(3, convert_to_tensor_fn=tfd.Distribution.mode)(X)
    
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
	weights = []
	thymMask = []

	if whichSet == 0.:

		for i, s in enumerate(t.analogueCalls):
			if s == '-': #not thymidine
				label.append([1., 0., 0.])
				weights.append(1.)
				thymMask.append(False)
			else:
				label.append([1., 0., 0.])
				weights.append(1.)
				thymMask.append(True)

	elif whichSet == 1: # BrdU read
		for i, s in enumerate(t.analogueCalls):
			if s == '-': #not thymidine
				label.append([1., 0., 0.])
				weights.append(1.)
				thymMask.append(False)
			else:
				label.append([1.-analogueInc, analogueInc, 0.])
				weights.append(1.)
				thymMask.append(True)

	elif whichSet == 2: # EdU read
		for i, s in enumerate(t.analogueCalls):
			if s == '-': #not thymidine
				label.append([1., 0., 0.])
				weights.append(1.)
				thymMask.append(False)
			else:
				label.append([1.-analogueInc, 0., analogueInc])
				weights.append(1)
				thymMask.append(True)
				
	return np.array(label), np.array(weights), np.array(thymMask)


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
		X, y, W = self.__data_generation(list_IDs_temp)

		return X, y, W

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
		m = np.empty((self.batch_size, maxLen), dtype=bool)		

		# Generate data
		for i, ID in enumerate(list_IDs_temp):

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
			y[i], w[i], m[i] = trainingReadToLabel(trainingRead,whichSet)

		samples = tf.one_hot(tfd.Categorical(probs=y).sample(), depth=3)
		
		y = y.reshape(y.shape[0],y.shape[1],3)
		w = w.reshape(w.shape[0],w.shape[1])
		m = w.reshape(m.shape[0],m.shape[1])
		return [X1, X2], samples, [w, m]


#-------------------------------------------------
#MAIN

#uncomment to train from scratch

filepaths = ['/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode24/trainingReads_slices',
'/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_11_MJ_ONT_Brdu_48hr_training_v14_5khz_fast5/trainingReads_slices',
'/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode22/trainingReads_slices']

maxReads = [48000,
48000,
48000]

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
params = {'dim': [(maxLen,5), (maxLen,3)],
          'batch_size': 32,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['training'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

#-------------------------------------------------
#CNN architecture

#with strategy.scope():

numTrainingReads = len(trainPaths)
model = buildModel(3, numTrainingReads)
op = Adam(learning_rate=0.0001)
model.compile(optimizer=RMSprop(), metrics=['accuracy'], loss=nll, sample_weight_mode="temporal")#, run_eagerly=True)
print(model.summary())

#uncomment to load weights from a trainign checkpoint
#model.load_weights(f_checkpoint)

#callbacks
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
chk = ModelCheckpoint(checkpointPath + '/weights.{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
csv = CSVLogger(logPath, separator=',', append=False)

#generator fit
model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=35, verbose=1, callbacks=[es,chk,csv])

# serialize model to JSON
model_json = model.to_json()
with open(checkpointPath + "/model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(checkpointPath + "/model.h5")
print("Saved model to disk")
