import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
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
tfd = tfp.distributions
tfpl = tfp.layers
from tensorflow_probability.python.layers import Convolution1DReparameterization, DenseReparameterization, OneHotCategorical, KLDivergenceRegularizer

maxReads = 200

dir_BrdUtest = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_11_MJ_ONT_Brdu_48hr_training_v14_5khz_fast5/testReads_slices'
dir_EdUtest = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode24/testReads_slices'
dir_Thymtest = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode22/testReads_slices'

probThresholds = np.array(range(0,21))/20.

checkpoint_fn = sys.argv[1]

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
	return DenseVariationalCustom(units=3, make_posterior_fn=posterior_fn, make_prior_fn=prior_fn, kl_weight=kl_weight)


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


def buildModel(classes = 3, numTrainingReads=100000):

    divergence_fn = lambda q, p, _ : tfd.kl_divergence(q, p) / float(numTrainingReads)
    
    # Define the input as a tensor with shape input_shape
    sequence_input = Input(shape=(None,5))
    
    signal_input = Input(shape=(None,3))

    X = Concatenate(axis=-1)([sequence_input,signal_input])

    # Stage 1
    X = get_convolutional_reparameterization_layer(64, 3, divergence_fn)(X)
    X = get_convolutional_reparameterization_layer(64, 5, divergence_fn)(X)
    X = get_convolutional_reparameterization_layer(128, 9, divergence_fn)(X)
    X = get_convolutional_reparameterization_layer(128, 13, divergence_fn)(X)
    X = get_convolutional_reparameterization_layer(256, 17, divergence_fn)(X)    
    # Stage 2
    #X = convolutional_block(X, divergence_fn, f=13, filters = [64, 64, 64, 64, 64, 64], stage = 2, block='a', s = 1)
    
    # Stage 3
    #X = convolutional_block(X, divergence_fn, f=9, filters=[64, 64, 64, 64, 64, 64], stage=3, block='a', s=2)
    
    # Stage 4
    X = convolutional_block(X, divergence_fn, f=17, filters=[128, 128, 128, 128, 128, 128], stage=4, block='a', s=2)

    # Stage 5
    #X = convolutional_block(X, divergence_fn, f=17, filters=[128, 128, 128, 128, 128, 128], stage=5, block='a', s=2)

    # Stage 6
    #X = convolutional_block(X, divergence_fn, f=17, filters=[256, 256, 256, 256, 256, 256], stage=6, block='a', s=2)
 
    #X = get_convolutional_reparameterization_layer(256, 3, divergence_fn)(X)   
    #X = get_convolutional_reparameterization_layer(128, 3, divergence_fn)(X)   
    X = get_convolutional_reparameterization_layer(64, 3, divergence_fn)(X)

    # Output layer
    X = get_dense_variational_layer(get_prior, get_posterior, kl_weight=1./float(numTrainingReads))(X)
    X = tfpl.OneHotCategorical(3, convert_to_tensor_fn=tfd.Distribution.mode)(X)
    
    # Create model
    model = Model(inputs = [sequence_input, signal_input] , outputs = X, name='R10_BrdU_EdU')

    return model


#-------------------------------------------------
#
def testReadToTensor(t):

	return t.sequence_tensor, t.signal_tensor


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
		sequence_tensor, signal_tensor = testReadToTensor(testRead)
		sequence_tensor = sequence_tensor.reshape(1,sequence_tensor.shape[0],sequence_tensor.shape[1])
		signal_tensor = signal_tensor.reshape(1,signal_tensor.shape[0],signal_tensor.shape[1],1)		

		pred = model([sequence_tensor,signal_tensor])
		print(pred)
		print(pred.shape)
		print(pred.mean().numpy())
		pred = pred.reshape(sequence_tensor.shape[1], 3)
		for i in range(pred.shape[0]):
		
			#skip non-thymidine positions
			if testRead.analogueCalls[i] == '-':
				continue

			sm = tf.nn.softmax(pred[i])
			thymProb = sm[0]
			brduProb = sm[1]
			eduProb = sm[2]
			print(sm)


			'''
			thymProb = pred[i,0]
			brduProb = pred[i,1]
			eduProb = pred[i,2]
			'''

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
model = buildModel(classes=3)
op = Adam(learning_rate=0.0001)
model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy', sample_weight_mode="temporal", run_eagerly=True)
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

