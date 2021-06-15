import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json, Model
from tensorflow.python.keras.layers import Dense, Dropout, ZeroPadding1D, Layer
from tensorflow.python.keras.layers import Embedding, Flatten, MaxPooling1D,AveragePooling1D, Input
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Activation, LSTM, SeparableConv1D, Add
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional, Reshape, BatchNormalization, Reshape, Lambda
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.python.keras import Input
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras import backend
from tensorflow.python.keras.backend import ctc_batch_cost, ctc_decode, get_value, cast


import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
import sys
import os
import pickle
from scipy.stats import halfnorm

tf.keras.backend.set_learning_phase(1)  # set inference phase

f_checkpoint = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/modelTraining/checkpoints_CTC_8/weights.17.h5'

maxLen = 2001
maxReads = 100000 #250#

#static params
truePositive = 0.5
trueNegative = 0.9
falsePositive = 1. - trueNegative
llThreshold = 1.25

#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sequence, read_raw, read_modelMeans, read_modelStdvs, seqIdx2LL, readID, analogueConc, ll):
		
		self.sequence = read_sequence
		self.raw = read_raw
		self.modelMeans = read_modelMeans
		self.modelStdvs = read_modelStdvs
		self.readID = readID
		self.analogueConc = analogueConc
		self.logLikelihood = seqIdx2LL
		self.labelLength = ll

#-------------------------------------------------
# from https://keras.io/examples/vision/captcha_ocr/
class CTCLayer(Layer):
	def __init__(self, name=None):
		super().__init__(name=name)
		self.loss_fn = ctc_batch_cost

	def call(self, input_tuple):
		y_true = input_tuple[0]
		y_pred = input_tuple[1]
		# Compute the training-time loss value and add it
		# to the layer using `self.add_loss()`.
		batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
		input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
		label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

		input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
		label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

		loss = self.loss_fn(y_true, y_pred, input_length, label_length)
		self.add_loss(loss)

		# At test time, just return the computed predictions
		return y_pred


#-------------------------------------------------
#
def identity_block(X, f, filters, stage, block):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = SeparableConv1D(filters = F1, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('tanh')(X)
    
    # Second component of main path
    X = SeparableConv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)
    X = Activation('tanh')(X)

    # Third component of main path 
    X = SeparableConv1D(filters = F3, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('tanh')(X)
    
    return X


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

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('tanh')(X)

    return X


#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}


#-------------------------------------------------
#
def trainingReadToTensor(t):

	wholeRead = []
	for i, s in enumerate(t.sixMers[0:maxLen]):

		for j in range(len(t.eventMean[i])):

			#base
			oneHot = [0]*4
			index = baseToInt[s[0]]
			oneHot[index] = 1

			#other features
			oneHot.append(t.eventMean[i][j])
			oneHot.append(t.modelMeans[i])

			wholeRead.append(np.array(oneHot))

	return np.array((wholeRead))


#-------------------------------------------------
#
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    #y_pred = y_pred[:, 2:, :]
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


#-------------------------------------------------
#
def buildModel():

	input_shape = (None,6)
	classes = 4 #no analogue, BrdU, EdU, plus blank

	# Define the input as a tensor with shape input_shape
	X_input = Input(name='signal_input',shape=input_shape)
	
	# Stage 1
	X = Conv1D(64, 3, strides = 1, padding='same', name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
	X = BatchNormalization(name = 'bn_conv1')(X)
	X = Activation('tanh')(X)

	# Stage 2
	X = convolutional_block(X, f = 3, filters = [64, 64, 64, 64, 64, 64], stage = 2, block='a', s = 1)

	# Stage 3
	X = convolutional_block(X, f=3, filters=[64, 64, 64, 64, 64, 64], stage=3, block='a', s=2)

	# Stage 4
	X = convolutional_block(X, f=9, filters=[128, 128, 128, 128, 128, 128], stage=4, block='a', s=2)

	# Stage 5
	X = convolutional_block(X, f=9, filters=[128, 128, 128, 128, 128, 128], stage=5, block='a', s=2)

	# Stage 6
	X = convolutional_block(X, f=15, filters=[256, 256, 256, 256, 256, 256], stage=6, block='a', s=2)

	X = Conv1D(64, 3, strides = 1, padding='same', name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(name = 'bn_conv2')(X)
	X = Activation('tanh')(X)

	X = Conv1D(64, 3, strides = 1, padding='same', name = 'conv3', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(name = 'bn_conv3')(X)
	X = Activation('tanh')(X)

	X = Conv1D(64, 3, strides = 1, padding='same', name = 'conv4', kernel_initializer = glorot_uniform(seed=0))(X)

	# Output layer
	X = TimeDistributed(Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0)))(X)

	# Add CTC layer for calculating CTC loss at each step
	labels = Input(name="the_labels", shape=(None,), dtype="float32")
	output = CTCLayer(name="ctc_loss")((labels, X))

	# Create model
	model = Model(inputs=[X_input, labels], outputs=output, name="BrdU_model")
	op = Adam()
	model.compile(optimizer=op)
	return model


#-------------------------------------------------
#MAIN



directory = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/Thym_trainingData/trainingFiles_raw'

'''
filepaths = ['/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdU_trainingData/EdU_augmentedTrainingData_slices_CNN_raw_gap20',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdU_trainingData/EdU_augmentedTrainingData_slices_CNN_raw_gap30',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdU_trainingData/EdU_augmentedTrainingData_slices_CNN_raw_gap40',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/BrdU_trainingData/BrdU_augmentedTrainingData_slices_CNN_raw_gap20',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/BrdU_trainingData/BrdU_augmentedTrainingData_slices_CNN_raw_gap30',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/BrdU_trainingData/BrdU_augmentedTrainingData_slices_CNN_raw_gap40',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdUinBrdU_trainingData_raw/gap20',
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdUinBrdU_trainingData_raw/gap30', 
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdUinBrdU_trainingData_raw/gap40',  
'/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/Thym_trainingData/trainingFiles_raw']
'''
maxReads = 5

allPaths = []
for fcount, fname in enumerate(os.listdir(directory)):
	allPaths.append(directory + '/' + fname)
	if fcount > maxReads:
		break

allPaths = allPaths[-5:]

model = buildModel()

print(model.summary())

#uncomment to load weights from a trainign checkpoint
model.load_weights(f_checkpoint)

prediction_model = Model(
    model.get_layer(name="signal_input").input, model.get_layer(name="time_distributed").output
)


#for i, ID in enumerate(val_readIDs):
f = open(allPaths[1], "rb")
trainingRead = pickle.load(f)
f.close()
tensor = trainingReadToTensor(trainingRead)
tensor = tensor.reshape(tensor.shape[0],1,6)
print(prediction_model.predict(tensor)[0:200])
print(tensor[0:200])
print(trainingRead.sixMers[0:200])


print(len(trainingRead.modelMeans))
output = prediction_model.predict(tensor)
inputshape = np.array([tensor.shape[0]])
output = output.reshape(1,output.shape[0],output.shape[2])
result = ctc_decode(output, input_length=inputshape, greedy=False)
sess = tf.Session()
with sess.as_default():
	print(result[0][0].eval())
