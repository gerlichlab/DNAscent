import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json, Model
from tensorflow.python.keras.layers import Dense, Dropout, ZeroPadding1D
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
import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
from scipy.stats import halfnorm
from tensorflow.tools.graph_transforms import TransformGraph

#if you don't do this in the beginning, you'll get an error on batch normalisation
tf.keras.backend.set_learning_phase(0)  # set inference phase

checkpointDirectory = os.path.dirname(sys.argv[1])
DNAscentExecutable = '/home/mb915/rds/hpc-work/development/twoAnalogues/DNAscent_dev'+str(sys.argv[2])
referenceGenome = '/home/mb915/rds/rds-mb915-notbackedup/genomes/SacCer3.fasta'
testBamBrdU = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode11/EdUTrainingTest.bam'
testBamThym = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/EdUTrainingTest.bam'
BrdUIndex = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/index.dnascent'
ThymIndex = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/index.dnascent'
analogueSets = [('Thymidine',testBamThym,ThymIndex),('BrdU',testBamBrdU,BrdUIndex)]
threads = 56 #number of threads to use for DNAscent detect


##############################################################################################################
def my_freeze_graph(input_node_names,output_node_names, destination, name="frozen_model.pb"):
	"""
	Freeze the current graph alongside its weights (converted to constants) into a protobuf file.
	:param output_node_names: The name of the output node names we are interested in
	:param destination: Destination folder or remote service (eg. gs://)
	:param name: Filename of the saved graph
	:return:
	"""

	sess = tf.keras.backend.get_session()
	input_graph_def = sess.graph.as_graph_def()     # get graph def proto from keras session's graph

	with sess.as_default():

		#transforms = ["quantize_weights", "quantize_nodes"]
		#transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], output_node_names, transforms)

		# Convert variables into constants so they will be stored into the graph def
		output_graph_def = tf.graph_util.convert_variables_to_constants(sess,input_graph_def,output_node_names=output_node_names)

		tf.train.write_graph(graph_or_graph_def=output_graph_def, logdir=destination, name=name, as_text=False)

	tf.keras.backend.clear_session()


##############################################################################################################
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


##############################################################################################################
def buildModel(input_shape = (64, 64, 3), classes = 6):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Stage 1
    X = Conv1D(64, 4, strides = 1, padding='same', name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(name = 'bn_conv1')(X)
    X = Activation('tanh')(X)

    # Stage 2
    X = convolutional_block(X, f = 4, filters = [64, 64, 64, 64, 64, 64], stage = 2, block='a', s = 1)

    # Stage 3
    X = convolutional_block(X, f=4, filters=[64, 64, 64, 64, 64, 64], stage=3, block='a', s=2)

    # Stage 4
    X = convolutional_block(X, f=8, filters=[128, 128, 128, 128, 128, 128], stage=4, block='a', s=2)

    # Stage 5
    X = convolutional_block(X, f=8, filters=[128, 128, 128, 128, 128, 128], stage=5, block='a', s=2)

    # Stage 6
    X = convolutional_block(X, f=16, filters=[256, 256, 256, 256, 256, 256], stage=6, block='a', s=2)

    X = Conv1D(64, 4, strides = 1, padding='same', name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv2')(X)
    X = Activation('tanh')(X)

    X = Conv1D(64, 4, strides = 1, padding='same', name = 'conv3', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv3')(X)
    X = Activation('tanh')(X)

    X = Conv1D(64, 4, strides = 1, padding='same', name = 'conv4', kernel_initializer = glorot_uniform(seed=0))(X)

    # Output layer
    X = TimeDistributed(Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0)))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='BrdUDetect')

    return model


##############################################################################################################
def modelFromCheckpoint(checkpointFilename):

	model = buildModel((None,8), 3)
	op = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy', sample_weight_mode="temporal")
	print(model.summary())

	model.load_weights(checkpointFilename)

	output_layer_name = model.output.name.split(':')[0]
	print(output_layer_name)

	input_layer_name = model.input.name.split(':')[0]
	print(input_layer_name)

	my_freeze_graph([input_layer_name],[output_layer_name],destination=DNAscentExecutable+'/dnn_models', name="BrdUEdU_detect.pb")


##############################################################################################################
#MAIN

#for fcount, fname in enumerate(os.listdir(checkpointDirectory)): #for each checkpoint

fname = os.path.basename(sys.argv[1])

print('Checkpoint: ',fname)

checkpoint_fullPath = checkpointDirectory + '/' + fname

#turn the checkpoint into a model in the DNAscent directory
modelFromCheckpoint(checkpoint_fullPath)
print('   Model built...')

detectOutputDir =  checkpointDirectory + '/DNAscentDetect'

os.system('mkdir ' + detectOutputDir)

base2DetectFiles = {'Thymidine':'','BrdU':'','EdU':''}

for aSet in analogueSets:

	detectOutputFilename = detectOutputDir + '/' + aSet[0] + '_' + os.path.splitext(fname)[0] + '_cerevisiae.detect'
	base2DetectFiles[aSet[0]] = detectOutputFilename

	#use the model we built to run DNAscent detect
	os.system(DNAscentExecutable +'/bin/DNAscent' + ' detect -r ' + referenceGenome + ' -i ' + aSet[2] + ' -b ' + aSet[1] + ' -o ' + detectOutputFilename + ' -t ' + str(threads))

print('   Plotting...')
#plot the ROC curves
maxReads = 1000
probTests = np.array(range(1,10))/10.

thym_calls = np.zeros(len(probTests))
thym_attempts = np.zeros(len(probTests))
brdu_attempts = np.zeros(len(probTests))
edu_attempts = np.zeros(len(probTests))

BrdU_in_BrdU_calls = np.zeros(len(probTests))
BrdU_in_Thym_calls = np.zeros(len(probTests))
BrdU_in_EdU_calls = np.zeros(len(probTests))
EdU_in_BrdU_calls = np.zeros(len(probTests))
EdU_in_Thym_calls = np.zeros(len(probTests))
EdU_in_EdU_calls = np.zeros(len(probTests))

#thymidine
f = open(base2DetectFiles['Thymidine'],'r')
readCtr = 0
for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':
		splitLine = line.rstrip().split()

		readCtr += 1
		if readCtr > maxReads:
			break

		continue
	else:
		splitLine = line.split('\t')
		position = int(splitLine[0])

		BrdUprob = float(splitLine[2])
		EdUprob = float(splitLine[1])

		for j in range(0,len(probTests)):
			if BrdUprob >= probTests[j]:
				BrdU_in_Thym_calls[j] += 1.
			if EdUprob >= probTests[j]:
				EdU_in_Thym_calls[j] += 1.
			thym_attempts[j] += 1.
f.close()

#brdu
f = open(base2DetectFiles['BrdU'],'r')
readCtr = 0
for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':
		splitLine = line.rstrip().split()

		readCtr += 1
		if readCtr > maxReads:
			break

		continue
	else:
		splitLine = line.split('\t')
		position = int(splitLine[0])

		BrdUprob = float(splitLine[2])
		EdUprob = float(splitLine[1])

		for j in range(0,len(probTests)):
			if BrdUprob >= probTests[j]:
				BrdU_in_BrdU_calls[j] += 1.
			if EdUprob >= probTests[j]:
				EdU_in_BrdU_calls[j] += 1.
			brdu_attempts[j] += 1.
f.close()

plotOutputDir =  checkpointDirectory + '/ROC_curves'
os.system('mkdir ' + plotOutputDir)

fig, ax = plt.subplots()

#brdu
x1 = BrdU_in_Thym_calls/thym_attempts
x2 = EdU_in_Thym_calls/thym_attempts
x3 = EdU_in_BrdU_calls/brdu_attempts
y = BrdU_in_BrdU_calls/brdu_attempts

plt.plot(x1[::-1], y[::-1],'r-', label='BrdU Calls in Thym',alpha=0.5)
for p,txt in enumerate(probTests):
	ax.annotate(str(txt),(x1[p],y[p]),fontsize=6)

plt.plot(x2[::-1], y[::-1],'k--', label='EdU Calls in Thym',alpha=0.5)
for p,txt in enumerate(probTests):
	ax.annotate(str(txt),(x2[p],y[p]),fontsize=6)

plt.plot(x3[::-1], y[::-1],'b--', label='EdU Calls in BrdU',alpha=0.5)
for p,txt in enumerate(probTests):
	ax.annotate(str(txt),(x3[p],y[p]),fontsize=6)

plt.legend(framealpha=0.5)
plt.xlim(0,1.0)
plt.xlabel('False Positive Rate')
plt.ylabel('Calls/Attempts')
plt.ylim(0,1)
plt.savefig(plotOutputDir + '/' + os.path.splitext(fname)[0] + '_cerevisiae.pdf')
plt.xlim(0,0.2)
plt.savefig(plotOutputDir + '/' + os.path.splitext(fname)[0] + '_zoom_cerevisiae.pdf')
plt.close()
