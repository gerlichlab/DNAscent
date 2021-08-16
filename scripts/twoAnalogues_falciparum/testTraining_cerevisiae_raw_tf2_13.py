import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, model_from_json, Model
from tensorflow.python.keras.layers import Dense, Dropout, ZeroPadding1D
from tensorflow.python.keras.layers import Embedding, Flatten, MaxPooling1D,AveragePooling1D, Input
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Activation, LSTM, SeparableConv1D, Add, SeparableConv2D, GRU
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional, Reshape, BatchNormalization, Masking
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.python.keras import Input
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
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
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

#if you don't do this in the beginning, you'll get an error on batch normalisation
tf.keras.backend.set_learning_phase(0)  # set inference phase

checkpointDirectory = os.path.dirname(sys.argv[1])
DNAscentExecutable = '/home/mb915/rds/hpc-work/development/twoAnalogues/DNAscent_raw'+str(sys.argv[2])
referenceGenome = '/home/mb915/rds/rds-mb915-notbackedup/genomes/SacCer3.fasta'
#testBamBrdU = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode10/EdUTrainingTest.bam'
#testBamThym = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/EdUTrainingTest.bam'
#BrdUIndex = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/index.dnascent'
#ThymIndex = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/index.dnascent'

testBamBrdU = '/home/mb915/rds/rds-mb915-notbackedup/data/hyrienLab/BTF_BB_ONT_1_FAH58543_A_fast5/EdUTrainingTest.bam'
testBamThym = '/home/mb915/rds/rds-mb915-notbackedup/data/hyrienLab/BTF_AS_ONT_1_FAH58492_A_fast5/EdUTrainingTest.bam'
BrdUIndex = '/home/mb915/rds/rds-mb915-notbackedup/data/hyrienLab/BTF_BB_ONT_1_FAH58543_A_fast5/index.dnascent'
ThymIndex = '/home/mb915/rds/rds-mb915-notbackedup/data/hyrienLab/BTF_AS_ONT_1_FAH58492_A_fast5/index.dnascent'

analogueSets = [('Thymidine',testBamThym,ThymIndex),('BrdU',testBamBrdU,BrdUIndex)]
threads = 112 #number of threads to use for DNAscent detect


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
#
def buildModel(input_shape = (64, 64, 3), classes = 6):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = TimeDistributed(Masking(mask_value=0.0))(X_input)
    X = TimeDistributed(Bidirectional(GRU(16,return_sequences=True)))(X)
    X_shortcut = X
    X = TimeDistributed(Bidirectional(GRU(16,return_sequences=True)))(X)
    X = TimeDistributed(Bidirectional(GRU(16,return_sequences=True)))(X)
    X = Add()([X, X_shortcut])
    X = Activation('tanh')(X)
    X = TimeDistributed(Bidirectional(GRU(16,return_sequences=False)))(X)

    # Stage 1
    X = Conv1D(64, 4, strides = 1, padding='same', name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
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

	model = buildModel((None,15,6), 3)
	op = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy', sample_weight_mode="temporal")
	print(model.summary())

	model.load_weights(checkpointFilename)

	#use this to find output names
	model.save(DNAscentExecutable+'/dnn_models/test')
	#then: saved_model_cli show --dir /home/mb915/rds/hpc-work/development/DNAscent_raw/dnn_models/test/ --tag_set serve --signature_def serving_default

	'''
	output_layer_name = model.output.name.split(':')[0]
	print(output_layer_name)

	input_layer_name = model.input.name.split(':')[0]
	print(input_layer_name)



	full_model = tf.function(lambda inputs: model(inputs))    
	full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])

	# Get frozen ConcreteFunction    
	frozen_func = convert_variables_to_constants_v2(full_model)
	frozen_func.graph.as_graph_def()

	tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=DNAscentExecutable+'/dnn_models',name="BrdUEdU_detect.pb", as_text=False)
	'''


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
	#if not os.path.isfile(detectOutputFilename):
	print(DNAscentExecutable +'/bin/DNAscent' + ' detect -r ' + referenceGenome + ' -i ' + aSet[2] + ' -b ' + aSet[1] + ' -o ' + detectOutputFilename + ' -t ' + str(threads))
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

	if line[0] == '#' or line[0] == '%' or line[0] == '\n':
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

	if line[0] == '#' or line[0] == '%' or line[0] == '\n':
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
plt.savefig(plotOutputDir + '/' + os.path.splitext(fname)[0] + 'raw_cerevisiae_FAH58543.pdf')
plt.xlim(0,0.2)
plt.savefig(plotOutputDir + '/' + os.path.splitext(fname)[0] + 'raw_zoom_cerevisiae_FAH58543.pdf')
plt.close()
