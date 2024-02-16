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
import pickle
import sys

print('Tensorflow version:',tf.__version__)

maxEvents = 20

#f_checkpoint = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/workspace/R10_4_1_DNAtraining/checkpoints/9_pt2/weights.13-0.21.h5'
f_checkpoint = sys.argv[1]

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
    core_sequence_input = Input(shape=(1))
    X_sequence_core = core_embedding_layer(core_sequence_input)
    
    #residual base sequence input
    residual_sequence_input = Input(shape=(1))
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
model.load_weights(f_checkpoint)
model.save('detect_model_BrdUEdU_DNAr10_4_1')

#pop noisy channel layers
'''
final_layer = model.layers[-4].output 
final_model = Model(inputs = model.input, outputs = final_layer, name='final')
final_model.compile(optimizer=op, metrics=['accuracy'], loss='categorical_crossentropy', sample_weight_mode="temporal")
print(final_model.summary())
final_model.save('detect_model_BrdUEdU_DNAr10_4_1')
'''

