import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Dot, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

from Bio import SeqIO

import textwrap


kmer_len = 4
num_ns = 4
vocab_size=4**kmer_len+1
window_size = kmer_len
checkpointPath="/home/mb915/rds/hpc-work/development/DNAscent_R10align/DNAscent_dev/utils/kmer_embedding_checkpoints4"

#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}

index2kmer = {}

#-------------------------------------------------
#
def kmer2index(kmer):

	p = 1;
	r = 0;
	for i in range(kmer_len):

		r += baseToInt[kmer[kmer_len-i-1]] * p;
		p *= 4;
	index2kmer[r+1]=kmer
	return r+1

#-------------------------------------------------
#
sequence = []
f = open('residual_4mers_sorted.txt','r')
for line in f:
	splitLine = line.rstrip().split()
	sequence.append( kmer2index(splitLine[0]) )
f.close()

# Generate skip-gram pairs
print('Making skipgrams...')
skip_grams = skipgrams(sequence, vocabulary_size=vocab_size, window_size=window_size)
pairs = skip_grams[0]
labels = skip_grams[1]
print('Done.')

# Define the model
embed_size = 16
input_target = Input((1,))
input_context = Input((1,))
embedding = Embedding(vocab_size, embed_size)
target = embedding(input_target)
context = embedding(input_context)
dot_product = Dot(axes=2)([target, context])
dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
output = Dense(1, activation='sigmoid')(dot_product)
model = Model(inputs=[input_target, input_context], outputs=output)
op = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=op)
print(model.summary())

# Prepare data for training
targets = np.array([pair[0] for pair in pairs], dtype=np.int32)
contexts = np.array([pair[1] for pair in pairs], dtype=np.int32)
labels = np.array(labels, dtype=np.int32)

chk = ModelCheckpoint(checkpointPath + '/weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
model.fit([targets, contexts], labels, validation_split=0.1, epochs=1000, batch_size=16, callbacks=[chk,es])

model.save('kmer_residual_embedding')

# Get the word embeddings
word_embeddings = model.get_layer('embedding').get_weights()[0]

# Test the embeddings
test_word = 'TAAG'
test_word_index = kmer2index(test_word)
similarity_scores = np.dot(word_embeddings, word_embeddings[test_word_index])
sorted_indexes = np.argsort(similarity_scores)[::-1][1:6]

# Print most similar words
print(f'Most similar words to "{test_word}":')
for index in sorted_indexes:
    word = index2kmer[index]
    print(f'{word}: {similarity_scores[index]}')
