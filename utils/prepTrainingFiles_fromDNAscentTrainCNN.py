import sys
import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
import random
import math

outputPath = sys.argv[1]    # directory for pickled training reads
inputTrainCNN = sys.argv[2] # from DNAscent trainCNN
label = sys.argv[3]         # e.g., thymidine, BrdU, or EdU
maxReads = int(sys.argv[4])

maxLen = 2000
maxRaw = 20


#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}


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
#pickle reads on highest protocol
def saveRead(trainingRead, readID, folderPath):

	f = open(folderPath + '/' + readID + '.p', 'wb')
	pickle.dump(trainingRead, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()



#-------------------------------------------------
#driver
print('Parsing training data file...')

read_kmers = []
read_signals = []
read_modelSignal = []
read_positions = []
read_analogueCalls = []

pos_signals = []

prevReadID = ''
prevPos = -1

readsLoaded = 0
switch = False
f = open(inputTrainCNN,'r')
for line in f:

	if line[0] == '#':
		continue

	if line[0] == '>':

		#make an object out of this read
		if len(read_kmers) > maxLen:

			if switch:
			
				numReadSlices = math.floor(float(len(read_kmers))/float(maxLen))
				for s in range(numReadSlices):
				
					readsLoaded += 1
					if readsLoaded > maxReads:
						break
				
					tr = trainingRead(read_kmers[maxLen*s:maxLen*(s+1)], read_signals[maxLen*s:maxLen*(s+1)], read_modelSignal[maxLen*s:maxLen*(s+1)], read_positions[maxLen*s:maxLen*(s+1)], read_analogueCalls[maxLen*s:maxLen*(s+1)], prevReadID, label)
					saveRead(tr, prevReadID+'_slice'+str(s), outputPath)

		#reset for read
		read_kmers = []
		read_signals = []
		read_modelSignal = []
		read_positions = []
		read_analogueCalls = []


		#reset for position
		pos_eventMeans = []
		prevPos = -1

		splitLine = line.rstrip().split()
		readID = splitLine[0][1:]
		chromosome = splitLine[1]
		mappingStart = int(splitLine[2])
		mappingEnd = int(splitLine[3])
		prevPos = -1
		
		switch = True

	elif switch:
		splitLine = line.rstrip().split('\t')

		#skip insertion events
		if splitLine[3] == "NNNNNNNNN":
			continue

		#position on reference genome
		pos = int(splitLine[0])

		#don't get too close to the ends of the read, because we can't get an HMM detect there
		if abs(pos-mappingStart) < 30 or abs(pos-mappingEnd) < 30:
			continue

		#on a position change, log all signals, the kmer, the model value, and the analogue call
		if pos != prevPos and prevPos != -1:

			read_kmers.append(kmer)
			read_signals.append(pos_signals)
			read_modelSignal.append(modelSignal)
			read_positions.append(prevPos)
			read_analogueCalls.append(analogueCall)

			#reset for position
			pos_signals = []
			prevPos = pos

		kmer = splitLine[3]
		modelSignal = float(splitLine[4])
		signal = float(splitLine[2])
		prevReadID = readID
		prevPos = pos

		analogueCall = '-'
		if len(splitLine) >= 6:
			analogueCall = splitLine[5]
			
		pos_signals.append(signal)

f.close()
print('Done.')

