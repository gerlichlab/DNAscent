import sys
import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
import random

barcode = int(sys.argv[1])

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/Thym_trainingData/trainingFiles_LSTMevents'

bc08dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2019_11_29_FT_ONT_Plasmodium_Barcoded/barcode01/DNAscentv2.trainingData'
inputFiles = [('thymidine', bc08dnascent, folderPath)]
maxLen = 4000
maxRaw = 10
maxReads = int(sys.argv[2])

llThreshold = 1.25


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sixMers, read_eventMeans, read_eventLength, read_modelMeans, read_modelStd, read_positions, readID, label):
		self.sixMers = read_sixMers[0:maxLen]
		self.eventMean = read_eventMeans[0:maxLen]
		self.eventLength = read_eventLength[0:maxLen]
		self.modelMeans = read_modelMeans[0:maxLen]
		self.modelStd = read_modelStd[0:maxLen]
		self.readID = readID
		self.label = label

		allPositions = []
		for i, s in enumerate(self.sixMers):

			oneSet = []
			for j in range(len(self.eventMean[i])):

				#base
				oneHot = [0]*4
				index = baseToInt[s[0]]
				oneHot[index] = 1

				#other features
				oneHot.append(self.eventMean[i][j])
				oneHot.append(self.eventLength[i][j])
				oneHot.append(self.modelMeans[i])
				oneHot.append(self.modelStd[i])
				oneSet.append(oneHot)

			if len(oneSet) < maxRaw:
				for b in range(maxRaw - len(oneSet)):
					oneSet.append([0.]*8)

			allPositions.append(np.array(oneSet[0:maxRaw]))

		self.trainingTensor = np.array((allPositions))


#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}


#-------------------------------------------------
#reshape into tensors
def saveRead(trainingRead, readID, folderPath):

	f = open(folderPath + '/' + readID + '.p', 'wb')
	pickle.dump(trainingRead, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()


#-------------------------------------------------
#pull from training data file
def importFromFile(label, fname, folderPath):

	#count the number of reads in order to randomize across the genome
	'''
	totalReads = 0
	readsToUse = []	
	f = open(fname,'r')
	for line in f:
		if line[0] == '>':
			totalReads += 1
	f.close()
	readsToUse = random.sample(range(0,totalReads), maxReads)
	'''

	print('Parsing training data file...')

	read_sixMers = []
	read_eventMeans = []
	read_stutter = []
	read_lengthMeans = []
	read_lengthStd = []
	read_modelMeans = []
	read_modelStd = []
	read_positions = []
	read_BrdUcalls = []

	pos_eventMeans = []
	pos_lengths = []

	prevReadID = ''
	prevPos = -1
	
	readsLoaded = 0
	switch = False
	f = open(fname,'r')
	for line in f:

		if line[0] == '#':
			continue

		if line[0] == '>':

			readsLoaded += 1
			#print('Reads loaded: ',readsLoaded)
			if readsLoaded > maxReads:
				break

			#make an object out of this read
			if len(read_sixMers) > maxLen:

				if switch:
					tr = trainingRead(read_sixMers, read_eventMeans, read_eventLength, read_modelMeans, read_modelStd, read_positions, prevReadID, label)
					saveRead(tr, prevReadID, folderPath)

			#reset for read
			read_sixMers = []
			read_eventMeans = []
			read_eventLength = []
			read_modelMeans = []
			read_modelStd = []
			read_positions = []


			#reset for position
			pos_eventMeans = []
			pos_lengths = []
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
			if splitLine[4] == "NNNNNN":
				continue

			pos = int(splitLine[0])

			#don't get too close to the ends of the read, because we can't get an HMM detect there
			if abs(pos-mappingStart) < 26 or abs(pos-mappingEnd) < 26:
				continue

			#we've changed position
			if pos != prevPos and prevPos != -1:

				read_sixMers.append(sixMer)
				read_eventMeans.append(pos_eventMeans)
				read_eventLength.append(pos_lengths)
				read_modelMeans.append(modelMean)
				read_modelStd.append(modelStd)
				read_positions.append(prevPos)

				#reset for position
				pos_eventMeans = []
				pos_lengths = []
				prevPos = pos

			sixMer = splitLine[4]
			modelMean = float(splitLine[5])
			modelStd = float(splitLine[6])
			eventMean = float(splitLine[2])
			eventLength = float(splitLine[3])
			prevReadID = readID
			prevPos = pos
				
			pos_eventMeans.append(eventMean)
			pos_lengths.append(eventLength)

	f.close()
	print('Done.')


#-------------------------------------------------
#MAIN
tup = inputFiles[barcode]
importFromFile(tup[0], tup[1], tup[2])
