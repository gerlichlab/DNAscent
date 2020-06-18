import sys
import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
from scipy.stats import halfnorm

barcode = int(sys.argv[1])

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data8'

bc08eventalign = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/eventalign_readNames.out'
bc10eventalign = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode10/eventalign_readNames.out'
bc11eventalign = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode11/eventalign_readNames.out'
bc12eventalign = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode12/eventalign_readNames.out'

bc08dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/commit6a663ee_l_6000_q_20.detect'
bc10dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode10/commit6a663ee_l_6000_q_20.detect'
bc11dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode11/commit6a663ee_l_6000_q_20.detect'
bc12dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode12/commit6a663ee_l_6000_q_20.detect'

inputFiles = [(0., bc08eventalign, bc08dnascent),(0.26, bc10eventalign, bc10dnascent),(0.5, bc11eventalign, bc11dnascent),(0.8, bc12eventalign, bc12dnascent)]
maxLen = 3000

llThreshold = 1.25

#-------------------------------------------------
#
def parseDNAscentDetect(fname):
	readid2calls = {}
	problemIDs = []
	f = open(fname,'r')	
	readsLoaded = 0
	for line in f:
		if line[0] == '>':
			splitLine = line.rstrip().split()
			readID = splitLine[0][1:]
			if readID in readid2calls:
				problemIDs.append(readID)
			readsLoaded += 1
			if readsLoaded % 1000 == 0:
				print('DNAscent detect reads loaded: ',readsLoaded)
				
		else:
			splitLine = line.rstrip().split('\t')
			if float(splitLine[1]) > llThreshold:
				if readID in readid2calls:
					readid2calls[readID].append( (int(splitLine[0]), splitLine[1]) )
				else:
					readid2calls[readID] = [(int(splitLine[0]), splitLine[1])]

	for i in problemIDs:
		del readid2calls[i]

	return readid2calls


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sixMers, read_eventMeans, read_eventStd, read_stutter, read_lengthMeans, read_lengthStd, read_modelMeans, read_modelStd, read_positions, readID, analogueConc, readID2calls):
		self.sixMers = read_sixMers[0:maxLen]
		self.eventMean = read_eventMeans[0:maxLen]
		self.eventStd = read_eventStd[0:maxLen]
		self.stutter = read_stutter[0:maxLen]
		self.lengthMean = read_lengthMeans[0:maxLen]
		self.lengthStd = read_lengthStd[0:maxLen]
		self.modelMeans = read_modelMeans[0:maxLen]
		self.modelStd = read_modelStd[0:maxLen]
		self.readID = readID
		self.analogueConc = analogueConc

		#get log likelihoods from DNAscent detect
		positiveCallsOnRef = readID2calls[readID]
		positiveCallsOnRef = dict(positiveCallsOnRef)
		logLikelihood = []
		for p in read_positions[0:maxLen]:
			if p in positiveCallsOnRef:
				logLikelihood.append(positiveCallsOnRef[p])
			else:
				logLikelihood.append('-')
		self.logLikelihood = logLikelihood

		gaps = [1]
		for i in range(1, maxLen):
			gaps.append(read_positions[i] - read_positions[i-1])
		self.gaps = gaps

		'''
		print(self.sixMers[0:5])
		print(self.eventMean[0:5])
		print(self.eventStd[0:5])
		print(self.stutter[0:5])
		print(self.lengthMean[0:5])
		print(self.lengthStd[0:5])
		print(self.modelMeans[0:5])
		print(self.modelStd[0:5])
		print(self.gaps[0:5])
		print(read_positions[0:5])
		'''

		if not len(self.sixMers) == len(self.eventMean) == len(self.eventStd) == len(self.stutter) == len(self.lengthMean) == len(self.lengthStd) == len(self.modelMeans) == len(self.modelStd) == len(self.gaps):
			print("Length Mismatch")
			sys.exit()


#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}


#-------------------------------------------------
#reshape into tensors
def saveRead(trainingRead, readID):

	f = open(folderPath + '/' + readID + '.p', 'wb')
	pickle.dump(trainingRead, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()


#-------------------------------------------------
#pull from training data file
def importFromFile(fname, analogueConc, readID2calls):

	print('Parsing training data file...')

	read_sixMers = []
	read_eventMeans = []
	read_eventStd = []
	read_stutter = []
	read_lengthMeans = []
	read_lengthStd = []
	read_modelMeans = []
	read_modelStd = []
	read_positions = []

	pos_eventMeans = []
	pos_lengths = []

	prevReadID = ''
	prevPos = -1
	
	readsLoaded = 0
	f = open(fname,'r')
	for line in f:

		splitLine = line.rstrip().split('\t')
		if splitLine[0] == 'contig':
			continue

		pos = int(splitLine[1])
		readID = splitLine[3]

		#we've changed position
		if pos != prevPos and readID == prevReadID:

			read_sixMers.append(sixMer)
			read_eventMeans.append(np.mean(pos_eventMeans))
			read_eventStd.append(np.std(pos_eventMeans))
			read_stutter.append(len(pos_eventMeans))
			read_lengthMeans.append(np.mean(pos_lengths))
			read_lengthStd.append(np.std(pos_lengths))
			read_modelMeans.append(modelMean)
			read_modelStd.append(modelStd)
			read_positions.append(prevPos)

			#reset for position
			pos_eventMeans = []
			pos_lengths = []
			prevPos = pos

		#we've changed to a new read
		elif readID != prevReadID:

			if prevReadID in readID2calls and len(read_sixMers) > maxLen:
				readsLoaded += 1
				print('Reads loaded: ',readsLoaded)
				tr = trainingRead(read_sixMers, read_eventMeans, read_eventStd, read_stutter, read_lengthMeans, read_lengthStd, read_modelMeans, read_modelStd, read_positions, prevReadID, analogueConc, readID2calls)
				saveRead(tr, prevReadID)

			#reset for read
			read_sixMers = []
			read_eventMeans = []
			read_eventStd = []
			read_stutter = []
			read_lengthMeans = []
			read_lengthStd = []
			read_modelMeans = []
			read_modelStd = []
			read_positions = []

			#reset for position
			pos_eventMeans = []
			pos_lengths = []
			prevPos = -1
			
		sixMer = splitLine[9]
		modelMean = float(splitLine[10])
		modelStd = float(splitLine[11])
		eventMean = float(splitLine[6])
		eventLength = float(splitLine[8])
		prevReadID = readID
		prevPos = pos
		
		pos_eventMeans.append(eventMean)
		pos_lengths.append(eventLength)

	f.close()
	print('Done.')


#-------------------------------------------------
#MAIN
tup = inputFiles[barcode]
readID2calls = parseDNAscentDetect(tup[2])
importFromFile(tup[1], tup[0], readID2calls)
