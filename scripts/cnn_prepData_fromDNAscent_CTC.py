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

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_DNAscentTrainingData_highIns_noBrdUScaling_CTC'

bc08dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/commite338d93_l_3000_q_20.barcode08.trainingData'
bc10dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode10/commite338d93_l_3000_q_20.barcode10.trainingData'
bc11dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode11/commite338d93_l_3000_q_20.barcode11.trainingData'
bc12dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode12/commite338d93_l_3000_q_20.barcode12.trainingData'

inputFiles = [(0., bc08dnascent),(0.26, bc10dnascent),(0.5, bc11dnascent),(0.8, bc12dnascent)]
maxLen = 3000
maxReads = 26000

llThreshold = 1.25

class trainingRead:

	def __init__(self, read_sequence, read_eventMeans, read_eventLengths, read_modelMeans, read_modelStdvs, seqIdx2LL, readID, analogueConc):
		
		#reshape log likelihood calls to sequence
		logLikelihood = []
		for i in range(0, len(read_sequence)):
			if i in seqIdx2LL:
				logLikelihood.append(seqIdx2LL[i])
			else:
				logLikelihood.append('-')


		self.sequence = read_sequence
		self.eventMeans = read_eventMeans
		self.eventLengths = read_eventLengths
		self.modelMeans = read_modelMeans
		self.modelStdvs = read_modelStdvs
		self.readID = readID
		self.analogueConc = analogueConc
		self.logLikelihood = logLikelihood

		'''
		print(self.sequence[0:40])
		print(self.eventMeans[0:5])
		print(self.eventLengths[0:5])
		print(self.modelMeans[0:5])
		print(self.modelStdvs[0:5])
		print(self.readID)
		print(self.analogueConc)
		print(self.logLikelihood[0:5])
		print('-----------------------')
		'''




#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3, 'B':5, 'X':4, '-':5}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C', 5:'B', 4:'X', 5:'-'}


#-------------------------------------------------
#reshape into tensors
def saveRead(trainingRead, readID):

	f = open(folderPath + '/' + readID + '.p', 'wb')
	pickle.dump(trainingRead, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()


#-------------------------------------------------
#pull from training data file
def importFromFile(analogueConc, fname):

	print('Parsing training data file...')

	read_sequence = []
	read_eventMeans = []
	read_eventLength = []
	read_modelMeans = []
	read_modelStd = []
	seqIdx2LL = {}
	prevPos = -1
	
	readsLoaded = 0
	f = open(fname,'r')
	for line in f:

		#skip header
		if line[0] == '#':
			continue

		if line[0] == '>':

			#make an object out of this read
			if len(read_sequence) > maxLen:
				readsLoaded += 1
				if readsLoaded % 100 == 0:
					print('Reads loaded: ',readsLoaded)
				tr = trainingRead(read_sequence, read_eventMeans, read_eventLength, read_modelMeans, read_modelStd, seqIdx2LL, readID, analogueConc)
				saveRead(tr, readID)
				if readsLoaded >= maxReads:
					break

			#reset for read
			read_sequence = []
			read_eventMeans = []
			read_eventLength = []
			read_modelMeans = []
			read_modelStd = []
			seqIdx2LL = {}
			prevPos = -1

			#get the header for this read
			splitLine = line.rstrip().split()
			readID = splitLine[0][1:]
			mappingStart = int(splitLine[2])
			mappingEnd = int(splitLine[3])


		else:

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

				if abs(pos - prevPos) > 1:
					for dummy in range(prevPos,pos):
						read_sequence.append('-') #append blank
						read_sequence.append('X') #append gap

				read_sequence.append('-') #place a gap to signify we've moved

			#get the data from the line
			sixMer = splitLine[4]
			modelMean = float(splitLine[5])
			modelStd = float(splitLine[6])
			eventMean = float(splitLine[2])
			eventLength = float(splitLine[3])

			#append it
			read_sequence.append(sixMer[0])
			read_eventMeans.append(eventMean)
			read_eventLength.append(eventLength)
			read_modelMeans.append(modelMean)
			read_modelStd.append(modelStd)

			prevPos = pos

			#sort out BrdU calls
			BrdUcall = '-'
			if len(splitLine) == 8:
				seqIdx2LL[len(read_sequence) - 1] = splitLine[7]

	f.close()
	print('Done.')


#-------------------------------------------------
#MAIN
tup = inputFiles[barcode]
importFromFile(tup[0], tup[1])
