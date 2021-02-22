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

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_CTC'

bc08dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/DNAscentv2.raw.trainingData'
bc10dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode10/DNAscentv2.raw.trainingData'
bc11dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode11/DNAscentv2.raw.trainingData'
bc12dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode12/DNAscentv2.raw.trainingData'
augmented = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/CTCtraining_augmentation/bc08_bc12.augmented'

inputFiles = [(0., bc08dnascent),(0.26, bc10dnascent),(0.5, bc11dnascent),(0.8, bc12dnascent),(-1,augmented)]
maxLen = 2000
#maxReads = int(sys.argv[2])

llThreshold = 1.25

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
	read_raw = []
	read_eventLength = []
	read_modelMeans = []
	read_modelStd = []
	seqIdx2LL = {}
	prevPos = -1
	splitCounter = 0
	
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

				#increment this one more time for the last position of the read
				labelLength += 1

				tr = trainingRead(read_sequence, read_raw, read_modelMeans, read_modelStd, seqIdx2LL, readID+'-'+str(splitCounter), analogueConc, labelLength)
				saveRead(tr, readID+'-'+str(splitCounter))
				#if readsLoaded >= maxReads:
				#	break

			#reset for read
			read_sequence = []
			read_raw = []
			read_modelMeans = []
			read_modelStd = []
			seqIdx2LL = {}
			prevPos = -1
			prevSixmer = "M"
			labelLength = 0
			rawCount = 0
			splitCounter = 0

			#get the header for this read
			splitLine = line.rstrip().split()
			readID = splitLine[0][1:]
			mappingStart = int(splitLine[2])
			mappingEnd = int(splitLine[3])


		else:

			if rawCount > maxLen:

				#increment this one more time for the last position of the read
				labelLength += 1
				tr = trainingRead(read_sequence, read_raw, read_modelMeans, read_modelStd, seqIdx2LL, readID+'-'+str(splitCounter), analogueConc, labelLength)
				saveRead(tr, readID+'-'+str(splitCounter))
				splitCounter += 1

				#reset
				read_sequence = []
				read_raw = []
				read_modelMeans = []
				read_modelStd = []
				seqIdx2LL = {}
				prevPos = -1
				prevSixmer = "M"
				labelLength = 0
				rawCount = 0


			splitLine = line.rstrip().split('\t')

			pos = int(splitLine[0])

			#don't get too close to the ends of the read, because we can't get an HMM detect there
			if abs(pos-mappingStart) < 50 or abs(pos-mappingEnd) < 50:
				continue

			#get the data from the line
			sixMer = splitLine[4]
			modelMean = float(splitLine[5])
			modelStd = float(splitLine[6])
			rawSignal = float(splitLine[2])

			#we've changed position
			if (pos != prevPos or (sixMer[0] == "N" and prevSixmer != "N")) and prevPos != -1:

				labelLength += 1
				read_sequence.append('-') #place a gap to signify we've moved

			#append it
			read_sequence.append(sixMer[0])
			read_raw.append(rawSignal)
			read_modelMeans.append(modelMean)
			read_modelStd.append(modelStd)

			prevPos = pos
			prevSixmer = sixMer[0]

			#sort out BrdU calls from the HMM (if there is one)
			if len(splitLine) == 8:
				seqIdx2LL[len(read_sequence) - 1] = splitLine[7]

			rawCount += 1

	f.close()
	print('Done.')


#-------------------------------------------------
#MAIN
tup = inputFiles[barcode]
importFromFile(tup[0], tup[1])
