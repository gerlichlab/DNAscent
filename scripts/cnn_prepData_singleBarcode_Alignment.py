import sys
import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
import random

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/commit620d798_trainingData_8features_bc8bc12_augmentation'

trainingFilesFolderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/data_alignment_bc08'

bc08dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/commit620d798_l_4000_q_20.barcode08.trainingData'

maxLen = 4000
llThreshold = 1.25


def getChromosomeLength( chromosome ):
#get chromosome length by number in kb
	
	if chromosome == "chrI":
		return 230000
	elif chromosome == "chrII":
		return 813000
	elif chromosome == "chrIII":
		return 317000
	elif chromosome == "chrIV":
		return 1532000
	elif chromosome == "chrV":
		return 577000
	elif chromosome == "chrVI":
		return 270000
	elif chromosome == "chrVII":
		return 1091000
	elif chromosome == "chrVIII":
		return 563000
	elif chromosome == "chrIX":
		return 440000
	elif chromosome == "chrX":
		return 746000
	elif chromosome == "chrXI":
		return 667000
	elif chromosome == "chrXII":
		return 1078000
	elif chromosome == "chrXIII":
		return 924000
	elif chromosome == "chrXIV":
		return 784000
	elif chromosome == "chrXV":
		return 1091000
	elif chromosome == "chrXVI":
		return 948000


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_sixMers, read_eventMeans, read_eventLength, read_positions, readID, analogueConc, logLikelihood, matchingReadDic):

		self.sixMers = read_sixMers[0:maxLen]
		self.eventMean = read_eventMeans[0:maxLen]
		self.eventLength = read_eventLength[0:maxLen]
		self.logLikelihood = logLikelihood[0:maxLen]
		self.readID = readID
		self.positions = read_positions
		self.analogueConc = -1

		if not len(self.sixMers) == len(self.eventMean) == len(self.eventLength) == len(self.logLikelihood):
			print(len(self.sixMers), len(self.logLikelihood))
			print("Length Mismatch")
			sys.exit()


#-------------------------------------------------
#reshape into tensors
def saveRead(trainingRead, readID):

	f = open(trainingFilesFolderPath + '/' + readID + '.p', 'wb')
	pickle.dump(trainingRead, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

#-------------------------------------------------
#MAIN

print('Parsing training data file...')

allChromosomes = ["chrI","chrII","chrIII","chrIV","chrV","chrVI","chrVII","chrVIII","chrIX","chrX","chrXI","chrXII","chrXIII","chrXIV","chrXV","chrXVI"]

set2Chromosomes = {0:["chrI","chrII"], 1:["chrIII","chrIV"],2:["chrV","chrVI"],3:["chrVII","chrVIII"],4:["chrIX","chrX"],5:["chrXI","chrXII"],6:["chrXIII","chrXIV"],7:["chrXV","chrXVI"]}

#for matchChr in allChromosomes:
for matchChr in set2Chromosomes[int(sys.argv[1])]:
	maxLength = getChromosomeLength( matchChr )
	marks = range(0,maxLength,100000)
	for strand in ['fwd','rev']:
		for m in marks:

			#set filenames
			bc08dnascent = folderPath+'/bc08_'+matchChr+'_'+strand+'_'+str(m)+'.detect'

			if not os.path.isfile(bc08dnascent):
				print(bc08dnascent)
				continue

			read_sixMers = []
			read_eventMeans = []
			read_eventLength = []
			read_positions = []
			read_BrdUcalls = []

			pos_eventMeans = []
			pos_lengths = []

			prevReadID = ''
			prevPos = -1
			usedIDs = []

			readsLoaded = 0
			switch = False
			f = open(bc08dnascent,'r')
			for line in f:

				if line[0] == '#':
					continue

				if line[0] == '>':

					readsLoaded += 1
					
					print('Reads loaded: ',readsLoaded)

					#make an object out of this read
					if len(read_sixMers) > maxLen + 20:

						if switch:

							if strand == 'rev':
								read_sixMers = read_sixMers[::-1]
								read_eventMeans = read_eventMeans[::-1]
								read_eventLength = read_eventLength[::-1]
								read_positions = read_positions[::-1]
								read_BrdUcalls = read_BrdUcalls[::-1]

							tr = trainingRead(read_sixMers, read_eventMeans, read_eventLength, read_positions, prevReadID, -1, read_BrdUcalls, {})
							saveRead(tr, prevReadID)

					#reset for read
					read_sixMers = []
					read_eventMeans = []
					read_eventLength = []
					read_positions = []
					read_BrdUcalls = []

					#reset for position
					pos_eventMeans = []
					pos_lengths = []
					prevPos = -1

					splitLine = line.rstrip().split()
					readID = splitLine[0][1:]
					chromosome = splitLine[1]
					mappingStart = int(splitLine[2])
					mappingEnd = int(splitLine[3])
					strand = splitLine[4]
					prevPos = -1

					if chromosome != 'chrM':
						switch = True
					else:
						switch = False

				elif switch:

					splitLine = line.rstrip().split('\t')

					pos = int(splitLine[0])

					#don't get too close to the ends of the read, because we can't get an HMM detect there
					if abs(pos-mappingStart) < 26 or abs(pos-mappingEnd) < 26:
						continue

					#we've changed position
					if pos != prevPos and prevPos != -1:

						read_sixMers.append(sixMer)
						read_eventMeans.append(pos_eventMeans)
						read_eventLength.append(pos_lengths)
						read_positions.append(prevPos)
						read_BrdUcalls.append(BrdUcall)

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

					#sort out BrdU calls
					BrdUcall = '-'
					if len(splitLine) == 8:
						BrdUcall = splitLine[7]
							
					pos_eventMeans.append(eventMean)
					pos_lengths.append(eventLength)

			f.close()
			print('Done.')
