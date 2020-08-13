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

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/commit620d798_trainingData_8features_bc8bc12_augmentation'

trainingFilesFolderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/commit620d798_trainingData_8features_bc8bc12_augmentation/trainingFiles_gap10_withStd'

bc08dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/commit620d798_l_4000_q_20.barcode08.trainingData'
bc12dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode12/commit620d798_l_4000_q_20.trainingData'

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

	def __init__(self, read_sixMers, read_eventMeans, read_eventStd, read_eventLength, read_modelMeans, read_modelStd, read_positions, readID, analogueConc, logLikelihood, matchingReadDic):

		self.sixMers = []
		self.eventMean = []
		self.eventStd = []
		self.eventLength = []
		self.modelMeans = []
		self.modelStd =[]
		self.logLikelihood = []

		#mix
		BrdUwindow = []
		for i in range(0, len(read_positions)-12):

			#look ahead			
			if read_sixMers[i+5][0] == 'T' and random.choice(range(0,10)) == 0:

				#make sure all the reference positions are defined
				noDels = True
				for j in range(i+1,i+13):
					if abs(read_positions[j] - read_positions[j-1]) > 1 or read_positions[j-1] not in matchingReadDic:
						noDels = False
						break

				#make sure we have a positive BrdU call from the HMM
				positiveCall = False
				if noDels:
					if matchingReadDic[read_positions[i+5]][6] != '-':
						if float(matchingReadDic[read_positions[i+5]][6]) > llThreshold:
							positiveCall = True


				#for the next 12 bases, pull from the BrdU read
				if noDels and positiveCall:
					BrdUwindow = range(i,i+12)

			if i in BrdUwindow:

				#testing
				
				print('----------------------------',i, readID)
				print(read_sixMers[i], matchingReadDic[read_positions[i]][0])
				print(read_eventMeans[i], matchingReadDic[read_positions[i]][1])
				print(read_eventStd[i], matchingReadDic[read_positions[i]][2])
				print(read_eventLength[i], matchingReadDic[read_positions[i]][3])
				print(read_modelMeans[i], matchingReadDic[read_positions[i]][4])
				print(read_modelStd[i], matchingReadDic[read_positions[i]][5])
				print(matchingReadDic[read_positions[i]][6]+'X')
				

				self.sixMers.append(matchingReadDic[read_positions[i]][0])
				self.eventMean.append(matchingReadDic[read_positions[i]][1])
				self.eventStd.append(matchingReadDic[read_positions[i]][2])
				self.eventLength.append(matchingReadDic[read_positions[i]][3])
				self.modelMeans.append(matchingReadDic[read_positions[i]][4])
				self.modelStd.append(matchingReadDic[read_positions[i]][5])
				self.logLikelihood.append(matchingReadDic[read_positions[i]][6]+'X')

			else:
				
				self.sixMers.append(read_sixMers[i])
				self.eventMean.append(read_eventMeans[i])
				self.eventStd.append(read_eventStd[i])
				self.eventLength.append(read_eventLength[i])
				self.modelMeans.append(read_modelMeans[i])
				self.modelStd.append(read_modelStd[i])
				self.logLikelihood.append(logLikelihood[i])




		self.sixMers = self.sixMers[0:maxLen]
		self.eventMean = self.eventMean[0:maxLen]
		self.eventStd = self.eventStd[0:maxLen]
		self.eventLength = self.eventLength[0:maxLen]
		self.modelMeans = self.modelMeans[0:maxLen]
		self.modelStd = self.modelStd[0:maxLen]
		self.logLikelihood = self.logLikelihood[0:maxLen]

		self.readID = readID
		self.analogueConc = -1

		gaps = [1]
		for i in range(1, maxLen):
			gaps.append( abs(read_positions[i] - read_positions[i-1]) )
		self.gaps = gaps

		'''
		print(self.sixMers[0:5])
		print(self.eventMean[0:5])
		print(self.eventStd[0:5])
		print(self.stutter[0:5])
		print(self.eventLength[0:5])
		print(self.modelMeans[0:5])
		print(self.modelStd[0:5])
		print(self.gaps[0:5])
		print(read_positions[0:5])
		print(logLikelihood[0:5])
		print('-----------------------')
		'''

		if not len(self.sixMers) == len(self.eventMean) == len(self.eventStd) == len(self.eventLength) == len(self.modelMeans) == len(self.modelStd) == len(self.gaps) == len(self.logLikelihood):
			print(len(self.sixMers), len(self.logLikelihood))
			print("Length Mismatch")
			sys.exit()


#-------------------------------------------------
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}


#-------------------------------------------------
#reshape into tensors
def saveRead(trainingRead, readID):

	f = open(trainingFilesFolderPath + '/' + readID + '.p', 'wb')
	pickle.dump(trainingRead, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()


#-------------------------------------------------
#reshape into tensors
def weighted_avg_and_std(values, weights):
	"""
	Return the weighted average and standard deviation.

	values, weights -- Numpy ndarrays with the same shape.
	"""

	values = np.array(values)
	weights = np.array(weights)

	average = np.average(values, weights=weights)
	# Fast and numerically precise:
	variance = np.average((values-average)**2, weights=weights)
	return math.sqrt(variance)


#-------------------------------------------------
#MAIN
def fetchMatchingRead(fname, baseChromosome, baseStart, baseEnd, baseStrand,usedIDs):

	g = open(fname,'r')

	read_sixMers = []
	read_eventMeans = []
	read_eventStd = []
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

	for line in g:

		if line[0] == '#':
			continue

		if line[0] == '>':

			#make an object out of this read
			if len(read_sixMers) > maxLen + 20:

				if switch:
					usedIDs.append(readID)
					return readID, usedIDs, refPos2feature

			#reset for read
			read_sixMers = []
			read_eventMeans = []
			read_eventStd = []
			read_eventLength = []
			read_modelMeans = []
			read_modelStd = []
			read_positions = []
			read_BrdUcalls = []
			refPos2feature = {}

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

			if chromosome != 'chrM' and baseChromosome == chromosome and baseStrand == strand and min(mappingStart,mappingEnd) <= baseStart and max(mappingStart,mappingEnd) >= baseEnd and readID not in usedIDs:
				#print(baseChromosome,chromosome,baseStrand,strand,min(mappingStart,mappingEnd),baseStart,max(mappingStart,mappingEnd),baseEnd)
				switch = True
			else:
				switch = False

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

				read_eventMeans.append(np.average(pos_eventMeans, weights=pos_lengths))
				read_eventStd.append(weighted_avg_and_std(pos_eventMeans, pos_lengths))

				read_eventLength.append(sum(pos_lengths))
				read_modelMeans.append(modelMean)
				read_modelStd.append(modelStd)
				read_positions.append(prevPos)
				read_BrdUcalls.append(BrdUcall)

				refPos2feature[prevPos] = (sixMer, np.average(pos_eventMeans, weights=pos_lengths), weighted_avg_and_std(pos_eventMeans, pos_lengths), sum(pos_lengths), modelMean, modelStd, BrdUcall)

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

	return '',usedIDs,{}


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
			bc12dnascent = folderPath+'/bc12_'+matchChr+'_'+strand+'_'+str(m)+'.detect'

			if not os.path.isfile(bc08dnascent) or not os.path.isfile(bc12dnascent):
				print(bc08dnascent,bc12dnascent)
				continue

			read_sixMers = []
			read_eventMeans = []
			read_eventStd = []
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
			usedIDs = []

			readsLoaded = 0
			switch = False
			f = open(bc08dnascent,'r')
			for line in f:

				if line[0] == '#':
					continue

				if line[0] == '>':

					readsLoaded += 1
					#print('Reads loaded: ',readsLoaded)

					#make an object out of this read
					if len(read_sixMers) > maxLen + 20:

						if switch:

							matchID, usedIDs, matchingReadDic = fetchMatchingRead(bc12dnascent,chromosome, read_positions[0], read_positions[maxLen], strand, usedIDs)

							if len(matchingReadDic) > 0:
								tr = trainingRead(read_sixMers, read_eventMeans, read_eventStd, read_eventLength, read_modelMeans, read_modelStd, read_positions, prevReadID+'.'+matchID, -1, read_BrdUcalls, matchingReadDic)
								saveRead(tr, prevReadID+'.'+matchID)

					#reset for read
					read_sixMers = []
					read_eventMeans = []
					read_eventStd = []
					read_eventLength = []
					read_modelMeans = []
					read_modelStd = []
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

						read_eventMeans.append(np.average(pos_eventMeans, weights=pos_lengths))
						read_eventStd.append(weighted_avg_and_std(pos_eventMeans, pos_lengths))

						read_eventLength.append(sum(pos_lengths))
						read_modelMeans.append(modelMean)
						read_modelStd.append(modelStd)
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
