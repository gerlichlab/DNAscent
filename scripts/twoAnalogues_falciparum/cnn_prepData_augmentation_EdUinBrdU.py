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

folderPathEdU = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdU_trainingData/splitData_CNNbootstrap'
folderPathBrdU = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/BrdU_trainingData/splitData_CNNbootstrap'

trainingFilesFolderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdUinBrdU_trainingData/gap40'

f_analoguePositiveIDs_EdU = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/EdU_trainingData/analogueIDs.txt'
f_analoguePositiveIDs_BrdU = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/BrdU_trainingData/analogueIDs.txt'

maxLen = 4000
llThreshold = 1.25

analoguePositiveIDs_EdU = []
f = open(f_analoguePositiveIDs_EdU,'r')
for line in f:
	analoguePositiveIDs_EdU.append(line.rstrip())
f.close()

analoguePositiveIDs_BrdU = []
f = open(f_analoguePositiveIDs_BrdU,'r')
for line in f:
	analoguePositiveIDs_BrdU.append(line.rstrip())
f.close()


def getChromosomeLength( chromosome ):
#get chromosome length by number in kb
	
	if chromosome == "1":
		return 640851
	elif chromosome == "2":
		return 947102
	elif chromosome == "3":
		return 1067971
	elif chromosome == "4":
		return 1200490
	elif chromosome == "5":
		return 1343557
	elif chromosome == "6":
		return 1418242
	elif chromosome == "7":
		return 1445207
	elif chromosome == "8":
		return 1472805
	elif chromosome == "9":
		return 1541735
	elif chromosome == "10":
		return 1687656
	elif chromosome == "11":
		return 2038340
	elif chromosome == "12":
		return 2271494
	elif chromosome == "13":
		return 2925236
	elif chromosome == "14":
		return 3291936


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, augmented_sixMers, augmented_eventMean, augmented_eventStd, augmented_eventLength, augmented_modelMeans, augmented_modelStd, logLikelihood, readID, analogueConc):


		self.sixMers = augmented_sixMers
		self.eventMean = augmented_eventMean
		self.eventStd = augmented_eventStd
		self.eventLength = augmented_eventLength
		self.modelMeans = augmented_modelMeans
		self.modelStd = augmented_modelStd
		self.logLikelihood = logLikelihood
		self.readID = readID
		self.analogueConc = analogueConc

		if not len(self.sixMers) == len(self.eventMean) == len(self.eventStd) == len(self.eventLength) == len(self.modelMeans) == len(self.modelStd):
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
	read_EdUcalls = []

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
			read_EdUcalls = []
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

			if chromosome != 'chrM' and baseChromosome == chromosome and baseStrand == strand and min(mappingStart,mappingEnd) <= baseStart and max(mappingStart,mappingEnd) >= baseEnd and readID not in usedIDs and readID in analoguePositiveIDs_BrdU:
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
				read_eventMeans.append(np.mean(pos_eventMeans))
				read_eventStd.append(np.std(pos_eventMeans))
				read_eventLength.append(sum(pos_lengths))
				read_modelMeans.append(modelMean)
				read_modelStd.append(modelStd)
				read_positions.append(prevPos)
				read_EdUcalls.append(EdUcall)

				refPos2feature[prevPos] = (sixMer,np.mean(pos_eventMeans),sum(pos_lengths),modelMean,modelStd,EdUcall)

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

			#sort out EdU calls
			EdUcall = '-'
			if len(splitLine) == 8:
				EdUcall = splitLine[7]
						
			pos_eventMeans.append(eventMean)
			pos_lengths.append(eventLength)

	return '',usedIDs,{}


#-------------------------------------------------
#MAIN

print('Parsing training data file...')

allChromosomes = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14"]

set2Chromosomes = {0:["1","2"], 1:["3","4"],2:["5","6"],3:["7","8"],4:["9","10"],5:["11","12"],6:["13","14"]}

#for matchChr in allChromosomes:
for matchChr in set2Chromosomes[int(sys.argv[1])]:
	maxLength = getChromosomeLength( matchChr )
	marks = range(0,maxLength,100000)
	for strand in ['fwd','rev']:
		for m in marks:

			#set filenames
			bc08dnascent = folderPathBrdU+'/BrdU_'+matchChr+'_'+strand+'_'+str(m)+'.trainingData'
			bc12dnascent = folderPathEdU+'/EdU_'+matchChr+'_'+strand+'_'+str(m)+'.trainingData'

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
			read_EdUcalls = []

			pos_eventMeans = []
			pos_lengths = []

			prevReadID = ''
			prevPos = -1
			usedIDs = []

			readsLoaded = 0
			switch = False
			f = open(bc12dnascent,'r')
			for line in f:

				if line[0] == '#':
					continue

				if line[0] == '>':

					readsLoaded += 1
					#print('Reads loaded: ',readsLoaded)

					#make an object out of this read
					if len(read_sixMers) > maxLen + 20:

						if switch:

							matchID, usedIDs, matchingReadDic = fetchMatchingRead(bc08dnascent,chromosome, read_positions[0], read_positions[-1], strand, usedIDs)

							if len(matchingReadDic) > 0:

								augmented_sixMers = []
								augmented_eventMean = []
								augmented_eventStd = []
								augmented_eventLength = []
								augmented_modelMeans = []
								augmented_modelStd =[]
								augmented_logLikelihood = []

								#mix
								ThymWindow = []
								for i in range(0, len(read_positions)-12):

									#look ahead			
									if read_sixMers[i+5][0] == 'T' and random.choice(range(0,40)) == 0:

										#make sure all the reference positions are defined
										noDels = True
										for j in range(i+1,i+13):
											if abs(read_positions[j] - read_positions[j-1]) > 1 or read_positions[j-1] not in matchingReadDic:
												noDels = False
												break

										#make sure we have a positive EdU call from the HMM
										#positiveCall = False
										#if noDels:
										#	if matchingReadDic[read_positions[i+5]][5] != '-':
										#		if float(matchingReadDic[read_positions[i+5]][5]) > llThreshold:
										#			positiveCall = True


										#for the next 12 bases, pull from the EdU read
										if noDels:# and positiveCall:
											ThymWindow = range(i,i+12)

									if i in ThymWindow:

										#testing
										'''
										print('----------------------------',i, readID)
										print(read_sixMers[i], matchingReadDic[read_positions[i]][0])
										print(read_eventMeans[i], matchingReadDic[read_positions[i]][1])
										print(read_eventLength[i], matchingReadDic[read_positions[i]][2])
										print(read_modelMeans[i], matchingReadDic[read_positions[i]][3])
										print(read_modelStd[i], matchingReadDic[read_positions[i]][4])
										print(matchingReadDic[read_positions[i]][5]+'X')
										'''

										augmented_sixMers.append(matchingReadDic[read_positions[i]][0])
										augmented_eventMean.append(matchingReadDic[read_positions[i]][1])
										augmented_eventStd.append(0)
										augmented_eventLength.append(matchingReadDic[read_positions[i]][2])
										augmented_modelMeans.append(matchingReadDic[read_positions[i]][3])
										augmented_modelStd.append(matchingReadDic[read_positions[i]][4])
										augmented_logLikelihood.append(matchingReadDic[read_positions[i]][5]+'X')

									else:
										
										augmented_sixMers.append(read_sixMers[i])
										augmented_eventMean.append(read_eventMeans[i])
										augmented_eventStd.append(0)
										augmented_eventLength.append(read_eventLength[i])
										augmented_modelMeans.append(read_modelMeans[i])
										augmented_modelStd.append(read_modelStd[i])
										augmented_logLikelihood.append(read_EdUcalls[i])

								numReadSlices = math.floor(float(len(augmented_sixMers))/float(maxLen))

								for s in range(numReadSlices):

									tr = trainingRead(augmented_sixMers[maxLen*s:maxLen*(s+1)], augmented_eventMean[maxLen*s:maxLen*(s+1)], augmented_eventStd[maxLen*s:maxLen*(s+1)], augmented_eventLength[maxLen*s:maxLen*(s+1)], augmented_modelMeans[maxLen*s:maxLen*(s+1)], augmented_modelStd[maxLen*s:maxLen*(s+1)], augmented_logLikelihood[maxLen*s:maxLen*(s+1)], prevReadID+'.'+matchID+'_slice'+str(s), -3)
									saveRead(tr, prevReadID+'.'+matchID+'_slice'+str(s))

					#reset for read
					read_sixMers = []
					read_eventMeans = []
					read_eventStd = []
					read_eventLength = []
					read_modelMeans = []
					read_modelStd = []
					read_positions = []
					read_EdUcalls = []

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

					if chromosome != 'chrM' and readID in analoguePositiveIDs_EdU:
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
						read_eventMeans.append(np.mean(pos_eventMeans))
						read_eventStd.append(np.std(pos_eventMeans))
						read_eventLength.append(sum(pos_lengths))
						read_modelMeans.append(modelMean)
						read_modelStd.append(modelStd)
						read_positions.append(prevPos)
						read_EdUcalls.append(EdUcall)

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

					#sort out EdU calls
					EdUcall = '-'
					if len(splitLine) == 8:
						EdUcall = splitLine[7]
							
					pos_eventMeans.append(eventMean)
					pos_lengths.append(eventLength)

			f.close()
			print('Done.')
