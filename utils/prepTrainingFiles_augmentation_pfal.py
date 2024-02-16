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

folderPathAnalogue = sys.argv[3] + '/split_labelled/'
folderPathThym = sys.argv[3] + '/split_unlabelled/'

trainingFilesFolderPath = sys.argv[3] + '/augmented_'+sys.argv[2]

maxRaw = 20
maxLen = 2000

attempts = 0
swaps = 0


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
#one-hot for bases
baseToInt = {'A':0, 'T':1, 'G':2, 'C':3}
intToBase = {0:'A', 1:'T', 2:'G', 3:'C'}


#-------------------------------------------------
#
class trainingRead:

	def __init__(self, read_kmers, read_signals, read_modelMeans, read_positions, read_EdUCalls, read_BrdUCalls, readID, label):
		self.kmers = read_kmers[0:maxLen]
		self.signal = read_signals[0:maxLen]
		self.modelMeans = read_modelMeans[0:maxLen]
		self.readID = readID
		self.label = label
		self.BrdUCalls = read_BrdUCalls[0:maxLen]
		self.EdUCalls = read_EdUCalls[0:maxLen]

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
#reshape into tensors
def saveRead(trainingRead, readID):

	f = open(trainingFilesFolderPath + '/' + readID + '.p', 'wb')
	pickle.dump(trainingRead, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()


#-------------------------------------------------
#MAIN
def fetchMatchingRead(fname, baseChromosome, baseStart, baseEnd, baseStrand,usedIDs):
	f = open(fname,'r')
	read_kmers = []

	for line in f:

		if line[0] == '#':
			continue

		if line[0] == '>':

			#make an object out of this read
			if len(read_kmers) > maxLen + 20:

				if switch:
					usedIDs.append(readID)
					return readID, usedIDs, refPos2feature


			#reset for read
			read_kmers = []
			read_signals = []
			read_modelSignal = []
			read_positions = []
			read_EdUCalls = []
			read_BrdUCalls = []
			pos_signals = []
			refPos2feature = {}

			#reset for position
			pos_eventMeans = []
			prevPos = -1

			splitLine = line.rstrip().split()
			readID = splitLine[0][1:]
			chromosome = splitLine[1]
			mappingStart = int(splitLine[2])
			mappingEnd = int(splitLine[3])
			strand = splitLine[4]
			prevPos = -1
			
			if baseChromosome == chromosome and baseStrand == strand and min(mappingStart,mappingEnd) <= baseStart and max(mappingStart,mappingEnd) >= baseEnd and readID not in usedIDs:
				switch = True
			else:
				switch = False

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
				read_EdUCalls.append(EdUCall)
				read_BrdUCalls.append(BrdUCall)

				refPos2feature[prevPos] = (kmer,pos_signals,modelSignal,prevPos, EdUCall, BrdUCall)

				#reset for position
				pos_signals = []
				prevPos = pos

			kmer = splitLine[3]
			modelSignal = float(splitLine[4])
			signal = float(splitLine[2])
			prevReadID = readID
			prevPos = pos

			BrdUCall = '-'
			EdUCall = '-'
			if len(splitLine) >= 6:
				EdUCall = splitLine[5]
				BrdUCall = splitLine[6]
				
			pos_signals.append(signal)

	f.close()

	return '',usedIDs,{}


#-------------------------------------------------
#MAIN

usedIDs = []

print('Parsing training data file...')

set2Chromosomes = {0:["1","2"], 1:["3","4"],2:["5","6"],3:["7","8"],4:["9","10"],5:["11","12"],6:["13","14"]}

for matchChr in set2Chromosomes[int(sys.argv[1])]:
	maxLength = getChromosomeLength( matchChr )
	marks = range(0,maxLength,100000)
	for strand in ['fwd','rev']:
		for m in marks:

			#set filenames
			fn_thymidineTrainingData = folderPathThym + matchChr + '_' + strand + '_' + str(m) + '.trainingData'
			fn_analogueTrainingData = folderPathAnalogue + matchChr + '_' + strand + '_' + str(m) + '.trainingData'

			if not os.path.isfile(fn_thymidineTrainingData) or not os.path.isfile(fn_analogueTrainingData):
				print(fn_thymidineTrainingData, fn_analogueTrainingData)
				continue

			read_kmers = []
			read_signals = []
			read_modelSignal = []
			read_positions = []
			read_EdUCalls = []
			read_BrdUCalls = []
			pos_signals = []

			prevReadID = ''
			prevPos = -1
			usedIDs = []

			readsLoaded = 0
			switch = False
			f = open(fn_analogueTrainingData,'r')
			for line in f:

				if line[0] == '#':
					continue

				if line[0] == '>':

					readsLoaded += 1
					#print('Reads loaded: ',readsLoaded)

					#make an object out of this read
					if len(read_kmers) > maxLen + 20:

						if switch:

							matchID, usedIDs, matchingReadDic = fetchMatchingRead(fn_thymidineTrainingData, chromosome, read_positions[0], read_positions[-1], strand, usedIDs)

							if len(matchingReadDic) > 0:

								augmented_kmers = []
								augmented_signals = []
								augmented_modelSignal = []
								augmented_positions = []
								augmented_EdUCalls = []
								augmented_BrdUCalls = []

								#mix
								unlabelledWindow = []
								for i in range(0, len(read_positions)-12):

									#look ahead			
									if read_kmers[i+8][0] == 'T' and random.choice(range(0,int(sys.argv[2]))) == 0:

										#make sure all the reference positions are defined
										noDels = True
										for j in range(i+1,i+10):
											if abs(read_positions[j] - read_positions[j-1]) > 1 or read_positions[j-1] not in matchingReadDic:
												noDels = False
												break

										#for the next window of bases, pull from the analogue-negative read
										if noDels:
											unlabelledWindow = range(i,i+9)
											#swaps += 1
											#print(swaps/attempts)

									if i in unlabelledWindow:

										#testing
										'''
										print('----------------------------',i,readID)
										print(read_kmers[i], matchingReadDic[read_positions[i]][0])
										print(np.mean(read_signals[i]), np.mean(matchingReadDic[read_positions[i]][1]))
										print(read_positions[i], matchingReadDic[read_positions[i]][2])
										'''

										augmented_kmers.append(matchingReadDic[read_positions[i]][0])
										augmented_signals.append(matchingReadDic[read_positions[i]][1])
										augmented_modelSignal.append(matchingReadDic[read_positions[i]][2])
										augmented_positions.append(matchingReadDic[read_positions[i]][3])
										augmented_EdUCalls.append(matchingReadDic[read_positions[i]][4]+'X')
										augmented_BrdUCalls.append(matchingReadDic[read_positions[i]][5]+'X')

									else:

										augmented_kmers.append(read_kmers[i])
										augmented_signals.append(read_signals[i])
										augmented_modelSignal.append(read_modelSignal[i])
										augmented_positions.append(read_positions[i])
										augmented_EdUCalls.append(read_EdUCalls[i])
										augmented_BrdUCalls.append(read_BrdUCalls[i])

								numReadSlices = math.floor(float(len(augmented_kmers))/float(maxLen))

								for s in range(numReadSlices):


									tr = trainingRead(augmented_kmers[maxLen*s:maxLen*(s+1)], augmented_signals[maxLen*s:maxLen*(s+1)], augmented_modelSignal[maxLen*s:maxLen*(s+1)], augmented_positions[maxLen*s:maxLen*(s+1)], augmented_EdUCalls[maxLen*s:maxLen*(s+1)], augmented_BrdUCalls[maxLen*s:maxLen*(s+1)], prevReadID+'.'+matchID+'_slice'+str(s), -1)
									saveRead(tr, prevReadID+'.'+matchID+'_slice'+str(s))

					#reset for read
					read_kmers = []
					read_signals = []
					read_modelSignal = []
					read_positions = []
					read_EdUCalls = []
					read_BrdUCalls = []
					pos_signals = []

					#reset for position
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
						read_EdUCalls.append(EdUCall)
						read_BrdUCalls.append(BrdUCall)

						#reset for position
						pos_signals = []
						prevPos = pos

					kmer = splitLine[3]
					modelSignal = float(splitLine[4])
					signal = float(splitLine[2])
					prevReadID = readID
					prevPos = pos

					BrdUCall = '-'
					EdUCall = '-'
					if len(splitLine) >= 6:
						EdUCall = splitLine[5]
						BrdUCall = splitLine[6]
						
					pos_signals.append(signal)

			f.close()
			print('Done.')
