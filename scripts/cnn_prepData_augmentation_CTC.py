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

#bc08dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/DNAscentv2.raw.trainingData'
#bc12dnascent = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode12/DNAscentv2.raw.trainingData'

folderPath = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/CTCtraining_augmentation'

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
#MAIN
def fetchMatchingRead(fname, baseChromosome, baseStart, baseEnd, baseStrand,usedIDs):

	g = open(fname,'r')

	read_sixMers = []
	read_raw = []
	read_modelMeans = []
	read_modelStd = []
	read_positions = []
	read_BrdUcalls = []

	pos_eventMeans = []

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
			read_raw = []
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

				read_raw.append(pos_eventMeans)

				read_modelMeans.append(modelMean)
				read_modelStd.append(modelStd)
				read_positions.append(prevPos)
				read_BrdUcalls.append(BrdUcall)

				refPos2feature[prevPos] = (sixMer, pos_eventMeans, modelMean, modelStd, BrdUcall)

				#reset for position
				pos_eventMeans = []
				prevPos = pos

			sixMer = splitLine[4]
			modelMean = float(splitLine[5])
			modelStd = float(splitLine[6])
			eventMean = float(splitLine[2])
			prevReadID = readID
			prevPos = pos

			#sort out BrdU calls
			BrdUcall = '-'
			if len(splitLine) == 8:
				BrdUcall = splitLine[7]
						
			pos_eventMeans.append(eventMean)


	return '',usedIDs,{}


#-------------------------------------------------
#MAIN

print('Parsing training data file...')

allChromosomes = ["chrI","chrII","chrIII","chrIV","chrV","chrVI","chrVII","chrVIII","chrIX","chrX","chrXI","chrXII","chrXIII","chrXIV","chrXV","chrXVI"]

set2Chromosomes = {0:["chrI","chrII"], 1:["chrIII","chrIV"],2:["chrV","chrVI"],3:["chrVII","chrVIII"],4:["chrIX","chrX"],5:["chrXI","chrXII"],6:["chrXIII","chrXIV"],7:["chrXV","chrXVI"]}

for matchChr in set2Chromosomes[int(sys.argv[1])]:
	maxLength = getChromosomeLength( matchChr )
	marks = range(0,maxLength,100000)
	for strand in ['fwd','rev']:
		for m in marks:

			#set filenames
			bc08dnascent = folderPath+'/bc08_'+matchChr+'_'+strand+'_'+str(m)+'.detect'
			bc12dnascent = folderPath+'/bc12_'+matchChr+'_'+strand+'_'+str(m)+'.detect'

			fout = folderPath+'/bc08_bc12_'+matchChr+'_'+strand+'_'+str(m)+'.merged.detect'
			out = open(fout,'w')

			if not os.path.isfile(bc08dnascent) or not os.path.isfile(bc12dnascent):
				print(bc08dnascent,bc12dnascent)
				continue

			read_sixMers = []
			read_raw = []
			read_modelMeans = []
			read_modelStd = []
			read_positions = []
			read_BrdUcalls = []

			pos_eventMeans = []

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

								out.write('>'+readID + ' ' + chromosome + ' ' + str(mappingStart) + ' ' + str(mappingEnd) + ' ' + strand + '\n')
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
											if matchingReadDic[read_positions[i+5]][4] != '-':
												if float(matchingReadDic[read_positions[i+5]][4]) > llThreshold:
													positiveCall = True


										#for the next 12 bases, pull from the BrdU read
										if noDels and positiveCall:
											BrdUwindow = range(i,i+12)

									if i in BrdUwindow:
										for j in matchingReadDic[read_positions[i]][1]:
											out.write(str(read_positions[i]) + '\t' + read_sixMers[i] + '\t' + str(j) + '\t0\t' + read_sixMers[i] + '\t' + str(read_modelMeans[i]) + '\t' + str(read_modelStd[i]) + '\t' + matchingReadDic[read_positions[i]][4]+'X' + '\n') 

									else:
										for j in read_raw[i]:
											out.write(str(read_positions[i]) + '\t' + read_sixMers[i] + '\t' + str(j) + '\t0\t' + read_sixMers[i] + '\t' + str(read_modelMeans[i]) + '\t' + str(read_modelStd[i]) + '\t' + read_BrdUcalls[i] + '\n')
					
					#reset for read
					read_sixMers = []
					read_raw = []
					read_modelMeans = []
					read_modelStd = []
					read_positions = []
					read_BrdUcalls = []

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

						read_raw.append(pos_eventMeans)
						read_modelMeans.append(modelMean)
						read_modelStd.append(modelStd)
						read_positions.append(prevPos)
						read_BrdUcalls.append(BrdUcall)

						#reset for position
						pos_eventMeans = []
						prevPos = pos

					sixMer = splitLine[4]
					modelMean = float(splitLine[5])
					modelStd = float(splitLine[6])
					eventMean = float(splitLine[2])
					prevReadID = readID
					prevPos = pos

					#sort out BrdU calls
					BrdUcall = '-'
					if len(splitLine) == 8:
						BrdUcall = splitLine[7]
							
					pos_eventMeans.append(eventMean)

			f.close()
			print('Done.')
