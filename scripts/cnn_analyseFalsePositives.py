import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import itertools
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys
import os
import pickle
from scipy.stats import halfnorm

scanPositions1 = 30000
scanPositions2 = 10000
threshold = 0.7

#-------------------------------------------------------------------------------------------------
#find positive call positions
fp_readID2pos = {}
tn_readID2pos = {}
positivesFound = 0
f = open(sys.argv[1],'r')
for line in f:
	if line[0] == '#':
		continue
	if line[0] == '>':
		splitLine = line.rstrip().split()
		readID = splitLine[0][1:]
		strand = splitLine[4]
	else:
		splitLine = line.rstrip().split('\t')
		sixMer = splitLine[3]

		#ignore non-T positions
		if ((strand == "fwd" and sixMer[0] != "T") or (strand == "rev" and sixMer[-1:] != "A")):
			continue

		pos = int(splitLine[0])
		BrdUprob = float(splitLine[2])

		if BrdUprob > threshold:
			print(positivesFound)
			positivesFound += 1
			if readID in fp_readID2pos:
				fp_readID2pos[readID].append(pos)
			else:
				fp_readID2pos[readID] = [pos]

		if BrdUprob < 1 - threshold:
			if readID in tn_readID2pos:
				tn_readID2pos[readID].append(pos)
			else:
				tn_readID2pos[readID] = [pos]
		
		if positivesFound > scanPositions1:
			break


#-------------------------------------------------------------------------------------------------
#find what's special about them
fp_read_sixMers = []
fp_read_eventMeans = []
fp_read_eventStd = []
fp_read_stutter = []
fp_read_lengthMeans = []
fp_read_lengthStd = []
fp_read_modelMeans = []
fp_read_modelStd = []
fp_read_positions = []
fp_read_BrdUcalls = []

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
positivesFound = 0
f = open(sys.argv[2],'r')
for line in f:

	if line[0] == '>':

		#reset for position
		pos_eventMeans = []
		pos_lengths = []
		prevPos = -1

		splitLine = line.rstrip().split()
		readID = splitLine[0][1:]
		mappingStart = int(splitLine[2])
		mappingEnd = int(splitLine[3])
		prevPos = -1

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

			record = False
			if readID in fp_readID2pos:
				if prevPos in fp_readID2pos[readID]:
					record = True

			if record:
				positivesFound += 1
				print(positivesFound)
				fp_read_sixMers.append(sixMer)
				fp_read_eventMeans.append(modelMean - np.mean(pos_eventMeans))
				fp_read_eventStd.append(np.std(pos_eventMeans))
				fp_read_stutter.append(min(50,len(pos_eventMeans)))
				fp_read_lengthMeans.append(np.mean(pos_lengths))
				fp_read_lengthStd.append(np.std(pos_lengths))
				fp_read_modelMeans.append(modelMean)
				fp_read_modelStd.append(modelStd)
				fp_read_positions.append(prevPos)
				fp_read_BrdUcalls.append(BrdUcall)

			record = False
			if readID in tn_readID2pos:
				if prevPos in tn_readID2pos[readID]:
					record = True

			if record:
				read_sixMers.append(sixMer)
				read_eventMeans.append(modelMean-np.mean(pos_eventMeans))
				read_eventStd.append(np.std(pos_eventMeans))
				read_stutter.append(min(50,len(pos_eventMeans)))
				read_lengthMeans.append(np.mean(pos_lengths))
				read_lengthStd.append(np.std(pos_lengths))
				read_modelMeans.append(modelMean)
				read_modelStd.append(modelStd)
				read_positions.append(prevPos)
				read_BrdUcalls.append(BrdUcall)

			#reset for position
			pos_eventMeans = []
			pos_lengths = []
			prevPos = pos

		if positivesFound > scanPositions2:
			break


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


#-------------------------------------------------
#do the plotting
plt.figure()
plt.hist(fp_read_eventMeans,50, alpha = 0.3, density = True, label='false positives')
plt.hist(read_eventMeans,50, alpha = 0.3, density = True, label='true negatives')
plt.legend()
plt.xlabel('Event Mean')
plt.savefig('eventMeans.pdf')
plt.close()

plt.figure()
plt.hist(fp_read_eventStd,50, alpha = 0.3, density = True, label='false positives')
plt.hist(read_eventStd,50, alpha = 0.3, density = True, label='true negatives')
plt.legend()
plt.xlabel('Event Stdv')
plt.savefig('eventStd.pdf')
plt.close()

plt.figure()
plt.hist(fp_read_stutter,50, alpha = 0.3, density = True, label='false positives')
plt.hist(read_stutter,50, alpha = 0.3, density = True, label='true negatives')
plt.xlim(0,25)
plt.legend()
plt.xlabel('Event Stutter')
plt.savefig('eventStutter.pdf')
plt.close()

plt.figure()
plt.hist(fp_read_lengthMeans,50, alpha = 0.3, density = True, label='false positives')
plt.hist(read_lengthMeans,50, alpha = 0.3, density = True, label='true negatives')
plt.legend()
plt.xlabel('Event Length Mean')
plt.savefig('eventLengthMean.pdf')
plt.close()

plt.figure()
plt.hist(fp_read_lengthStd,50, alpha = 0.3, density = True, label='false positives')
plt.hist(read_lengthStd,50, alpha = 0.3, density = True, label='true negatives')
plt.legend()
plt.xlabel('Event Length Stdv')
plt.savefig('eventLengthStd.pdf')
plt.close()

