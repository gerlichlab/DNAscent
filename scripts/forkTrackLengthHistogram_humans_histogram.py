import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys


#left forks first
readID2left = {}
f = open(sys.argv[1],'r')
for line in f:
	splitLine = line.rstrip().split()

	readStart = int(splitLine[5])
	readEnd = int(splitLine[6])

	forkStart = int(splitLine[1])
	forkEnd = int(splitLine[2])
	readID = splitLine[3]

	if abs(forkStart - readStart) < 3000 or abs(forkStart - readEnd) < 3000:
		continue

	if abs(forkEnd - readStart) < 3000 or abs(forkEnd - readEnd) < 3000:
		continue

	if readID in readID2left:
		readID2left[readID].append((forkStart,forkEnd))
	else:
		readID2left[readID] = [(forkStart,forkEnd)]

f.close()


#right forks first
readID2right = {}
f = open(sys.argv[2],'r')
for line in f:
	splitLine = line.rstrip().split()

	readStart = int(splitLine[5])
	readEnd = int(splitLine[6])

	forkStart = int(splitLine[1])
	forkEnd = int(splitLine[2])
	readID = splitLine[3]

	if abs(forkStart - readStart) < 3000 or abs(forkStart - readEnd) < 3000:
		continue

	if abs(forkEnd - readStart) < 3000 or abs(forkEnd - readEnd) < 3000:
		continue

	if readID in readID2right:
		readID2right[readID].append((forkStart,forkEnd))
	else:
		readID2right[readID] = [(forkStart,forkEnd)]

f.close()

x_ori = []
y_ori = []

x_term = []
y_term = []

#go through right forks
for readID in readID2right:
		
	for rightFork in readID2right[readID]:

		if readID in readID2left:

			if len(readID2right[readID]) != 1 or len(readID2left[readID]) != 1:
				continue

			if readID2right[readID][0][0] > readID2left[readID][0][1]:
				x_ori.append(readID2right[readID][0][1] - readID2right[readID][0][0])
				y_ori.append(readID2left[readID][0][1] - readID2left[readID][0][0])
			else:
				x_term.append(readID2right[readID][0][1] - readID2right[readID][0][0])
				y_term.append(readID2left[readID][0][1] - readID2left[readID][0][0])
plt.figure()
plt.scatter(x_ori,y_ori,alpha=0.5,label='Origins')
plt.scatter(x_term,y_term,alpha=0.5,label='Terminations')
plt.legend(framealpha=0.3)
plt.ylabel('Leftward-moving Fork Length (bp)')
plt.xlabel('Rightward-moving Fork Length (bp)')
plt.savefig('forkTrackLengths_human_scatter.pdf')
plt.close()
