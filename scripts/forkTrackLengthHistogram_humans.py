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

	#if readEnd - readStart < 30000:
	#	continue

	forkStart = int(splitLine[1])
	forkEnd = int(splitLine[2])
	readID = splitLine[3]

	'''
	if abs(forkStart - readStart) < 1000 or abs(forkStart - readEnd) < 1000:
		continue

	if abs(forkEnd - readStart) < 1000 or abs(forkEnd - readEnd) < 1000:
		continue
	'''
	if forkEnd - forkStart < 1000:
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

	#if readEnd - readStart < 30000:
	#	continue

	forkStart = int(splitLine[1])
	forkEnd = int(splitLine[2])
	readID = splitLine[3]
	
	'''
	if abs(forkStart - readStart) < 1000 or abs(forkStart - readEnd) < 1000:
		continue

	if abs(forkEnd - readStart) < 1000 or abs(forkEnd - readEnd) < 1000:
		continue
	'''
	if forkEnd - forkStart < 1000:
		continue
	
	if readID in readID2right:
		readID2right[readID].append((forkStart,forkEnd))
	else:
		readID2right[readID] = [(forkStart,forkEnd)]

f.close()

distances = []

#go through right forks
for readID in readID2right:
		
	for rightFork in readID2right[readID]:

		#if rightFork[1] - rightFork[0] < 1000:
		#	continue

		if readID in readID2left:
			'''
			merged = False

			for leftFork in readID2left[readID]:

				if abs(rightFork[0] - leftFork[1]) < 5000 and rightFork[0] > leftFork[1]: #if forks match into an origin call

					#left fork is errantly called
					if (rightFork[1] - rightFork[0] > 5000 and leftFork[1] - leftFork[0] < 5000):
						print(readID)
						distances.append( rightFork[1] - leftFork[0] )
						merged = True
						break

			if not merged:
			'''
			distances.append(rightFork[1] - rightFork[0])

		else:
			distances.append(rightFork[1] - rightFork[0])


#go through left forks
for readID in readID2left:
		
	for leftFork in readID2left[readID]:

		#if leftFork[1] - leftFork[0] < 1000:
		#	continue

		if readID in readID2right:

			merged = False

			for rightFork in readID2right[readID]:
				'''

				if abs(rightFork[0] - leftFork[1]) < 5000 and rightFork[0] > leftFork[1]: #if forks match into an origin call

					#right fork is errantly called
					if (rightFork[1] - rightFork[0] < 5000 and leftFork[1] - leftFork[0] > 5000):
						print(readID)
						distances.append( rightFork[1] - leftFork[0] )
						merged = True
						break

			if not merged:
				'''
			distances.append(leftFork[1] - leftFork[0])

		else:
			distances.append(leftFork[1] - leftFork[0])

				

print('Track length mean:',np.mean(distances))
print('Track length stdv:',np.std(distances))

plt.figure()
plt.hist(distances,50)
plt.ylabel('Count')
plt.xlabel('Fork Track Length (bp)')
plt.xlim(0,40000)
plt.savefig('forkTrackLengths.pdf')
plt.close()
