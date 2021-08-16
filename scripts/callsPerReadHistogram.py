import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


threshold = 0.50

f = open(sys.argv[1],'r')
BrdUnumCalls = 0
EdUnumCalls = 0
numAttempts = 0

g = open('analoguePositiveIDs.txt','w')

#maxReads = 5000
readCount = 0

fractionsBrdU = []
fractionsEdU = []

for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':

		readCount += 1

		splitLine = line.rstrip().split()
		strand = splitLine[4]

		if numAttempts != 0:
			fractionsBrdU.append(float(BrdUnumCalls)/float(numAttempts))
			fractionsEdU.append(float(EdUnumCalls)/float(numAttempts))

			if fractionsEdU[-1] > 0.2:
				g.write(readID+'\n')

		numAttempts = 0
		BrdUnumCalls = 0
		EdUnumCalls = 0

		readID = splitLine[0][1:]

		continue

	#if readCount > maxReads:
	#	break

	else:

		splitLine = line.split('\t')
		EdUprob = float(splitLine[1])
		BrdUprob = float(splitLine[2])
		position = int(splitLine[0])
		
		if BrdUprob > threshold:
			BrdUnumCalls += 1
		if EdUprob > threshold:
			EdUnumCalls += 1
		numAttempts += 1

f.close()
g.close()

plt.figure()
plt.hist(fractionsBrdU,25,label='BrdU',alpha=0.3)
plt.hist(fractionsEdU,25,label='EdU',alpha=0.3)
plt.yscale("log")
plt.xlim(0,1)
plt.xlabel('Analogue_Calls/Attempts')
plt.ylabel('Number of Reads')
plt.legend(framealpha=0.3)
plt.savefig('callsPerRead.pdf')
plt.close()
