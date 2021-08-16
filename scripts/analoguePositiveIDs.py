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

for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':

		readCount += 1

		splitLine = line.rstrip().split()
		strand = splitLine[4]

		if numAttempts != 0:
			fractionsBrdU = float(BrdUnumCalls)/float(numAttempts)
			fractionsEdU = float(EdUnumCalls)/float(numAttempts)

			if fractionsBrdU > 0.2:
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

