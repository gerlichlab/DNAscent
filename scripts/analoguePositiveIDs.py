import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


threshold = 0.50

f = open(sys.argv[1],'r')
numCalls = 0
numAttempts = 0

readCount = 0
passedReads = 0

for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':

		readCount += 1

		#progress
		if readCount % 500 == 0:
			sys.stderr.write(str(readCount) + " " + str(passedReads) + "\n")

		if numAttempts != 0:
			if float(numCalls)/float(numAttempts) >= 0.2:
				print(readID)
				passedReads += 1

		splitLine = line.rstrip().split()
		readID = splitLine[0][1:]
		strand = splitLine[4]

		numAttempts = 0
		numCalls = 0

		continue

	else:

		splitLine = line.split('\t')
		BrdUprob = float(splitLine[1])
		position = int(splitLine[0])
		
		if BrdUprob > threshold:
			numCalls += 1
		numAttempts += 1

f.close()
