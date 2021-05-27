import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


threshold = 0.25

f = open(sys.argv[1],'r')
numCalls = 0
numAttempts = 0

maxReads = 5000
readCount = 0

fractions = []

for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':

		readCount += 1

		splitLine = line.rstrip().split()
		strand = splitLine[4]

		if numAttempts != 0:
			fractions.append(float(numCalls)/float(numAttempts))

		numAttempts = 0
		numCalls = 0

		continue

	if readCount > maxReads:
		break

	else:

		splitLine = line.split('\t')
		BrdUprob = float(splitLine[1])
		position = int(splitLine[0])
		
		if BrdUprob > threshold:
			numCalls += 1
		numAttempts += 1

f.close()

plt.figure()
plt.hist(fractions,25)
plt.yscale("log")
plt.xlim(0,1)
plt.xlabel('Analogue_Calls/Attempts')
plt.ylabel('Number of Reads')
plt.savefig('callsPerRead.pdf')
plt.close()
