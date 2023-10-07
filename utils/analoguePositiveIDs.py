import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


callThreshold = 0. #2.5
readFractionThreshold = 0.4

f = open(sys.argv[1],'r')
analogueCalls = 0
numAttempts = 0

g = open('analoguePositiveIDs.txt','w')

maxReads = 5000
readCount = 0

allFractions = []

for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':

		readCount += 1

		splitLine = line.rstrip().split()

		if numAttempts != 0:
			fractionsAnalogue = float(analogueCalls)/float(numAttempts)

			allFractions.append(fractionsAnalogue)

			if fractionsAnalogue > readFractionThreshold:
				g.write(readID+'\n')

		numAttempts = 0
		analogueCalls = 0

		readID = splitLine[0][1:]

		continue

	if readCount > maxReads:
		break

	else:

		splitLine = line.split('\t')
		analogueLL = float(splitLine[1])
		
		if analogueLL > callThreshold:
			analogueCalls += 1
		numAttempts += 1

f.close()
g.close()

plt.figure()
plt.hist(allFractions,50,alpha=0.3)
plt.xlabel('Positive Calls/Attempts')
plt.ylabel('Count')
#plt.yscale("log")
#plt.legend(framealpha=0.3)
plt.savefig('callsPerRead_histogram.pdf')
plt.close()
