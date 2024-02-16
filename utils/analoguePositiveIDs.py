import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


callThreshold = 1.5#1.
readFractionThreshold = 0.3 #0.1 for EdU, 0.2 for BrdU

f = open(sys.argv[1],'r')
analogueCalls = 0
numAttempts = 0

g = open('analoguePositiveIDs.txt','w')

maxReads = 5000#5000000
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

#print(float(analogueCalls)/numAttempts)

plt.figure()
plt.hist(allFractions,50,alpha=0.3)
plt.xlabel('Positive Calls/Attempts')
plt.ylabel('Number of Reads')
plt.xlim(0,1)
#plt.yscale("log")
#plt.legend(framealpha=0.3)
plt.savefig('callsPerRead_histogram.pdf')
plt.close()
