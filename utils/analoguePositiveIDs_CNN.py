import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#0.4 for HMM
#0.3 for human
#0.3 for Pfal

callThreshold = 0.5
readFractionThreshold = 0.3

f = open(sys.argv[1],'r')
analogueCalls_BrdU = 0
analogueCalls_EdU = 0
numAttempts = 0

g = open('analoguePositiveIDs.txt','w')

maxReads = 5000
readCount = 0

allFractions_BrdU = []
allFractions_EdU = []

for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':

		readCount += 1

		splitLine = line.rstrip().split()

		if numAttempts != 0:
			fractionsAnalogue_BrdU = float(analogueCalls_BrdU)/float(numAttempts)
			fractionsAnalogue_EdU = float(analogueCalls_EdU)/float(numAttempts)

			allFractions_BrdU.append(fractionsAnalogue_BrdU)
			allFractions_EdU.append(fractionsAnalogue_EdU)

			if fractionsAnalogue_EdU > readFractionThreshold:
				g.write(readID+'\n')

		numAttempts = 0
		analogueCalls_BrdU = 0
		analogueCalls_EdU = 0

		readID = splitLine[0][1:]

		continue

	if readCount > maxReads:
		break

	else:

		splitLine = line.split('\t')
		EdUprob = float(splitLine[1])
		BrdUprob = float(splitLine[2])
				
		if BrdUprob > callThreshold:
			analogueCalls_BrdU += 1
		if EdUprob > callThreshold:
			analogueCalls_EdU += 1			
		numAttempts += 1

f.close()
g.close()

plt.figure()
plt.hist(allFractions_BrdU,25,alpha=0.3,label='BrdU')
plt.hist(allFractions_EdU,25,alpha=0.3,label='EdU')
plt.xlabel('Positive Calls/Attempts')
plt.ylabel('Count')
plt.yscale("log")
plt.xlim(0,1)
plt.legend(framealpha=0.3)
plt.savefig('callsPerRead_histogram.pdf')
plt.close()
