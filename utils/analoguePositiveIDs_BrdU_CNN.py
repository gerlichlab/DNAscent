import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math


#0.75 for human EdU
#0.8 for human BrdU
callThreshold = 0.5 #0.75

#0.4 for HMM
#0.3 for human
#0.2 for Pfal
readFractionThreshold = 0.3

f = open(sys.argv[1],'r')
analogueCalls_BrdU = 0
analogueCalls_EdU = 0
numAttempts = 0

g = open('analoguePositiveIDs.txt','w')

maxReads = 50000000
readCount = 0

allFractions_BrdU = []
allFractions_EdU = []
incorporation2readID = {}
for i in range(int(10*readFractionThreshold), 11):
	incorporation2readID[i] = []	

for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':

		readCount += 1
		
		if readCount % 100 == 0:
			print(readCount)

		splitLine = line.rstrip().split()

		if numAttempts != 0:
			fractionsAnalogue_BrdU = float(analogueCalls_BrdU)/float(numAttempts)
			fractionsAnalogue_EdU = float(analogueCalls_EdU)/float(numAttempts)

			allFractions_BrdU.append(fractionsAnalogue_BrdU)
			allFractions_EdU.append(fractionsAnalogue_EdU)

			if fractionsAnalogue_BrdU > readFractionThreshold:
				incorporation2readID[math.floor(10*fractionsAnalogue_BrdU)].append(readID)
				g.write(readID + '\n')

		numAttempts = 0
		analogueCalls_BrdU = 0
		analogueCalls_EdU = 0

		readID = line[1:].rstrip()#splitLine[0][1:]

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

minNum = 10000000
for i in incorporation2readID:
	print(str(i)+'%: ',len(incorporation2readID[i]))
	if len(incorporation2readID[i]) < minNum:
		minNum = len(incorporation2readID[i])
print('Minimum:',minNum)

ctr = 0
g = open('analoguePositiveIDs.txt','w')
for i in incorporation2readID:
	for r in incorporation2readID[i]:
		g.write(r + '\n')
		ctr += 1
		if ctr > maxReads:
			break
g.close()

plt.figure()
plt.hist(allFractions_BrdU,25,alpha=0.3,label='BrdU')
plt.hist(allFractions_EdU,25,alpha=0.3,label='EdU')
plt.xlabel('Positive Calls/Attempts')
plt.ylabel('Count')
plt.yscale("log")
plt.xlim(0,1)
plt.legend(framealpha=0.3)
plt.savefig('callsPerRead_histogram_CNN.pdf')
plt.close()
