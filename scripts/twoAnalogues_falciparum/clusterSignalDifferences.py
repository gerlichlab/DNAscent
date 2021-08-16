import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys

differences = []
f = open(sys.argv[1],'r')
eventCtr = 0
deltaRunning = 0.
readCtr = 0
for line in f:
	if line[0] =='>':
		if eventCtr != 0:
			differences.append(deltaRunning/eventCtr)

		eventCtr = 0
		deltaRunning = 0.
		readCtr += 1

		if readCtr % 100 == 0:
			print(readCtr)

		if readCtr > 20000:
			break

	else:
		splitLine = line.rstrip().split()
		eventSignal = float(splitLine[2])
		eventModel = float(splitLine[5])
		sixMer = splitLine[4]
		if sixMer.count('T') != 0:
			eventCtr +=1
			deltaRunning += abs(eventSignal - eventModel)

plt.figure()
plt.hist(differences,50)
plt.xlabel('Mean Event Differences From Model (pA)')
plt.ylabel('Count')
plt.savefig('differencesFromModelHist.pdf')
plt.close()			

			
		
			
