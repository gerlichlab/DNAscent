import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys


files = [sys.argv[3],sys.argv[4]]
#get origins
ignoreIDs = []

for fn in files:
	f = open(fn,'r')
	for line in f:
		splitLine = line.rstrip().split()
		ignoreIDs.append(splitLine[3])
f.close()


lengths = []
files = [sys.argv[1],sys.argv[2]]
for fn in files:
	f = open(fn,'r')
	for line in f:
		splitLine = line.rstrip().split()

		readID = splitLine[3]
		if readID in ignoreIDs:
			continue

		readStart = int(splitLine[5])
		readEnd = int(splitLine[6])

		forkStart = int(splitLine[1])
		forkEnd = int(splitLine[2])


		if forkEnd - forkStart < 1000:
			print(readID)

		if abs(forkStart - readStart) < 2000 or abs(forkStart - readEnd) < 2000:
			continue
		if abs(forkEnd - readStart) < 2000 or abs(forkEnd - readEnd) < 2000:
			continue

		lengths.append(int(splitLine[2]) - int(splitLine[1]))
	f.close()

print('Track length mean:',np.mean(lengths))
print('Track length stdv:',np.std(lengths))

plt.figure()
plt.hist(lengths,25)
plt.ylabel('Count')
plt.xlabel('Fork Track Length (bp)')
plt.savefig('forkTrackLengths.pdf')
plt.close()
