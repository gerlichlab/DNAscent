import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

target = '3'
length = 1067971
BrdUCalls = [0]*length
coverage = [0]*length
BrdUCalls = np.array(BrdUCalls)
coverage = np.array(coverage)

#import detect data
f = open(sys.argv[1],'r')
for line in f:

	if line[0] == '>':

		splitLine = line.rstrip().split(' ')
		chromosome = splitLine[1]

		if int(chromosome) > int(target):
			break

		continue

	else:

		splitLine = line.rstrip().split('\t')

		if chromosome == target:

			coverage[int(splitLine[0])] += 1
			if float(splitLine[1]) > 2.5:

				BrdUCalls[int(splitLine[0])] += 1



xBrdU = []
yBrdU = []
for i in range( 0, length, 100 ):

	if float(sum( coverage[i:i+100])) == 0.0:
		continue
	else:
		yBrdU.append(float(sum( BrdUCalls[i:i+100] )) / float(sum( coverage[i:i+100])))
		xBrdU.append(i+50)

yBrdUSmooth = np.convolve(yBrdU, np.ones((10,))/10, mode='same')

plt.figure(1)
plt.plot(xBrdU, yBrdUSmooth, alpha=0.5)
plt.xlim(0,length)
#plt.ylim(0,6)
plt.legend(framealpha=0.5)
plt.xlabel('Position on Chromosome (bp)')
plt.ylabel('BrdU Calls / Coverage')
plt.title('BrdU IP')
plt.savefig(target + '_nanoporeIP.pdf')
