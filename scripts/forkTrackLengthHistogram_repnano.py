import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys

distances = []
f = open(sys.argv[1],'r')
for line in f:
	splitLine = line.rstrip().split()

	lb = int(splitLine[1]) 
	ub = int(splitLine[2])	

	'''
	if abs(forkStart - readStart) < 1000 or abs(forkStart - readEnd) < 1000:
		continue

	if abs(forkEnd - readStart) < 1000 or abs(forkEnd - readEnd) < 1000:
		continue
	'''
	#if forkEnd - forkStart < 1000:
	#	continue
	
	distances.append(abs(ub-lb))

f.close()

print('Track length mean:',np.mean(distances))
print('Track length stdv:',np.std(distances))

plt.figure()
plt.hist(distances,50)
plt.ylabel('Count')
plt.xlabel('Fork Track Length (bp)')
plt.xlim(0,40000)
plt.savefig('forkTrackLengths.pdf')
plt.close()
