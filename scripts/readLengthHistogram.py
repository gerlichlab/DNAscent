import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

f = open(sys.argv[1],'r')

lengths = []

for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':
		splitLine = line.rstrip().split()
		lengths.append(int(splitLine[3]) - int(splitLine[2]))
f.close()

print('Mean length:',np.mean(lengths))

plt.figure()
plt.hist(np.array(lengths)/1000,50)
plt.yscale("log")
plt.xlabel('Length (Kb)')
plt.ylabel('Count')
plt.savefig('readLengthHistogram.pdf')
plt.close()
