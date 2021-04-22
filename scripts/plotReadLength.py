import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

f = open(sys.argv[1],'r')
lengths = []

for line in f:
	if line[0] == '>':
		splitLine = line.rstrip().split()
		lengths.append((int(splitLine[3]) - int(splitLine[2]))/1000.)
f.close()

plt.figure()
plt.hist(lengths, 50)
plt.xlabel('Read Length (kb)')
plt.ylabel('Count')
plt.savefig('readLengthHistogram.pdf')
plt.close('all')
