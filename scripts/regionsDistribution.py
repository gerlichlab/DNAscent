import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys

maxReads = 1000
reads = 0
buf = []

f = open(sys.argv[1],'r')
for line in f:
	if line[0] == '#':
		continue
	elif line[0] == '>':
		reads += 1
		if reads > maxReads:
			break
	else:
		splitLine = line.strip().split()
		buf.append(float(splitLine[2]))
f.close()

plt.figure()
plt.hist(buf,50)
plt.savefig('regionsDistribution.pdf')
plt.close()
