import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import itertools

kmer2count = {}
kmer2calls = {}
bases=['A','T','G','C']
kmers = [''.join(p) for p in itertools.product(bases, repeat=9)]
for i in kmers:
	kmer2count[i] = 0
	if i[0] == 'T':
		kmer2calls[i] = 0

log_threshold = 0.5 #2.5

readCount = 0
currentPos = -1
f = open(sys.argv[1],'r')
for line in f:
	
	if line[0] == '>':
		readCount += 1

		if readCount % 100 == 0:
			print('Read count:', readCount)

	#if readCount > 2000:
	#	break

	if line[0] in ['>','#']:
		continue
	
	splitLine = line.rstrip().split()
	
	pos = int(splitLine[0])
	if pos == currentPos:
		continue
	else:
		currentPos = pos
	
	kmer = splitLine[3]
	if kmer == 'NNNNNNNNN':
		continue
	
	kmer2count[kmer] += 1

	if len(splitLine) >= 6:
		call = float(splitLine[5])
		if call > log_threshold:
			kmer2calls[kmer] += 1
f.close()

plt.figure()
plt.hist( list(kmer2count.values()), bins = range(0,50,2) )
plt.xlabel('9mer Count')
plt.ylabel('Frequency')
plt.xlim(0,50)
plt.yscale("log")
plt.savefig('kmerCountHist.pdf')
plt.close()

plt.figure()
plt.hist( list(kmer2calls.values()), bins = range(0,50,2) )
plt.xlabel('9mer Calls')
plt.ylabel('Frequency')
plt.xlim(0,50)
plt.yscale("log")
plt.savefig('kmerCallsHist.pdf')
plt.close()
