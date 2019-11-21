import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys

#Usage: python callsAgainstKL.py stderr.out
#where stderr.out is the result of a stderr redirect after running DNAscent detect with the --testAlignment flag

maxReads = 20

x=[]
y=[]
progress = 0


f = open(sys.argv[1],'r')
for line in f:
	if line[0] == '>':

		progress += 1
		if progress >= maxReads:
			break


		if len(x) > 0:
			plt.figure()
			plt.plot(x,y)
			plt.xlabel('Event Index')
			plt.ylabel('kmer Index')
			plt.savefig(readID+'.png')
			plt.close()
		readID = line.rstrip()[1:]
		x=[]
		y=[]
	else:
		[eventIdx,kmerIdx] = line.rstrip().split()
		x.append(int(eventIdx))
		y.append(int(kmerIdx))
f.close()
