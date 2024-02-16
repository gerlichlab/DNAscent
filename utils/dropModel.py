import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from itertools import product


stdv = 0.14

kmer2signal = {}
f = open('/home/mb915/rds/hpc-work/development/DNAscent_R10align/DNAscent_dev/pore_models/r10.4.1_400bps.nucleotide.9mer.model','r')
for line in f:
	splitLine = line.rstrip().split()
	kmer2signal[splitLine[0]] = float(splitLine[1])
f.close()

drop2signal = {}
kmer2residual = {}

kmers_5 = [''.join(c) for c in product('ATGC', repeat=5)]
kmers_4 = [''.join(c) for c in product('ATGC', repeat=4)]

X = []
Y = []

for kmer5 in kmers_5:
	print(kmer5)
	signals = []
	for kmer4 in kmers_4:
		kmer = kmer4[0:2]+kmer5+kmer4[2:4]
		signals += list(np.random.normal(kmer2signal[kmer], stdv, 100))
	mu = np.mean(signals)
	med = np.median(signals)
	std = np.std(signals)
	drop2signal[kmer5] = (mu,std)
	
	Y.append(med)
	X.append(kmer5)
	
	for kmer4 in kmers_4:
		if kmer4 not in kmer2residual:
			kmer2residual[kmer4] = []
		kmer = kmer4[0:2]+kmer5+kmer4[2:4]
		kmer2residual[kmer4].append(mu - kmer2signal[kmer])
	'''
	plt.figure()
	plt.hist(signals,20)
	plt.xlabel('5mer Signal')
	plt.ylabel('Count')
	plt.savefig('dropModel_plots_core/'+kmer5+'.png')
	plt.close()
	'''
#write the sorted output
f_5mers = open('core_5mers_sorted.txt','w')
xy = list(zip(Y,X))
xy.sort()
for x in xy:
	f_5mers.write(x[1] + ' ' + str(x[0]) + '\n')
f_5mers.close()


X = []
Y = []	
print('Plotting...')
for kmer4 in kmer2residual:
	'''
	plt.figure()
	plt.hist(kmer2residual[kmer4],20)
	plt.xlabel('5mer Signal - 9mer Signal')
	plt.ylabel('Count')
	plt.savefig('dropModel_plots_residual/'+kmer4+'.png')
	plt.close()
	'''
	med = np.median(kmer2residual[kmer4])
	
	Y.append(med)
	X.append(kmer4)
	
f_4mers = open('residual_4mers_sorted.txt','w')
xy = list(zip(Y,X))
xy.sort()
for x in xy:
	f_4mers.write(x[1] + ' ' + str(x[0]) + '\n')
f_4mers.close()
