import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

f_poreModel = '/home/mb915/rds/hpc-work/development/DNAscent_R10align/DNAscent_dev/pore_models/r10.4.1_unlabelled_cauchy.model'
f_fitModel = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode23/mixturePlots/R10model_EdU.2000events_50kreads.cauchy.fromCauchyAlignment.model'

#parse fit model
fitModel = {}
runningStdv = 0.
count = 0
f = open(f_fitModel,'r')
for line in f:
	splitLine = line.rstrip().split()
	if splitLine[0] == "6mer":
		continue
	fitModel[splitLine[0]] = (float(splitLine[1]),float(splitLine[2]))
	count += 1
	runningStdv += float(splitLine[2])
f.close()

print('Fit model parsed')

avgStdv = runningStdv/count

#parse pore model
poreModel = {}
f = open(f_poreModel,'r')
for line in f:
	splitLine = line.rstrip().split()
	poreModel[splitLine[0]] = float(splitLine[1])
f.close()

f = open('r10.4.1_EdU_cauchy.model','w')

print('Pore model parsed')

meanDifferences = []
allStdv = []
#build the final pore model	
for i,kmer in enumerate(poreModel):
	if i % 1000 == 0:
		print('Written ',str(i),' kmers')
	if kmer in fitModel and 'T' in kmer:
	
		f.write(kmer + '\t' + str(fitModel[kmer][0]) + '\t' + str(fitModel[kmer][1]) + '\n')
		meanDifferences.append(fitModel[kmer][0] - poreModel[kmer])
		allStdv.append(fitModel[kmer][1])
	elif 'T' in kmer:
		f.write(kmer + '\t' + str(poreModel[kmer]) + '\t' + str(avgStdv) + '\n')
f.close()

plt.figure()
plt.hist(meanDifferences,30)
plt.xlabel('Fit Model - Pore Model')
plt.ylabel('Number of Kmers')
plt.savefig('meanDifferences.pdf')
plt.close()

plt.figure()
plt.hist(allStdv,30)
plt.axvline(x = np.mean(allStdv), color = 'r')
plt.xlabel('Fit Standard Deviations')
plt.ylabel('Number of Kmers')
plt.savefig('allStdv.pdf')
plt.close()
