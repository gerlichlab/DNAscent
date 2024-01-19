import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from itertools import product
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
import itertools


f_thym = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode22/DNAscent_R10.theilSen.align'
f_analogue = '/home/mb915/rds/rds-boemo_3-tyMgmffheQw/2023_07_05_MJ_ONT_RPE1_Edu_BrdU_trainingdata_V14_5khz/barcode24/DNAscent_R10.theilSen.align'

maxEvents = 2000
maxReads = 50000


#----------------------------------------------------
#parse the pore model (unlabelled)

f_poreModel = '/home/mb915/rds/hpc-work/development/DNAscent_R10align/DNAscent_dev/pore_models/r10.4.1_400bps.nucleotide.9mer.model' #'r10.4.1_400bps.nucleotide.9mer.model'
poreModel = {}
print('Importing pore model')
f = open(f_poreModel,'r')
for line in f:
	if line[0] == '#':
		continue

	splitLine = line.rstrip().split()
	poreModel[splitLine[0]] = (float(splitLine[1]),0.08)
f.close()
print('Pore model imported')


#----------------------------------------------------
#parse the pore model (analogue)
'''
f_analogueModel = '/home/mb915/rds/hpc-work/development/DNAscent_R10align/DNAscent_dev/pore_models/r10.4.1_BrdU_cauchy.model'
analogueModel = {}
print('Importing pore model')
f = open(f_analogueModel,'r')
for line in f:
	if line[0] == '#':
		continue

	splitLine = line.rstrip().split()
	analogueModel[splitLine[0]] = (float(splitLine[1]),float(splitLine[2]))
f.close()
print('Pore model imported')
'''

#----------------------------------------------------
#
def getKmerToEvent(fn):

	print('Parsing ' + fn)

	kmerToSig = {'':[]}
	allKmers = [''.join(c) for c in product('ATCG', repeat=9)]
	for kmer in allKmers:
		kmerToSig[kmer] = []
		
	nreads = 0

	f = open(fn,'r')
	for line in f:
		if line[0] == '#':
			continue
		
		if line[0] == '>':
			
			nreads += 1

			if nreads > maxReads:
				break
				
			if nreads % 100 == 0:
				print('Imported ',nreads)
		
			continue
			
		splitLine = line.rstrip().split()
		
		pos = int(splitLine[0])
		
		kmer = splitLine[3]
		if 'N' in kmer:
			continue
			
		sig = float(splitLine[2])
		
		if len(kmerToSig[kmer]) < maxEvents:
			kmerToSig[kmer].append(sig)
			
	f.close()
	
	return kmerToSig


#----------------------------------------------------
#main	
	
thymDict = getKmerToEvent(f_thym)
print('Thymidine imported')
analogueDict = getKmerToEvent(f_analogue)
print('Analogue imported')

for kmer in thymDict:

	#skip kmers with low numbers of aligned signals
	if len(thymDict[kmer]) < 200:
		continue
	
	if kmer in analogueDict:

		#skip kmers with low numbers of aligned signals	
		if len(analogueDict[kmer]) < 200:
			continue
	
		#print(kmer)	
		#plot signal histograms
		#plt.figure()
		#plt.hist(thymDict[kmer],20,alpha=0.3,density=True,label='Thym Events')
		#if kmer in analogueDict:
		#	plt.hist(analogueDict[kmer],20,alpha=0.3,density=True,label='BrdU Events')
		
		#plot unlabelled pore model
		model_mu = poreModel[kmer][0]
		model_std = poreModel[kmer][1]
		x_model = np.linspace(model_mu - 5*model_std, model_mu + 5*model_std, 100)
		#plt.plot(x_model, stats.norm.pdf(x_model, model_mu, model_std), label='Pore Model')

		#fit normal distribution to unlabelled data
		fit_thym_loc, fit_thym_scale = stats.norm.fit(thymDict[kmer])
		#plt.plot(x_model, stats.norm.pdf(x_model, fit_thym_loc, fit_thym_scale), label='Unlabelled Fit')		
		
		#fit normal distribution to unlabelled data
		fit_analogue_loc, fit_analogue_scale = stats.norm.fit(analogueDict[kmer])
		#plt.plot(x_model, stats.norm.pdf(x_model, fit_analogue_loc, fit_analogue_scale), label='Analogue Fit')

		#plot analogue Cauchy pore model		
		#if kmer in analogueModel:
		#	model_mu = analogueModel[kmer][0]
		#	model_std = analogueModel[kmer][1]
		#	x_model = np.linspace(model_mu - 5*model_std, model_mu + 5*model_std, 100)
		#	plt.plot(x_model, stats.cauchy.pdf(x_model, model_mu, model_std), label='Analogue Model')	
		
		#fit Cauchy to unlabelled signals
		#fit_thym_loc, fit_thym_scale = stats.cauchy.fit(thymDict[kmer])
		#x_model = np.linspace(fit_thym_loc - 5*fit_thym_scale, fit_thym_loc + 5*fit_thym_scale, 100)
		#plt.plot(x_model, stats.cauchy.pdf(x_model, fit_thym_loc, fit_thym_scale), label='Thym Cauchy Fit')		
		
		#fit skewed Cauchy to analogue signals, then keep Cauchy location and scale parameters
		#ae, fit_analogue_loc, fit_analogue_scale = stats.skewcauchy.fit(analogueDict[kmer])
		#x_model = np.linspace(fit_analogue_loc - 5*fit_analogue_scale, fit_analogue_loc + 5*fit_analogue_scale, 100)
		#plt.plot(x_model, stats.cauchy.pdf(x_model, fit_analogue_loc, fit_analogue_scale), label='Analogue Cauchy Fit')		
		
		#plot for this kmer
		#plt.title(kmer)
		#plt.xlabel('Signal (Normalised)')
		#plt.ylabel('Density')		
		#plt.legend(framealpha=0.3)		
		#plt.savefig(kmer+'.png')
		#plt.close()
		
		#write the fit for unlabelled
		f_thym = open('R10model_Thym.2000events_50kreads.model','a')		
		strThym = kmer + '\t' + str(fit_thym_loc) + '\t' + str(fit_thym_scale) + '\t' + str(len(thymDict[kmer])) + '\n'
		f_thym.write(strThym)
		f_thym.close()

		#write the fit for analogue
		f_analogue = open('R10model_analogue.model','a')
		strAnalogue = kmer + '\t' + str(fit_analogue_loc) + '\t' + str(fit_analogue_scale) + '\t' + str(len(analogueDict[kmer])) + '\n'
		f_analogue.write(strAnalogue)
		f_analogue.close()

