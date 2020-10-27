#----------------------------------------------------------
# Copyright 2017 University of Oxford
# Written by Michael A. Boemo (michael.boemo@path.ox.ac.uk)
# This software is licensed under GPL-2.0.  You should have
# received a copy of the license with this software.  If
# not, please Email the author.
#----------------------------------------------------------


import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from scipy import stats
import math

#outlier detection
from sklearn.cluster import DBSCAN
from sklearn import mixture

#--------------------------------------------------------------------------------------------------------------------------------------
class arguments:
	pass


#--------------------------------------------------------------------------------------------------------------------------------------
def splashHelp():
	s = """plotGMM.py: Plots the results of DNAscent trainGMM against the results of DNAscent align.
To run plotGMM.py, do:
  python plotGMM.py [arguments]
Example:
  python trainFromEventalign.py -m /path/to/thymidine_model -e /path/to/nanopolish_eventalignment -o output_prefix
Required arguments are:
  -g,--gmm                  path to DNAscent trainGMM output file,
  -e,--alignment            path to DNAscent align output file,
  -b,--brdu                 path to DNAscent BrdU model,
  -n,--maxReads             maximum number of reads to import from eventalign."""

	print(s)
	exit(0)


#--------------------------------------------------------------------------------------------------------------------------------------
def parseArguments(args):

	a = arguments()
	a.clipToMax = False
	a.maxReads = 1

	for i, argument in enumerate(args):

		if argument == '-e' or argument == '--alignment':
			a.eventalign = str(args[i+1])

		elif argument == '-g' or argument == '--gmm':
			a.gmm = str(args[i+1])

		elif argument == '-b' or argument == '--brdu':
			a.brdu = str(args[i+1])

		elif argument == '-n' or argument == '--maxReads':
			a.maxReads = int(args[i+1])
			a.clipToMax = True 

		elif argument == '-h' or argument == '--help':
			splashHelp()

	#check that required arguments are met
	if not hasattr( a, 'eventalign') or not hasattr( a, 'gmm') or not hasattr( a, 'brdu') or not hasattr( a, 'maxReads'):
		splashHelp() 

	return a


#--------------------------------------------------------------------------------------------------------------------------------------
def displayProgress(current, total):

	barWidth = 70
	progress = float(current)/float(total)

	if progress <= 1.0:
		sys.stdout.write('[')
		pos = int(barWidth*progress)
		for i in range(barWidth):
			if i < pos:
				sys.stdout.write('=')
			elif i == pos:
				sys.stdout.write('>')
			else:
				sys.stdout.write(' ')
		sys.stdout.write('] '+str(int(progress*100))+' %\r')
		sys.stdout.flush()


#--------------------------------------------------------------------------------------------------------------------------------------
def KLdivergence( mu1, sigma1, mu2, sigma2 ):

	return math.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2.0*sigma2**2) - 0.5


#--------------------------------------------------------------------------------------------------------------------------------------
#parse arguments
args = sys.argv
a = parseArguments(args)

#eventalign output
sixmer2eventsBrdU = {}
f = open(a.eventalign,'r')
currentRead = ''
readCounter = 0
for line in f:

	splitLine = line.rstrip().split('\t')

	#ignore the header line
	if line[0] == '#':
		continue

	if line[0] == '>':
		readCounter += 1
		displayProgress(readCounter, a.maxReads)
		continue

	if a.clipToMax and readCounter == a.maxReads:
		break

	eventTime = float(splitLine[3])
	if eventTime < 0.002:
		continue

	sixmer = splitLine[4]
	if sixmer not in sixmer2eventsBrdU:
		sixmer2eventsBrdU[sixmer] = [float(splitLine[2])]
	elif len(sixmer2eventsBrdU[sixmer]) < 10000:
		sixmer2eventsBrdU[sixmer].append( float(splitLine[2]) )
f.close()

#GMM model
fit1 = {}
fit2 = {}
model = {}
f = open(a.gmm,'r')
for line in f:

	if line[0] == '#' or line[0] == '6':
		continue
	
	splitLine = line.rstrip().split('\t')
	model[splitLine[0]] = [	float(splitLine[1]), float(splitLine[2]) ]
	fit1[splitLine[0]] = [	float(splitLine[4]), float(splitLine[5]) ]
	fit2[splitLine[0]] = [	float(splitLine[7]), float(splitLine[8]) ]
f.close()

#BrdU model
brdu = {}
f = open(a.brdu,'r')
for line in f:

	if line[0] == '#' or line[0] == 'k':
		continue
	
	splitLine = line.rstrip().split('\t')
	brdu[splitLine[0]] = [	float(splitLine[1]), float(splitLine[2]) ]
f.close()

for i, key in enumerate(sixmer2eventsBrdU):

	if key == 'NNNNNN':
		continue
	
	if len( sixmer2eventsBrdU[key] ) > 200:
		x = np.linspace( np.mean(sixmer2eventsBrdU[key])-15, np.mean(sixmer2eventsBrdU[key])+15, 1000 )

		#noise reduction
		ar = np.array(sixmer2eventsBrdU[key])
		db = DBSCAN( min_samples= (0.025*len( sixmer2eventsBrdU[key] )) ).fit(ar.reshape(-1,1))
		outliers_filtered = []
		for j, label in enumerate(db.labels_):
			if label == -1:
				continue
			else:
				outliers_filtered.append(sixmer2eventsBrdU[key][j])

		#plotting
		plt.figure()
		plt.hist(outliers_filtered, 50, density=True, alpha=0.3)

		model_mu = model[key][0]
		model_std = model[key][1]
		x_model = np.linspace(model_mu - 3*model_std, model_mu + 3*model_std, 100)
		plt.plot(x_model, stats.norm.pdf(x_model, model_mu, model_std), label='Pore Model')

		if key in fit1:
			model_mu = fit1[key][0]
			model_std = fit1[key][1]
			x_model = np.linspace(model_mu - 3*model_std, model_mu + 3*model_std, 100)
			plt.plot(x_model, stats.norm.pdf(x_model, model_mu, model_std), label='Fit 1')

		if key in fit2:
			model_mu = fit2[key][0]
			model_std = fit2[key][1]
			x_model = np.linspace(model_mu - 3*model_std, model_mu + 3*model_std, 100)
			plt.plot(x_model, stats.norm.pdf(x_model, model_mu, model_std), label='Fit 2')

		if key in brdu:
			model_mu = brdu[key][0]
			model_std = brdu[key][1]
			x_model = np.linspace(model_mu - 3*model_std, model_mu + 3*model_std, 100)
			plt.plot(x_model, stats.norm.pdf(x_model, model_mu, model_std), label='BrdU')

		plt.title(key)
		plt.legend(framealpha=0.3)
		plt.xlabel('Event Magnitude (pA)')
		plt.title(key)
		plt.savefig(key + '.png')
		plt.close()



