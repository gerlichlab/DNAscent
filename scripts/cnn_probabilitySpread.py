import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys

maxReads = 1000


Tprefix = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/'
Bprefix = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode11/'

fn_Thym = [Tprefix+'commit2d622cc1_DNAscentCNN_hardLabel_highIns_rightTestData_8Cores.detect', Tprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build23.detect']
fn_BrdU = [Bprefix+'commit2d622cc1_DNAscentCNN_hardLabel_highIns_rightTestData_8Cores.detect', Bprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build23.detect']
labels = ['highIns;hardLabel', 'build23']

#fn_BrdU = [Bprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels_epoch1.detect', Bprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels.detect', Bprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels_epoch19.detect']
#fn_Thym = [Tprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels_epoch1.detect', Tprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels.detect', Tprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels_epoch19.detect']
#labels = ['lowIns;softLabel - 1epochs', 'lowIns;softLabel - 5epochs', 'lowIns;softLabel - 19epochs' ]

thymidineProbabilities = []
BrdUProbabilities = []
fig, ax = plt.subplots()

for i in range(0,len(fn_BrdU)):

	thymidineProbabilities = []
	BrdUProbabilities = []

	print(fn_BrdU[i])

	#thymidine
	f = open(fn_Thym[i],'r')
	readCtr = 0
	for line in f:

		if line[0] == '>':
			splitLine = line.rstrip().split()
			strand = splitLine[4]

			readCtr += 1
			if readCtr > maxReads:
				break

			continue
		else:
			splitLine = line.split('\t')
			sixMer = splitLine[3]

			if (strand == 'rev' and not sixMer[-1:] == "A") or (strand == "fwd" and not sixMer[0] == "T"):
				continue

			BrdUprob = float(splitLine[2])
			thymidineProbabilities.append(BrdUprob)

	f.close()

	#brdu
	f = open(fn_BrdU[i],'r')
	readCtr = 0
	for line in f:

		if line[0] == '>':
			splitLine = line.rstrip().split()
			strand = splitLine[4]

			readCtr += 1
			if readCtr > maxReads:
				break

			continue
		else:
			splitLine = line.split('\t')
			sixMer = splitLine[3]

			if (strand == 'rev' and not sixMer[-1:] == "A") or (strand == "fwd" and not sixMer[0] == "T"):
				continue


			BrdUprob = float(splitLine[2])
			BrdUProbabilities.append(BrdUprob)
	f.close()

	#plot on the figure
	plt.hist(BrdUProbabilities, 50, density=True, alpha=0.3, label = labels[i] + '_BrdU')
	plt.hist(thymidineProbabilities, 50, density=True, alpha=0.3,  label = labels[i] + '_Thym')

plt.legend()
plt.xlim(0,1)
plt.xlabel('Thymidine Position Probability')
plt.ylabel('Density')
plt.savefig('probabilityHist.pdf')
plt.close()







