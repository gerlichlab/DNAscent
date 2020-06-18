import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys

maxReads = 5000


Tprefix = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/'
Bprefix = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode11/'
fn_Thym = [Tprefix+'commit2d622cc1_DNAscentCNN_hardLabel_highIns_rightTestData_8Cores.detect', Tprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_rightTestData_8Cores.detect', Tprefix+'commit2d622cc1_DNAscentCNN_hardLabel_lowIns.detect', Tprefix+'commit2d622cc1_DNAscentCNN_softLabel_lowIns.detect', Tprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build23.detect', Tprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build24.detect', Tprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build25.detect', Tprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build26.detect', Tprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build27.detect']
fn_BrdU = [Bprefix+'commit2d622cc1_DNAscentCNN_hardLabel_highIns_rightTestData_8Cores.detect', Bprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_rightTestData_8Cores.detect', Bprefix+'commit2d622cc1_DNAscentCNN_hardLabel_lowIns.detect', Bprefix+'commit2d622cc1_DNAscentCNN_softLabel_lowIns.detect', Bprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build23.detect', Bprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build24.detect', Bprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build25.detect', Bprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build26.detect', Bprefix+'commit2d622cc1_DNAscentCNN_softLabel_highIns_build27.detect']
labels = ['highIns;hardLabel', 'highIns;softLabel', 'lowIns;hardLabel', 'lowIns;softLabel', 'build23', 'build24', 'build25', 'build26', 'build27']

#fn_BrdU = [Bprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels_epoch1.detect', Bprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels.detect', Bprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels_epoch19.detect']
#fn_Thym = [Tprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels_epoch1.detect', Tprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels.detect', Tprefix+'commit2d622cc1_DNAscentCNN_lowIns_softLabels_epoch19.detect']
#labels = ['lowIns;softLabel - 1epochs', 'lowIns;softLabel - 5epochs', 'lowIns;softLabel - 19epochs' ]

probTests = np.array(range(1,10))/10.

thym_calls = np.zeros(len(probTests))
thym_attempts = np.zeros(len(probTests))
brdu_calls = np.zeros(len(probTests))
brdu_attempts = np.zeros(len(probTests))

fig, ax = plt.subplots()

for i in range(0,len(fn_BrdU)):

	thym_calls = np.zeros(len(probTests))
	thym_attempts = np.zeros(len(probTests))
	brdu_calls = np.zeros(len(probTests))
	brdu_attempts = np.zeros(len(probTests))

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

			for j in range(0,len(probTests)):
				if BrdUprob > probTests[j]:
					thym_calls[j] += 1.
				thym_attempts[j] += 1.
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

			for j in range(0,len(probTests)):
				if BrdUprob > probTests[j]:
					brdu_calls[j] += 1
				brdu_attempts[j] += 1
	f.close()

	#plot on the figure
	x = thym_calls/thym_attempts
	y = brdu_calls/brdu_attempts
	plt.plot(x[::-1], y[::-1], label=labels[i])
	for p,txt in enumerate(probTests):
		ax.annotate(str(txt),(x[p],y[p]),fontsize=6)

#CNN END



#DO FOR HMM
fn_BrdU = [Bprefix+'commit2d622cc1_HMM.detect']
fn_Thym = [Tprefix+'commit2d622cc1_HMM.detect']
labels = ['HMM' ]

probTests = np.array(range(0,10)) - 4.

thym_calls = np.zeros(len(probTests))
thym_attempts = np.zeros(len(probTests))
brdu_calls = np.zeros(len(probTests))
brdu_attempts = np.zeros(len(probTests))

for i in range(0,len(fn_BrdU)):

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
			sixMer = splitLine[2]

			if (strand == 'rev' and not sixMer[-1:] == "A") or (strand == "fwd" and not sixMer[0] == "T"):
				continue


			BrdUprob = float(splitLine[1])

			for j in range(0,len(probTests)):
				if BrdUprob > probTests[j]:
					thym_calls[j] += 1.
				thym_attempts[j] += 1.
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
			sixMer = splitLine[2]

			if (strand == 'rev' and not sixMer[-1:] == "A") or (strand == "fwd" and not sixMer[0] == "T"):
				continue


			BrdUprob = float(splitLine[1])

			for j in range(0,len(probTests)):
				if BrdUprob > probTests[j]:
					brdu_calls[j] += 1
				brdu_attempts[j] += 1
	f.close()

	#plot on the figure
	x = thym_calls/thym_attempts
	y = brdu_calls/brdu_attempts
	plt.plot(x[::-1], y[::-1], label=labels[i])
	for p,txt in enumerate(probTests):
		ax.annotate(str(txt),(x[p],y[p]),fontsize=6)

plt.legend()
plt.xlim(0,1.0)
plt.xlabel('False Positive Rate')
plt.ylabel('Calls/Attempts')
plt.ylim(0,1)
plt.savefig('roc_curves_all_5k.pdf')
plt.xlim(0,0.2)
plt.savefig('roc_curves_all_5k_zoom.pdf')
plt.close()







