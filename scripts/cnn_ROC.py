import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import sys

maxReads = 5000
cooldownThreshold = 0


Tprefix = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode08/'
Bprefix = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/barcode11/'

fn_Thym = [Tprefix+'commite338d93_DNAscentCNN_softLabel_highIns_build29.detect',
 Tprefix+'commite338d93_DNAscentCNN_build38.detect',
 Tprefix+'commite338d93_DNAscentCNN_build46.detect',
 Tprefix+'commite338d93_DNAscentCNN_build52.detect',
 Tprefix+'commit620d798_DNAscentCNN_build55.detect',
 Tprefix+'commit620d798_DNAscentCNN_build56.detect',
 Tprefix+'commit620d798_DNAscentCNN_build57.detect',
 Tprefix+'commit620d798_DNAscentCNN_build58.detect',
 Tprefix+'commit620d798_DNAscentCNN_build59.detect',
 Tprefix+'commit620d798_DNAscentCNN_build60.detect']

fn_BrdU = [Bprefix+'commite338d93_DNAscentCNN_softLabel_highIns_build29.HMM.detect',
 Bprefix+'commite338d93_DNAscentCNN_build38.detect',
 Bprefix+'commite338d93_DNAscentCNN_build46.detect',
 Bprefix+'commite338d93_DNAscentCNN_build52.detect',
 Bprefix+'commit620d798_DNAscentCNN_build55.detect',
 Bprefix+'commit620d798_DNAscentCNN_build56.detect',
 Bprefix+'commit620d798_DNAscentCNN_build57.detect',
 Bprefix+'commit620d798_DNAscentCNN_build58.detect',
 Bprefix+'commit620d798_DNAscentCNN_build59.detect',
 Bprefix+'commit620d798_DNAscentCNN_build60.detect']


labels = ['build29',
 'build38',
 'build46',
 'build52',
 'build55',
 'build56',
 'build57',
 'build58',
 'build59',
 'build60']

probTests = np.array(range(1,10))/10.

thym_calls = np.zeros(len(probTests))
thym_attempts = np.zeros(len(probTests))
brdu_calls = np.zeros(len(probTests))
brdu_attempts = np.zeros(len(probTests))

call_cooldown = np.zeros(len(probTests))
attempts_cooldown = np.zeros(len(probTests))

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

		if line[0] == '#':
			continue

		if line[0] == '>':
			splitLine = line.rstrip().split()
			strand = splitLine[4]

			call_cooldown = np.zeros(len(probTests))
			attempts_cooldown = np.zeros(len(probTests))
	
			readCtr += 1
			if readCtr > maxReads:
				break

			continue
		else:
			splitLine = line.split('\t')
			sixMer = splitLine[3]
			position = int(splitLine[0])

			if (strand == 'rev' and not sixMer[-1:] == "A") or (strand == "fwd" and not sixMer[0] == "T"):
				continue


			BrdUprob = float(splitLine[2])

			for j in range(0,len(probTests)):
				if BrdUprob > probTests[j] and position - call_cooldown[j] >= cooldownThreshold:
					call_cooldown[j] = position
					attempts_cooldown[j] = position
					thym_calls[j] += 1.
					thym_attempts[j] += 1.
				elif BrdUprob < probTests[j] and position - attempts_cooldown[j] >= cooldownThreshold:
					attempts_cooldown[j] = position
					thym_attempts[j] += 1.
	f.close()

	#brdu
	f = open(fn_BrdU[i],'r')
	readCtr = 0
	for line in f:

		if line[0] == '#':
			continue

		if line[0] == '>':
			splitLine = line.rstrip().split()
			strand = splitLine[4]

			call_cooldown = np.zeros(len(probTests))
			attempts_cooldown = np.zeros(len(probTests))
	
			readCtr += 1
			if readCtr > maxReads:
				break

			continue
		else:
			splitLine = line.split('\t')
			sixMer = splitLine[3]
			position = int(splitLine[0])

			if (strand == 'rev' and not sixMer[-1:] == "A") or (strand == "fwd" and not sixMer[0] == "T"):
				continue


			BrdUprob = float(splitLine[2])

			for j in range(0,len(probTests)):
				if BrdUprob > probTests[j] and position - call_cooldown[j] >= cooldownThreshold:
					call_cooldown[j] = position
					attempts_cooldown[j] = position
					brdu_calls[j] += 1.
					brdu_attempts[j] += 1.
				elif BrdUprob < probTests[j] and position - attempts_cooldown[j] >= cooldownThreshold:
					attempts_cooldown[j] = position
					brdu_attempts[j] += 1.
	f.close()

	#plot on the figure
	x = thym_calls/thym_attempts
	y = brdu_calls/brdu_attempts
	plt.plot(x[::-1], y[::-1], label=labels[i])
	for p,txt in enumerate(probTests):
		ax.annotate(str(txt),(x[p],y[p]),fontsize=6)

#CNN END
'''

#DO FOR HMM
fn_BrdU = [Bprefix+'commit2d622cc1_HMM.detect', Bprefix+'v0.1.barcode08.detect']
fn_Thym = [Tprefix+'commit2d622cc1_HMM.detect', Tprefix+'v0.1.barcode08.detect']
labels = ['HMM (v1.0)', 'HMM (v0.1)' ]

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

		if line[0] == '#':
			continue

		if line[0] == '>':
			splitLine = line.rstrip().split()

			call_cooldown = np.zeros(len(probTests))
			attempts_cooldown = np.zeros(len(probTests))

			readCtr += 1
			if readCtr > maxReads:
				break

			continue
		else:
			splitLine = line.split('\t')
			sixMer = splitLine[2]

			BrdUprob = float(splitLine[1])

			for j in range(0,len(probTests)):
				if BrdUprob > probTests[j] and position - call_cooldown[j] >= cooldownThreshold:
					call_cooldown[j] = position
					attempts_cooldown[j] = position
					thym_calls[j] += 1.
					thym_attempts[j] += 1.
				elif BrdUprob < probTests[j] and position - attempts_cooldown[j] >= cooldownThreshold:
					attempts_cooldown[j] = position
					thym_attempts[j] += 1.
	f.close()

	#brdu
	f = open(fn_BrdU[i],'r')
	readCtr = 0
	for line in f:

		if line[0] == '#':
			continue

		if line[0] == '>':
			splitLine = line.rstrip().split()

			call_cooldown = np.zeros(len(probTests))
			attempts_cooldown = np.zeros(len(probTests))

			readCtr += 1
			if readCtr > maxReads:
				break

			continue
		else:
			splitLine = line.split('\t')
			sixMer = splitLine[2]


			BrdUprob = float(splitLine[1])

			for j in range(0,len(probTests)):
				if BrdUprob > probTests[j] and position - call_cooldown[j] >= cooldownThreshold:
					call_cooldown[j] = position
					attempts_cooldown[j] = position
					brdu_calls[j] += 1.
					brdu_attempts[j] += 1.
				elif BrdUprob < probTests[j] and position - attempts_cooldown[j] >= cooldownThreshold:
					attempts_cooldown[j] = position
					brdu_attempts[j] += 1.
	f.close()

	#plot on the figure
	x = thym_calls/thym_attempts
	y = brdu_calls/brdu_attempts
	plt.plot(x[::-1], y[::-1], label=labels[i])
	for p,txt in enumerate(probTests):
		ax.annotate(str(txt),(x[p],y[p]),fontsize=6)
'''
plt.legend(framealpha=0.5)
plt.xlim(0,1.0)
plt.xlabel('False Positive Rate')
plt.ylabel('Calls/Attempts')
plt.ylim(0,1)
plt.savefig('roc_curves.pdf')
plt.xlim(0,0.2)
plt.savefig('roc_curves_zoom.pdf')
plt.close()
