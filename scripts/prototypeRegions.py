import pysam
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

windowLength = 500
llThreshold = 2.5

f_detect = '/home/mb915/rds/rds-mb915-notbackedup/data/2019_11_29_FT_ONT_Plasmodium_Barcoded/barcode07/commit57d6b83_l_20000_q_10.barcode07.detect'
dir_plots = '/home/mb915/rds/rds-mb915-notbackedup/data/2019_11_29_FT_ONT_Plasmodium_Barcoded/barcode07/plots'

buffer_score = []
buffer_positions = []

def rolling(positions,scores,windowLength):
	
	x = []
	y = []

	startPos = positions[0]
	for i, p in enumerate(positions):

		calls = 0
		attempts = 0

		for j,q in enumerate(positions[i:]):
			
			if q - p > windowLength:
				break
			else:
				if scores[j+i] > llThreshold:
					calls += 1
				attempts += 1
		
		x.append(q)
		y.append(float(calls)/attempts)

	return x,y


f = open(f_detect,'r')
for line in f:
	if line[0] == '>':

		if len(buffer_score) > 0:

			[x,y] = rolling(buffer_positions,buffer_score,windowLength)
			plt.figure()
			plt.plot(x,y)
			plt.xlabel(chromosome)
			plt.ylabel('Positive Calls / Attempts')
			plt.ylim(-0.1,1.1)
			plt.savefig(dir_plots+'/'+readID+'.png')
			plt.close()
			
		[readID,chromosome,mapStart,mapEnd,strand] = line.rstrip().split()
		readID = readID[1:]
		mapStart = int(mapStart)
		mapEnd = int(mapEnd)
		buffer_score = []
		buffer_positions = []

	else:

		splitLine = line.rstrip().split()
		buffer_score.append(float(splitLine[1]))
		buffer_positions.append(int(splitLine[0]))
		

