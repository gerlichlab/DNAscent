import sys
import os
import numpy as np
from scipy.signal import savgol_filter

f = open(sys.argv[1],'r')
first = True
count = 0
signalLineBuffer = []
forkDirLineBuffer = []

filesPerDir = 300
directoryCount = 0

for line in f:

	if not line.rstrip():
		continue

	if line[0] == '#':
		continue
	
	if line[0] == '>':

		if not first:

			if count % filesPerDir == 0:
				directoryCount += 1
				os.system('mkdir '+str(directoryCount))

			count += 1

			#leftward moving fork
			f_bg = open( str(directoryCount) + '/' + str(count) + '_' + readID + '.bedgraph','w')
			f_bg.write( 'track type=bedGraph name="'+readID +'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')


			#buffer_prob = savgol_filter(buffer_prob, 501, 5)

			buffer_prob= np.convolve(buffer_prob, np.ones((40,))/40, mode='same')

			for i, l in enumerate(buffer_prob):
				f_bg.write(chromosome + ' ' + str(buffer_pos[i]) + ' ' + str(buffer_pos[i]+1) + ' ' + str(l) + '\n')
			f_bg.close()
				
		#get readID and chromosome
		splitLine = line.rstrip().split(' ')
		readID = splitLine[0][1:]
		chromosome = splitLine[1]
		strand = splitLine[4]
		prevPos = int(splitLine[2]) #set the previous position for below to where this read started mapping

		first = False
		printThisOne = False
		buffer_prob = []
		buffer_pos = []

	else:

		splitLine = line.rstrip().split()
		pos = int(splitLine[0])
		probBrdU = float(splitLine[1])

		#left and right moving forks
		buffer_pos.append(pos)
		buffer_prob.append(probBrdU)

		prevPos = pos

f.close()

count += 1

#leftward moving fork
f_bg = open( str(directoryCount) + '/' + str(count) + '_' + readID + '.bedgraph','w')
f_bg.write( 'track type=bedGraph name="'+readID +'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

for l in buffer_prob:
	f_bg.write(l)
f_bg.close()

