import sys
import os

filesPerDir = 300
directoryCount = 0

maxReads = 900

####################################################################################################################################
#PARSE AND RECORD DETECT FILE
print('Parsing detect file...')
first = True
count = 0
f = open(sys.argv[2],'r')
readID2directory = {}
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

			if count > maxReads:
				break

			readID2directory[readID] = directoryCount
			f_bg = open( str(directoryCount) + '/' + readID + '.bedgraph','w')
			f_bg.write( 'track type=bedGraph name="'+readID +'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

			for l in buffer_prob:
				f_bg.write(l)
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

	else:

		splitLine = line.rstrip().split()
		pos = int(splitLine[0])
		probBrdU = float(splitLine[1])
		sixMer = splitLine[2]

		#left and right moving forks
		buffer_prob.append( chromosome + ' ' + str(pos) + ' ' + str(pos+1) + ' ' + str(probBrdU) + '\n' )

		prevPos = pos

f.close()


####################################################################################################################################
#REGIONS OUTPUT
print('Parsing forkSense file...')
f = open(sys.argv[1],'r')
first = True
count = 0
signalLineBuffer = []
forkDirLineBuffer = []

for line in f:

	if not line.rstrip():
		continue

	if line[0] == '#':
		continue
	
	if line[0] == '>':

		if not first:

			count += 1

			if count > maxReads:
				break

			#DNAscent regions
			if readID in readID2directory:

				f_regions = open( str(readID2directory[readID]) + '/' + readID + '_regions.bedgraph','w')
				f_regions.write( 'track type=bedGraph name="'+readID + '_' + strand + '_regions'+'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=-3.0:3.0'+'\n')

				for l in buffer_regions:
					f_regions.write(l)
				f_regions.close()
				
		#get readID and chromosome
		splitLine = line.rstrip().split(' ')
		readID = splitLine[0][1:]
		chromosome = splitLine[1]
		strand = splitLine[-1:][0]

		first = False
		buffer_regions = []

	else:

		splitLine = line.rstrip().split()
		posStart = int(splitLine[0])
		posEnd = int(splitLine[1])
		regionScore = float(splitLine[2])

		#left and right moving forks
		buffer_regions.append( chromosome + ' ' + str(posStart) + ' ' + str(posEnd) + ' ' + str(regionScore) + '\n' )

f.close()
