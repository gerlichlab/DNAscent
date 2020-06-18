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

		#only consider T positions
		#if (strand == "fwd" and sixMer[0] != "T") or (strand == "rev" and sixMer[-1:] != "A"):
		#	continue

		#left and right moving forks
		buffer_prob.append( chromosome + ' ' + str(pos) + ' ' + str(pos+1) + ' ' + str(probBrdU) + '\n' )

		prevPos = pos

f.close()


####################################################################################################################################
#FORKSENSE OUTPUT
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


				#leftward moving fork
				f_forkLeft = open( str(readID2directory[readID]) + '/' + readID + '_forkLeft.bedgraph','w')
				f_forkLeft.write( 'track type=bedGraph name="'+readID + '_' + strand + '_forkLeft'+'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

				for l in buffer_forkLeft:
					f_forkLeft.write(l)
				f_forkLeft.close()

				#rightward moving fork
				f_forkRight = open( str(readID2directory[readID]) + '/' + readID + '_forkRight.bedgraph','w')
				f_forkRight.write( 'track type=bedGraph name="'+readID + '_' + strand + '_forkRight'+'" description="BedGraph format" visibility=full color=0,0,255 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

				for l in buffer_forkRight:
					f_forkRight.write(l)
				f_forkRight.close()

				#saturated
				#f_forkStall = open( str(readID2directory[readID]) + '/' + readID + '_saturate.bedgraph','w')
				#f_forkStall.write( 'track type=bedGraph name="'+readID + '_' + strand + '_saturate'+'" description="BedGraph format" visibility=full color=255,0,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

				#for l in buffer_stall:
				#	f_forkStall.write(l)
				#f_forkStall.close()
				
		#get readID and chromosome
		splitLine = line.rstrip().split(' ')
		readID = splitLine[0][1:]
		chromosome = splitLine[1]
		strand = splitLine[-1:][0]
		prevPos = int(splitLine[2]) #set the previous position for below to where this read started mapping

		first = False
		buffer_forkLeft = []
		buffer_forkRight = []
		#buffer_stall = []

	else:

		splitLine = line.rstrip().split()
		pos = int(splitLine[0])
		probForkLeft = float(splitLine[1])
		probForkRight = float(splitLine[2])
		#probSaturate = float(splitLine[2])

		#left and right moving forks
		buffer_forkLeft.append( chromosome + ' ' + str(prevPos) + ' ' + str(pos) + ' ' + str(probForkLeft) + '\n' )
		buffer_forkRight.append( chromosome + ' ' + str(prevPos) + ' ' + str(pos) + ' ' + str(probForkRight) + '\n' )
		
		#buffer_stall.append( chromosome + ' ' + str(prevPos) + ' ' + str(pos) + ' ' + str(probSaturate) + '\n' )

		prevPos = pos

f.close()
