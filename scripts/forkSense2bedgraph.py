import sys
import os

includeStall = True

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
	
	if line[0] == '>':

		if not first:

			if count % filesPerDir == 0:
				directoryCount += 1
				os.system('mkdir '+str(directoryCount))

			count += 1

			#leftward moving fork
			f_forkLeft = open( str(directoryCount) + '/' + str(count) + '_' + readID + '_forkLeft.bedgraph','w')
			f_forkLeft.write( 'track type=bedGraph name="'+readID + '_forkLeft'+'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

			for l in buffer_forkLeft:
				f_forkLeft.write(l)
			f_forkLeft.close()

			#rightward moving fork
			f_forkRight = open( str(directoryCount) + '/' + str(count) + '_' + readID + '_forkRight.bedgraph','w')
			f_forkRight.write( 'track type=bedGraph name="'+readID + '_forkRight'+'" description="BedGraph format" visibility=full color=0,0,255 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

			for l in buffer_forkRight:
				f_forkRight.write(l)
			f_forkRight.close()

			#fork direction
			if len( buffer_stall ) > 0 and includeStall:

				f_forkStall = open( str(directoryCount) + '/' + str(count) + '_' + readID + '_forkStall.bedgraph','w')
				f_forkStall.write( 'track type=bedGraph name="'+readID + '_forkStall'+'" description="BedGraph format" visibility=full color=255,0,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

				for l in buffer_stall:
					f_forkStall.write(l)
				f_forkStall.close()
				
		#get readID and chromosome
		splitLine = line.rstrip().split(' ')
		readID = splitLine[0][1:]
		chromosome = splitLine[1]
		prevPos = int(splitLine[2]) #set the previous position for below to where this read started mapping

		first = False
		printThisOne = False
		buffer_forkLeft = []
		buffer_forkRight = []
		buffer_stall = []

	else:

		splitLine = line.rstrip().split()
		pos = int(splitLine[0])
		probForkLeft = float(splitLine[1])
		probForkRight = float(splitLine[3])

		#left and right moving forks
		buffer_forkLeft.append( chromosome + ' ' + str(prevPos) + ' ' + str(pos) + ' ' + str(probForkLeft) + '\n' )
		buffer_forkRight.append( chromosome + ' ' + str(prevPos) + ' ' + str(pos) + ' ' + str(probForkRight) + '\n' )
		
		#fork stall if we're using it
		if includeStall:
			probStall = float(splitLine[4])
			buffer_stall.append( chromosome + ' ' + str(prevPos) + ' ' + str(pos) + ' ' + str(probStall) + '\n' )

		prevPos = pos

f.close()

count += 1

#leftward moving fork
f_forkLeft = open( str(directoryCount) + '/' + str(count) + '_' + readID + '_forkLeft.bedgraph','w')
f_forkLeft.write( 'track type=bedGraph name="'+readID + '_forkLeft'+'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

for l in buffer_forkLeft:
	f_forkLeft.write(l)
f_forkLeft.close()

#rightward moving fork
f_forkRight = open( str(directoryCount) + '/' + str(count) + '_' + readID + '_forkRight.bedgraph','w')
f_forkRight.write( 'track type=bedGraph name="'+readID + '_forkRight'+'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

for l in buffer_forkRight:
	f_forkRight.write(l)
f_forkRight.close()

#fork direction
if len( buffer_stall ) > 0 and includeStall:
	f_forkStall = open( str(directoryCount) + '/' + str(count) + '_' + readID + '_forkStall.bedgraph','w')
	f_forkStall.write( 'track type=bedGraph name="'+readID + '_forkStall'+'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

	for l in buffer_stall:
		f_forkStall.write(l)
	f_forkStall.close()
