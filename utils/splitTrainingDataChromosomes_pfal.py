import sys
import math

baseDir = sys.argv[2]

f = open(sys.argv[1],'r')

readCount = 0
for line in f:
	if line[0] == '>':

		readCount += 1
		if readCount % 100 == 0:
			print(readCount)

		splitLine = line.rstrip().split()
		readID = splitLine[0][1:]
		chromosome = splitLine[1]
		mappingStart = int(splitLine[2])
		mappingEnd = int(splitLine[3])

		strand = splitLine[4]
		group = int(math.floor(mappingStart/100000.0)*100000.0)

		g = open(baseDir + chromosome + '_' + strand + '_' + str(group) + '.trainingData','a+')
		g.write(line)
	else:
		g.write(line)
