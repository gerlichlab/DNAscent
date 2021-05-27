import sys
import math

baseDir = '/home/mb915/rds/rds-mb915-notbackedup/data/2021_05_21_FT_ONT_Plasmodium_BrdU_EdU/Thym_trainingData/splitData/'

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

		g = open(baseDir + 'Thym_' + chromosome + '_' + strand + '_' + str(group) + '.trainingData','a+')
		g.write(line)
	else:
		g.write(line)
