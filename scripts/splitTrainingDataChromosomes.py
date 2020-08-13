import sys
import math

baseDir = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_06_18_CAM_ONT_gDNA_BrdU_40_60_80_100_full/cnn_training/commit620d798_trainingData_8features_bc8bc12_augmentation/'

f = open(sys.argv[1],'r')


for line in f:
	if line[0] == '>':
		splitLine = line.rstrip().split()
		readID = splitLine[0][1:]
		chromosome = splitLine[1]
		mappingStart = int(splitLine[2])
		mappingEnd = int(splitLine[3])

		strand = splitLine[4]
		group = int(math.floor(mappingStart/100000.0)*100000.0)

		g = open(baseDir + 'bc12_' + chromosome + '_' + strand + '_' + str(group) + '.detect','a+')
		g.write(line)
	else:
		g.write(line)
