import pysam
import sys

#INPUTS
f_IDs_BrdU = sys.argv[1] #the output from analoguePositiveIDs_CNN.py (BrdU)
f_IDs_EdU = sys.argv[2] #the output from analoguePositiveIDs_CNN.py (EdU)

#EdU should cover the BrdU reads

#OUTPUTS
#BrdUReadsCoveredByEdU.txt - readIDs of BrdU reads that are covered by a EdU read
#EdUReadsCoveringBrdU.txt - readIDs of the EdU reads that cover at least one BrdU read


#get analogue-positive IDs
fwd_chr2coords = {}
rev_chr2coords = {}
f = open(f_IDs_BrdU,'r')
for line in f:
	splitLine = line.rstrip().split() 
	readID = splitLine[0]
	contig = splitLine[1]
	mapStart = int(splitLine[2])
	mapEnd = int(splitLine[3])
	strand = splitLine[4]
	if strand == 'fwd':
		if contig not in fwd_chr2coords:
			fwd_chr2coords[contig] = []
		fwd_chr2coords[contig].append( (mapStart, mapEnd, readID) )
	elif strand == 'rev':
		if contig not in rev_chr2coords:
			rev_chr2coords[contig] = []
		rev_chr2coords[contig].append( (mapStart, mapEnd, readID) )
f.close()


f = open(f_IDs_EdU,'r')
g = open('EdUReadsCoveringBrdU.txt','w')
BrdUkeepIDs = []
for line in f:
	splitLine = line.rstrip().split() 
	readID = splitLine[0]
	contig = splitLine[1]
	mapStart = int(splitLine[2])
	mapEnd = int(splitLine[3])
	strand = splitLine[4]
	found = False
	if strand == 'rev' and contig in rev_chr2coords:
		for read in rev_chr2coords[contig]:
			if mapStart < read[0] and read[1] < mapEnd:
				
				#keep the matched analogue-positive ID
				if read[2] not in BrdUkeepIDs:
					BrdUkeepIDs.append(read[2])
					
				found = True
	elif strand == 'fwd' and contig in fwd_chr2coords:
		for read in fwd_chr2coords[contig]:
			if mapStart < read[0] and read[1] < mapEnd:
				
				#keep the matched analogue-positive ID
				if read[2] not in BrdUkeepIDs:
					BrdUkeepIDs.append(read[2])
					
				found = True
	if found:
		g.write(line)
			
f.close()
g.close()


f = open('BrdUReadsCoveredByEdU.txt','w')
for ID in BrdUkeepIDs:
	f.write(ID + '\n')
f.close()

