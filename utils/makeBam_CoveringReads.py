import pysam
import sys

#INPUTS
f_IDs = sys.argv[1] #the output from analoguePositiveIDs_CNN.py
f_alignment = sys.argv[2] #alignments.sorted.bam of the analogue-negative sample

#OUTPUTS
#coveringAnalogueNegativeReads.bam - the analogue-negative reads that map over at least one analogue-positive read
#coveredAnaloguePositiveIDs.txt - readIDs of the analogue-positive reads that are covered. Can be passed to makeTrainingBam.py for subsequent DNAscent trainCNN


#get analogue-positive IDs
fwd_chr2coords = {}
rev_chr2coords = {}
f = open(f_IDs,'r')
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

f = pysam.Samfile(f_alignment,'r')
g = pysam.Samfile('coveringAnalogueNegitiveReads.bam','wb',template=f)
keepIDs = []
for i, record in enumerate(f):

	found = False

	if i % 1000 == 0:
		print("Record:",i)

	if not record.is_unmapped and record.reference_name is not None:
		if record.is_reverse and record.reference_name in rev_chr2coords:
			for read in rev_chr2coords[record.reference_name]:
				if record.reference_start < read[0] and read[1] < record.reference_end:
					
					#keep the matched analogue-positive ID
					if read[2] not in keepIDs:
						keepIDs.append(read[2])
						
					found = True
						
		elif not record.is_reverse and record.reference_name in fwd_chr2coords:
			for read in fwd_chr2coords[record.reference_name]:
				if record.reference_start < read[0] and read[1] < record.reference_end:
					
					#keep the matched analogue-positive ID
					if read[2] not in keepIDs:
						keepIDs.append(read[2])
						
					found = True
	if found:
		g.write(record)		
						
						
f.close()
g.close()

f = open('coveredAnaloguePositiveIDs.txt','w')
for ID in keepIDs:
	f.write(ID + '\n')
f.close()

