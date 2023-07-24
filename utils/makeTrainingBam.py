import pysam
import sys

#get analogue-positive IDs
f_IDs = sys.argv[1]
IDs = []
f = open(f_IDs,'r')
for line in f:
	IDs.append(line.rstrip())
f.close()

f_alignment = sys.argv[2]

f = pysam.Samfile(f_alignment,'r')
g = pysam.Samfile('analoguePositiveIDs.bam','wb',template=f)
for record in f:
	if record.query_name in IDs:
		g.write(record)
f.close()
g.close()
