import pysam
import sys

recordsPerBam = 100000

f = pysam.Samfile(sys.argv[1],'r')
bamFileCtr = 0
g = pysam.Samfile('alignments_'+str(bamFileCtr)+'.bam','wb',template=f)

for i, record in enumerate(f):

	g.write(record)
	
	if (i+1) % recordsPerBam == 0:
		g.close()
		bamFileCtr += 1 
		g = pysam.Samfile('alignments_'+str(bamFileCtr)+'.bam','wb',template=f)
		
g.close()
f.close()
