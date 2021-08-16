import pysam
import sys

recordsPerBam = 20000
minLength = 15000
minQuality = 20


f = pysam.Samfile(sys.argv[1],'r')
bamFileCtr = 0
g = pysam.Samfile('alignments_'+str(bamFileCtr)+'.bam','wb',template=f)

ctr = 0
for i, record in enumerate(f):

	if not record.is_unmapped:
		if record.reference_length >= minLength and record.mapping_quality >= minQuality:
			g.write(record)
			ctr += 1

	#g.write(record)
	
			if (ctr) % recordsPerBam == 0:
				g.close()
				bamFileCtr += 1 
				g = pysam.Samfile('alignments_'+str(bamFileCtr)+'.bam','wb',template=f)
		
g.close()
f.close()
