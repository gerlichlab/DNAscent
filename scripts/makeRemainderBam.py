#this script takes a DNAscent detect file where DNAscent detect has been stopped prematurely
#and takes the BAM file it was running on, and makes a new BAM file out of the reads that
#DNAscent detect hasn't operated on yet

import pysam

f_bam = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_09_26_CAM_ONT_2085_1x_cell_cycle/alignments.sorted.fixEOF.bam'

f_grp1 = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_09_26_CAM_ONT_2085_1x_cell_cycle/alignments.sorted.fixEOF.grp1.bam'
f_grp2 = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_09_26_CAM_ONT_2085_1x_cell_cycle/alignments.sorted.fixEOF.grp2.bam'
f_grp3 = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_09_26_CAM_ONT_2085_1x_cell_cycle/alignments.sorted.fixEOF.grp3.bam'
f_grp4 = '/home/mb915/rds/rds-mb915-notbackedup/data/2018_09_26_CAM_ONT_2085_1x_cell_cycle/alignments.sorted.fixEOF.grp4.bam'

f = pysam.Samfile(f_bam,'r')
g_grp1 = pysam.Samfile(f_grp1,'wb',template=f)
g_grp2 = pysam.Samfile(f_grp2,'wb',template=f)
g_grp3 = pysam.Samfile(f_grp3,'wb',template=f)
g_grp4 = pysam.Samfile(f_grp4,'wb',template=f)

for record in f:
	if not record.is_unmapped:
		if record.reference_name in ['chrI','chrII','chrIII','chrIV']:
			g_grp1.write(record)

		elif record.reference_name in ['chrV','chrVI','chrVII','chrVIII']:
			g_grp2.write(record)

		elif record.reference_name in ['chrIX','chrX','chrXI','chrXII']:
			g_grp3.write(record)

		elif record.reference_name in ['chrXIII','chrXIV','chrXV','chrXVI']:
			g_grp4.write(record)

f.close()
g_grp1.close()
g_grp2.close()
g_grp3.close()
g_grp4.close()
