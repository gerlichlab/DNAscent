import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from Bio import SeqIO
from Bio.SeqUtils import GC
from Bio import AlignIO
from Bio.Align import AlignInfo
from Bio.Seq import Seq


forkSenseOutput = '/home/mb915/rds/rds-mb915-notbackedup/data/2019_11_29_FT_ONT_Plasmodium_Barcoded/barcode07/commit38c86ba_l_20000_q_10.barcode07.forkSense'
targetID = '6de1a824-d74c-4dbb-9004-26720e003932'
            
x = []
left = []
right = []
stall = []
flipOn = False
chromosome = ''
strand = ''
mappingStart = 0
mappingEnd = 0

f = open(forkSenseOutput,'r')
for line in f:
	if line[0] == '>':
		splitLine = line.rstrip().split(' ')
		if splitLine[0][1:] == targetID:
			chromosome = splitLine[1]
			mappingStart = int(splitLine[2])
			mappingEnd = int(splitLine[3])
			strand = splitLine[4]
			flipOn = True
		else:
			if flipOn:
				break
			flipOn = False
	else:
		if flipOn and line != '\n':
			splitLine = line.rstrip().split('\t')
			x.append(int(splitLine[0]))
			left.append(float(splitLine[1]))
			right.append(float(splitLine[3]))
			stall.append(float(splitLine[4]))

#OPTIONAL
#get AT-genome richness

f_genome = '/home/mb915/rds/rds-mb915-notbackedup/genomes/Plasmodium_falciparum.ASM276v2.fasta'
chr2seq = {}
for record in SeqIO.parse(f_genome,"fasta"):
	chr2seq[record.id] = record.seq

chrSeq = chr2seq[chromosome]
if strand == 'rev':
	chrSeq = str(Seq(str(chrSeq).reverse_complement()))

window = 2000
x_bar = range(mappingStart,mappingEnd,window)
y_bar = []
for indx in x_bar:
	y_bar.append( (100 - GC(chrSeq[indx:indx+window]))/float(100))

#ENDOPTIONAL





plt.figure()
plt.plot(x,left,label='left')
plt.plot(x,right,label='right')
plt.plot(x,stall,label='stall')
plt.xlabel('Chromosome ' + str(chromosome))
plt.ylabel('Probability')
plt.legend()
plt.bar(x_bar,y_bar,width=window,alpha=0.3)
plt.savefig(targetID + '.pdf')
