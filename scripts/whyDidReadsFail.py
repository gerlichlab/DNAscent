import pysam
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#QC
minLen = 20000
minMQ = 10

f = pysam.Samfile(sys.argv[1],'r')
totalRecords = 0
failedRefLen = 0
failedQueryLen = 0
failedMQ = 0
failedBoth = 0
nonUnique = []
mq2count = {}
refLength = []
queryLength = []
totalReadCount = 0
totalReadLen = 0
for record in f:

	if record.reference_length is not None and record.query_length is not None:
		refLength.append(record.reference_length)
		queryLength.append(record.query_length)

	totalRecords += 1
	if record.reference_length is not None:
		totalReadLen += record.reference_length
		totalReadCount += 1
	if record.reference_length < minLen and record.mapping_quality < minMQ:
		failedBoth += 1
	elif record.reference_length < minLen:
		failedRefLen += 1
	elif record.mapping_quality < minMQ:
		failedMQ += 1

	if record.query_length < minLen:
		failedQueryLen += 1

	if record.mapping_quality in mq2count:
		mq2count[record.mapping_quality] += 1
	else:
		mq2count[record.mapping_quality] = 1

f.close()

x = []
y = []
for key in sorted(mq2count):
	x.append(key)
	y.append(mq2count[key])

plt.figure()
plt.bar(x,y,log=True)
plt.xlabel('Mapping Quality')
plt.ylabel('Read Count')
plt.savefig('mappingQuality.pdf')
plt.close()

plt.figure()
refLength = np.array(refLength)
queryLength = np.array(queryLength)
plt.scatter(np.true_divide(refLength,1000.0),np.true_divide(queryLength,1000.0),alpha=0.3)
plt.xlabel('Reference Length (kb)')
plt.ylabel('Query Length (kb)')
plt.savefig('queryRefLength.png')
plt.close()

print "Total reads:",totalRecords
print "Failed reference length:",failedRefLen
print "Failed mapping quality:",failedMQ
print "Failed both:",failedBoth
print "Failed query length:",failedQueryLen
print "Average read length:",float(totalReadLen)/totalReadCount
