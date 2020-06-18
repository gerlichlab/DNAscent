import sys

f = open(sys.argv[1],'r')
numCalls = 0
numAttempts = 0

for line in f:

	if line[0] == '>':

		splitLine = line.rstrip().split()
		strand = splitLine[4]

		continue
	else:

		splitLine = line.split('\t')
		sixMer = splitLine[4]

		if (sixMer[0] != "T"):
			continue

		splitLine = line.split('\t')
		alignedBrdU = int(splitLine[7])
		
		if alignedBrdU == 1:
			numCalls += 1
		numAttempts += 1

				
	
print("All BrdU Calls: ",float(numCalls)/float(numAttempts) * 100, '%')
f.close()
