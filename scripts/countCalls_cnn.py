import sys

threshold = 0.70

f = open(sys.argv[1],'r')
numCalls = 0
numAttempts = 0

maxReads = 5000
readCount = 0

for line in f:

	if line[0] == '#' or line[0] == '%':
		continue

	if line[0] == '>':

		readCount += 1

		splitLine = line.rstrip().split()
		strand = splitLine[4]

		callsCooldown = 0
		attemptsCooldown = 0

		continue

	if readCount > maxReads:
		break

	else:

		splitLine = line.split('\t')
		BrdUprob = float(splitLine[1])
		position = int(splitLine[0])
		
		if BrdUprob > threshold:
			numCalls += 1
		numAttempts += 1

				
	
print("All BrdU Calls: ",float(numCalls)/float(numAttempts) * 100, '%')
f.close()
