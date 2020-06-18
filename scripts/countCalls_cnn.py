import sys

threshold = 0.50

f = open(sys.argv[1],'r')
numCalls = 0
numAttempts = 0

for line in f:

	if line[0] == '#':
		continue

	if line[0] == '>':

		splitLine = line.rstrip().split()
		strand = splitLine[4]

		callsCooldown = 0
		attemptsCooldown = 0

		continue
	else:

		splitLine = line.split('\t')
		sixMer = splitLine[3]

		if (strand == 'rev' and not sixMer[-1:] == "A") or (strand == "fwd" and not sixMer[0] == "T"):
			continue

		splitLine = line.split('\t')
		BrdUprob = float(splitLine[2])
		position = int(splitLine[0])
		
		if BrdUprob > threshold:
			numCalls += 1
		numAttempts += 1

				
	
print("All BrdU Calls: ",float(numCalls)/float(numAttempts) * 100, '%')
f.close()
