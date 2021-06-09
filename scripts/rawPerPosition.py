import sys
import numpy as np

maxReads = 100
readCtr = 0

currentPos = -1
ctrRaw = 0

rawAtEachPos = []

f = open(sys.argv[1],'r')
for line in f:
	if line[0] == '>':
		readCtr += 1
		if readCtr > maxReads:
			break

		currentPos = -1
		ctrRaw = 0
		continue
	if line[0] == '#':
		continue

	splitLine = line.rstrip().split()
	pos = int(splitLine[0])
	if pos != currentPos:
		if currentPos != -1:
			rawAtEachPos.append(ctrRaw)
		ctrRaw = 0
		currentPos = pos
	else:
		ctrRaw += 1
f.close()

print(np.mean(rawAtEachPos))
print(np.std(rawAtEachPos))
		
	
	
