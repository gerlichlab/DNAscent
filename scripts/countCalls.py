import sys

threshold = 1.25
cooldownThreshold = 4

f = open(sys.argv[1],'r')
numCalls = 0
numAttempts = 0

callsCooldown = 0
attemptsCooldown = 0

for line in f:

	if line[0] == '#':
		continue

	if line[0] == '>':

		callsCooldown = 0
		attemptsCooldown = 0

		continue
	else:
		splitLine = line.split('\t')
		logLikelihood = float(splitLine[1])
		position = int(splitLine[0])

		if logLikelihood > threshold and position - callsCooldown >= cooldownThreshold:
			callsCooldown = position
			attemptsCooldown = position
			numCalls += 1
			numAttempts += 1
		elif logLikelihood <= threshold and position - attemptsCooldown >= cooldownThreshold:
			numAttempts += 1
			attemptsCooldown = position
				
	
print("All BrdU Calls: ",float(numCalls)/float(numAttempts) * 100, '%')
f.close()
