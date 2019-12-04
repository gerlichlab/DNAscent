import sys

threshold = 2.5

f = open(sys.argv[1],'r')
numCalls = 0
numAttempts = 0

numFwdCalls = 0
numFwdMethylCalls = 0
numFwdAttempts = 0
numFwdMethylAttempts = 0
onFwd = False

numRevCalls = 0
numRevMethylCalls = 0
numRevAttempts = 0
numRevMethylAttempts = 0
onRev = False

numFwdMethylDeclined = 0
numRevMethylDeclined = 0

for line in f:

	if line[0] == '>':

		#splitLineHash = line.rstrip().split('#')
		splitLineHash = line.rstrip().split(' ')
		
#		if splitLineHash[1] == 'rev':
		if splitLineHash[-1:][0] == 'rev':
			onRev = True
			onFwd = False
		else:
			onRev = False
			onFwd = True

		continue
	else:
		splitLine = line.split('\t')
		logLikelihood = float(splitLine[1])
		if len(splitLine) > 4:
			logLikelihood_BrdUvsMethyl = float(splitLine[2])
			logLikelihood_MethylvsThym = float(splitLine[3])

			if onRev:
				if logLikelihood > threshold and logLikelihood_BrdUvsMethyl < threshold:
					numRevMethylDeclined += 1
				elif logLikelihood > threshold and logLikelihood_BrdUvsMethyl > threshold:
					numRevCalls += 1
				elif logLikelihood_MethylvsThym > threshold and logLikelihood_BrdUvsMethyl < threshold:
					numRevMethylCalls += 1
				numRevAttempts += 1
				numRevMethylAttempts += 1
			else:
				if logLikelihood > threshold and logLikelihood_BrdUvsMethyl < threshold:
					numFwdMethylDeclined += 1
				elif logLikelihood > threshold and logLikelihood_BrdUvsMethyl > threshold:
					numFwdCalls += 1
				elif logLikelihood_MethylvsThym > threshold and logLikelihood_BrdUvsMethyl < threshold:
					numFwdMethylCalls += 1
				numFwdAttempts += 1
				numFwdMethylAttempts += 1


			if logLikelihood > threshold:
				numCalls += 1
			numAttempts += 1


		else:
			if onRev:
				if logLikelihood > threshold:
					numRevCalls += 1
				numRevAttempts += 1
			else:
				if logLikelihood > threshold:
					numFwdCalls += 1
				numFwdAttempts += 1
			

			if logLikelihood > threshold:
				numCalls += 1
			numAttempts += 1
				
	
print "All BrdU Calls: ",float(numCalls)/float(numAttempts) * 100, '%'
print "Fwd Strand BrdU Calls: ",float(numFwdCalls)/float(numFwdAttempts) * 100, '%'
print "Fwd Strand BrdU Declined To Methyl: ",float(numFwdMethylDeclined)/float(numFwdAttempts) * 100, '%'
if numFwdMethylAttempts > 0:
	print "Fwd Strand Methyl Calls: ",float(numFwdMethylCalls)/float(numFwdMethylAttempts) * 100, '%'
print "Rev Strand BrdU Calls: ",float(numRevCalls)/float(numRevAttempts) * 100, '%'
print "Rev Strand BrdU Declined To Methyl: ",float(numRevMethylDeclined)/float(numRevAttempts) * 100, '%'
if numRevMethylAttempts > 0:
	print "Rev Strand Methyl Calls: ",float(numRevMethylCalls)/float(numRevMethylAttempts) * 100, '%'
f.close()
