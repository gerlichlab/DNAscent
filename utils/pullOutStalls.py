import sys

g = open('stalls.bed','w')
f = open(sys.argv[1],'r')
for line in f:
	if line[0] == '#':
		continue
	splitLine = line.rstrip().split()
	score = float(splitLine[7])
	if score > 0.7:
		g.write(line)
f.close()

f = open(sys.argv[2],'r')
for line in f:
	if line[0] == '#':
		continue
	splitLine = line.rstrip().split()
	score = float(splitLine[7])
	if score > 0.7:
		g.write(line)
f.close()

g.close()
