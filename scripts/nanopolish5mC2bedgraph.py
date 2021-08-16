#----------------------------------------------------------
# Copyright 2020 University of Cambridge
# Written by Michael A. Boemo (mb915@cam.ac.uk)
# This software is licensed under GPL-2.0.  You should have
# received a copy of the license with this software.  If
# not, please Email the author.
#----------------------------------------------------------

import sys
import os


#--------------------------------------------------------------------------------------------------------------------------------------
def splashHelp():
	s = """dnascent2bedgraph.py: Converts the output of DNAscent detect, regions, and forkSense into bedgraphs.
To run dnascent2bedgraph.py, do:
  python dnascent2bedgraph.py [arguments]
Example:
  python dnascent2bedgraph.py -d /path/to/dnascentDetect.out -f /path/to/dnascentForksense.out -o /path/to/newBedgraphDir
Required arguments are at least one of the following:
  -d,--detect               path to nanopolish methyl detect output file.
Required argument is:
  -o,--output               output directory which will be created.
Optional arguments are:
  -n,--maxReads             maximum number of reads to convert into bedgraphs (default: Inf),
     --targets              forkSense bed file with specific reads to plot,
     --filesPerDir          maximum reads per subdirectory (default: 300).
Written by Michael Boemo, Department of Pathology, University of Cambridge.
Please submit bug reports to GitHub Issues (https://github.com/MBoemo/DNAscent/issues)."""

	print(s)
	exit(0)


#--------------------------------------------------------------------------------------------------------------------------------------
class arguments:
	pass


#--------------------------------------------------------------------------------------------------------------------------------------
def parseArguments(args):

	a = arguments()

	a.maxReads = 1000000000
	a.filesPerDir = 300
	a.useTargets = False

	for i, argument in enumerate(args):
			
		if argument == '-d' or argument == '--detect':
			a.detectPath = str(args[i+1])

		elif argument == '-o' or argument == '--output':
			a.outDir = str(args[i+1])

		elif argument == '--targets':
			a.targetPath = str(args[i+1])
			a.useTargets = True

		elif argument == '-n' or argument == '--maxReads':
			a.maxReads = int(args[i+1])

		elif argument == '--filesPerDir':
			a.filesPerDir = int(args[i+1])

		elif argument == '-h' or argument == '--help':
			splashHelp()

	#check that required arguments are met
	if not ( hasattr( a, 'detectPath') and  hasattr( a, 'outDir') ):
		splashHelp() 
	return a


#--------------------------------------------------------------------------------------------------------------------------------------
def makeDetectLine(line, chromosome):
	splitLine = line.rstrip().split()
	pos = int(splitLine[2])
	probBrdU = float(splitLine[5])
	sixMer = splitLine[2]
	return chromosome + ' ' + str(pos) + ' ' + str(pos+1) + ' ' + str(probBrdU) + '\n'


#--------------------------------------------------------------------------------------------------------------------------------------
def parseBaseFile(fname, args, targetIDs):
	print('Parsing '+fname[0]+'...')
	first = True
	count = 0
	f = open(fname[0],'r')
	readID2directory = {}
	directoryCount = 0
	prevReadID = ''

	for line in f:

		#ignore blank lines just in case (but these shouldn't exist anyway)
		if not line.rstrip():
			continue

		#ignore the file header
		if line[0:10] == 'chromosome':
			continue

		splitLine = line.rstrip().split()
		currentReadID = splitLine[4]

		if currentReadID != prevReadID:

			if (not first) and ((args.useTargets and prevReadID in targetIDs) or (not args.useTargets) ):

				if count % args.filesPerDir == 0:
					directoryCount += 1
					os.system('mkdir '+args.outDir + '/'+str(directoryCount))

				count += 1

				#stop if we've hit the max reads
				if count > args.maxReads:
					break

				readID2directory[prevReadID] = directoryCount

				f_bg = open( args.outDir + '/' + str(directoryCount) + '/' + prevReadID + '.5mC_detect.bedgraph','w')
				f_bg.write( 'track type=bedGraph name="'+prevReadID +'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=2.5:5.0'+'\n')

				for l in buff:
					f_bg.write(l)
				f_bg.close()

			#get readID and chromosome
			chromosome = splitLine[0]
			strand = splitLine[1]

			first = False
			prevReadID = currentReadID
			buff = []
		else:

			buff.append( makeDetectLine(line,chromosome) )

	if count < args.maxReads:

		if count % args.filesPerDir == 0:
			directoryCount += 1
			os.system('mkdir '+args.outDir + '/'+str(directoryCount))

		readID2directory[currentReadID] = directoryCount

		f_bg = open( args.outDir + '/' + str(directoryCount) + '/' + currentReadID + '.5mC_detect.bedgraph','w')
		f_bg.write( 'track type=bedGraph name="'+currentReadID +'" description="BedGraph format" visibility=full color=200,100,0 altColor=0,100,200 priority=20 viewLimits=0.0:1.0'+'\n')

		for l in buff:
			f_bg.write(l)
		f_bg.close()

	f.close()
	print('Done.')
	return readID2directory


#--------------------------------------------------------------------------------------------------------------------------------------
#MAIN

args = parseArguments(sys.argv[1:])

#check the output 
if args.outDir[-1:] == "/":
	args.outDir = args.outDir[:-1]
if os.path.isdir(args.outDir):
	print('Output directory '+args.outDir+' already exists.  Exiting.')
	exit(0)
else:
	os.system('mkdir '+args.outDir)

baseFname = ""
secondaryFname = []

targetIDs = []
if args.useTargets:
	f = open(args.targetPath,'r')
	for line in f:
		splitLine = line.rstrip().split()
		readID = splitLine[3]
		
		targetIDs.append(readID)
	f.close()

baseFname = (args.detectPath,"detect")

readID2directory = parseBaseFile(baseFname, args, targetIDs)
for fname in secondaryFname:
	parseSecondaryFile(fname, readID2directory, args)

