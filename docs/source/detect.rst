.. _detect:

Detect
===============================

``DNAscent detect`` is a ``DNAscent`` subprogram that goes through each read and, at each thymidine position, assigns the probability the thymidine is BrdU.

Usage
-----

.. code-block:: console

   To run DNAscent detect, do:
      DNAscent detect -b /path/to/alignment.bam -r /path/to/reference.fasta -i /path/to/index.dnascent -o /path/to/output.detect
   Required arguments are:
     -b,--bam                  path to alignment BAM file,
     -r,--reference            path to genome reference in fasta format,
     -i,--index                path to DNAscent index,
     -o,--output               path to output file that will be generated.
   Optional arguments are:
     -t,--threads              number of threads (default is 1 thread),
     --HMM                     revert to old style HMM-based detection,
     -q,--quality              minimum mapping quality (default is 20),
     -l,--length               minimum read length in bp (default is 100).

The main input of ``DNAscent detect`` is an alignment (bam file) between the sequence fastq from Guppy and the organism's reference genome.  This bam file should be sorted using ``samtools sort`` and indexed using ``samtools index`` so that there is a .bam.bai file in the same directory as the bam file. (Please see the example in :ref:`workflows` for details on how to do this.)  The full path to the reference genome used in the alignment should be passed using the ``-r`` flag, and the index required by the ``-i`` flag is the file created using ``DNAscent index`` (see :ref:`index`).  

Optional arguments include the number of threads, specified using the ``-t`` flag.  ``DNAscent detect`` multithreads quite well by analysing a separate read on each thread, so multithreading is recommended.  It is sometimes useful to only run ``DNAscent detect`` on reads that exceed a certain mapping quality or length threshold (as measured by the subsequence of the contig that the read maps to).  In order to do this without having to filter the bam file, DNAscent provides the ``-l`` and ``-q`` flags.  Any read in the bam file with a reference length lower than the value specificed with ``-l`` or a mapping quality lower than the value specified with ``-q`` will be ignored.

DNAscent v1.0 and earlier used hidden Markov models to evaluate a log likelihood of BrdU at each thymidine position. Since v2.0, this has been migrated to a ResNet.  In v2.0, ResNet-based detection is the default and recommended approach: internal testing has indicated it is more accurate than the HMM.  However, users wishing to do the old-style HMM-based detection (perhaps as part of a legacy workflow) can do so by adding the ``--HMM`` flag, at which time ``DNAscent detect`` will use HMM detection as in v1.0.

Before calling BrdU in a read, ``DNAscent detect`` must first perform a fast event alignment (see https://www.biorxiv.org/content/10.1101/130633v2 for more details).  Quality control checks are performed on these alignments, and if they're not passed, then the read fails and is ignored.  Hence, the number of reads in the output file will be slightly lower than the number of input reads.  Typical failure rates are about 5-10%, although this will vary slightly depending on the read length, the BrdU substitution rate, and the genome sequenced.

Output
------

``DNAscent detect`` will produce a single human-readable output file with the name and location that you specified using the ``-o`` flag.  To aid organisation and reproducibility, each detect file starts with a short header.  The start of each header line is always a hash (#) character, and it specifies the input files and settings used, as well as the version and commit of DNAscent used.  An example is as follows:

.. code-block:: console

   #Alignment /path/to/alignment.bam
   #Genome /path/to/reference.fasta
   #Index /path/to/index.dnascent
   #Threads 1
   #Mode CNN
   #MappingQuality 20
   #MappingLength 5000
   #Version 2.0.0
   #Commit 90acb6c4c79fc06476a5e670101be8c1a46b40da
   #SignalDilation 1.000000

You can easily access the header of any .detect file with ``head -10 /path/to/output.detect`` or, alternatively, ``grep '#' /path/to/output.detect``.

Below the header is data for each read.  Note that everything in this output file orients to the reference genome in the 5' --> 3' direction.  Each read starts with a line in the format:

.. code-block:: console

   >readID contig mappingStart mappingEnd strand

These lines always begin with a greater-than (>) character.  Therefore, an easy way to count the number of reads in the file is ``grep '>' detect.out | wc -l``.  The fields are:

* ``readID`` is a unique hexadecimal string assigned to each read by the Oxford Nanopore software,
* the read mapped between ``mappingStart`` and ``mappingEnd`` on ``contig``,
* ``strand`` either takes the value ``fwd``, indicating that the read mapped to the forward strand, or ``rev`` indicating that the read mapped to the reverse complement strand.

The following shows an example for a read that to the reverse strand between 239248 and 286543 on chrII.

.. code-block:: console

   >c602f23f-e892-42ba-8140-da949abafbdd chrII 239248 286543 rev

Below these "start of read" lines, each line corresponds to the position of a thymidine in that read.  There are three tab-separated columns:

* the coordinate on the reference,
* probability that the thymidine is actually BrdU,
* 6mer on the reference.


Consider the following example:

.. code-block:: console

   >bce09d16-271e-49d3-a5c2-c6b56e186cf3 chrI 95420 101928 rev 
   95422   0.080785        CTAGAA
   95424   0.075855        AGAAGA
   95426   0.091836        AAGACA
   95428   0.075640        GACATA
   95429   0.083507        ACATAA

Here, we're looking at the sequence CTAGAAGACATAA on the reference genome.  Because this read maps to the reverse complement, a call is made at every A (instead of T) on the reference.  If instead we looked at a read that mapped to the forward strand, an example would be:

.. code-block:: console

   >4a3e879c-5334-4d3e-a774-137e0434126b chrI 97325 105509 fwd
   97325   0.051484        TCTAGC
   97327   0.133578        TAGCTT
   97331   0.065396        TTCTCG
   97332   0.076378        TCTCGG
   97334   0.078366        TCGGCT

In both of these output snippets, we see from the second column that the probability of BrdU is low, so these few bases are likely from a BrdU-negative region of DNA.  In contrast, here we see the start of a read that does contain BrdU, and accordingly, the probability of BrdU is much higher:

.. code-block:: console

   >8607596e-2175-4e7b-b1af-86b96a9c80f2 chrI 0 10994 fwd
   62	0.826316	TCCTAA
   65	0.726879	TAACAC
   71	0.858950	TACCCT
   76	0.897366	TAACAC
   87	0.766869	TAATCT
