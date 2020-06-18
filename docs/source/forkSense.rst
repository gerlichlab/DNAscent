.. _forkSense:

forkSense
===============================

``DNAscent forkSense`` is a ``DNAscent`` subprogram that provides a probability estimate at each thymidine that a leftward- or rightward-moving fork moved through that position during the BrdU pulse.

Usage
-----

.. code-block:: console

   To run DNAscent forkSense, do:
      DNAscent forkSense -d /path/to/BrdUCalls.detect -o /path/to/output.forkSense
   Required arguments are:
     -d,--detect               path to output file from DNAscent detect,
     -o,--output               path to output file for forkSense.
   Optional arguments are:
     -t,--threads              number of threads (default: 1 thread),
     --markOrigins             writes replication origin locations to a bed file (default: off).

The only required input of ``DNAscent forkSense`` is the output file produced by ``DNAscent detect``.  Note that the detect file must have been produced using the v2.0 default ResNet algorithm; ``DNAscent forkSense`` is not compatible with legacy HMM-based detection.

If the ``--markOrigins`` flag is passed, ``DNAscent forkSense`` will use detected leftward- and rightward-moving forks to infer the locations of fired replication origins and write these to a bed file called ``origins_DNAscent_forkSense.bed`` in the working directory.  

Output
------

If ``--markOrigins`` was used, the resulting bed file has one called origin per line and, in accordance with bed format, have the following space-separated columns:

* chromosome name,
* 5' boundary of the origin,
* 3' boundary of the origin,
* read header of the read that the called origin came from (similar to those in the output file of ``DNAscent detect``).

Note that the "resolution" of the origin call (i.e., the third column minus the second column) will depend on your experimental setup.  In synchronised early S-phase cells, the difference is likely to be small as the leftward- and rightward-moving forks from a fired origin are nearby one another.  In asynchronous or mid/late S-phase cells, the difference is likely to be larger as the forks from a single origin will have travelled some distance before the BrdU pulse.  The bed files only specify the region between matching leftward- and rightward-moving forks.  Any subsequent assumptions (such as assuming uniform fork speed and placing the origin in the middle of that region) are left to the user.

The output of ``DNAscent forkSense`` is a file with similar formatting to that of ``DNAscent detect``.  The format for the read headers is the same.  From left to right, the tab-delimited columns indicate:

* the coordinate on the reference,
* probability that a leftward-moving fork passed through that coordinate during a BrdU pulse,
* probability that a rightward-moving fork passed through that coordinate during a BrdU pulse.
