.. _regions:

Regions
===============================

``DNAscent regions`` is a ``DNAscent`` subprogram that interprets the output of ``DNAscent detect`` to call regions of high and low BrdU incorporation.

Note that as of v2.0, ``DNAscent regions`` has been largely superceded by ``DNAscent forkSense`` and the increased accuracy of ``DNAscent detect`` makes visualising BrdU incorporation in regions mostly unnecessary.  However, it is still included to avoid breaking legacy workflows, and it does still have some uses as explained below.

Usage
-----

.. code-block:: console

   To run DNAscent regions, do:
      DNAscent regions -d /path/to/DNAscentOutput.detect -o /path/to/DNAscentOutput.regions
   Required arguments are:
     -d,--detect               path to output file from DNAscent detect,
     -o,--output               path to output directory for bedgraph files.
   Optional arguments (if used with default ResNet-based detect) are:
     -r,--resolution           number of thymidines in a region (default is 10).
   Optional arguments (if used with HMM-based detect) are:
      --replication            detect fork direction and call origin firing (default: off),
     -l,--likelihood           log-likelihood threshold for a positive analogue call (default: 1.25),
     -c,--cooldown             minimum gap between positive analogue calls (default: 4),
     -r,--resolution           minimum length of regions (default is 2kb),
     -p,--probability          override probability that a thymidine 6mer contains a BrdU (default: automatically calculated),
     -z,--zScore               override zScore threshold for BrdU call (default: automatically calculated).

The only required input of ``DNAscent regions`` is the output file produced by ``DNAscent detect``.  It is important to note that ``DNAscent regions`` is polymorphic: Its usage and output depends on the type of detection file it is passed.

ResNet-based Detection (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current, default, and recommended algorithm for BrdU detection in ``DNAscent detect`` uses a ResNet to evaluate the probability of BrdU at each thymidine.  When the output detect file is passed to ``DNAscent regions``, these probabilities are averaged over consecutive windows.  By default, the width of these windows is 10 thymidines, but this can be changed with the ``-r`` flag.

HMM-based Detection (Legacy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``DNAscent regions`` is passed a detect file where HMM-based detection was used, it acts as in v1.0 and earlier: It will first look through the detect file and determine the approximate fraction of thymidines replaced by BrdU in BrdU-positive regions.  Using this probability, a z-score is assigned to each window (2 kb wide by default, but this can be changed using the ``-r`` flag) to indicate whether there is more or less BrdU than would be expected for an average BrdU-positive region.  Naturally, some regions will be BrdU-positive but will have a substitution rate lower than average for BrdU-positive regions. Hence, ``DNAscent regions`` determines an appropriate boundary threshold between BrdU-positive regions and thymidine regions and rescales all of the z-scores so that this boundary is 0. ``DNAscent regions`` will calculate these values for you, but they can be overridden with the  ``-p`` and ``-z`` flags, though this is generally not recommended.  The exceptions are runs with 0% BrdU or runs where a high BrdU incorporation is expected along the entirety of each read. This is because these parameters are computed assuming that there are two populations (BrdU-positive and thymidine-only segments of DNA).

In order to determine regions of high and low BrdU incorporation, ``DNAscent regions`` needs to count positive BrdU calls.  By default, a 6mer is considered to have BrdU incorporated if the log-likelihood of incorporation exceeds 1.25.  This value was tuned in-house to optimise signal-to-noise, but it can be changed with the ``-l`` flag.  Likewise, some care has to be given to how positive calls are counted.  In the example sequence AGCCATTGCAAC, the 6mers TTGCAA and TGCAAC will both be assessed by ``DNAscent detect``.  If only one of these Ts is a BrdU, their proximity means that both 6mers may flag as positive calls.  To prevent artefacts from overcounting while minimising undercounting, the default behaviour is to only make a positive call every 4 bases, though this can be changed with the ``-c`` flag.


Output
------

The output of DNAscent regions is a file with similar formatting to that of ``DNAscent detect``.  The format for the read headers is the same.  From left to right, the tab-delimited columns indicate:

* the start of the region,
* the end of the region,
* the BrdU score,
* (for HMM-based detection only) ``BrdU`` if the score is positive and ``Thym`` if the score is negative.

ResNet-based Detection (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``DNAscent regions`` is run on a ResNet-based detection file, the BrdU score in the third column will be an average probability.  An example output is as follows:

.. code-block:: console

   >2f8ef2ca-a472-4849-8c2a-966e578aa94b chrI 92777 99331 rev
   92779   92801   0.0681856
   92801   92834   0.087161
   92834   92850   0.0615494
   92850   92894   0.0786939
   92894   92913   0.0909456


HMM-based Detection (Legacy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``DNAscent regions`` is run on a ResNet-based detection file, the BrdU score in the third column will be a z-score.  A large positive score indicates high BrdU incorporation in that region, and a large negative score indicates very little BrdU incorporation in that region.  An example output is as follows:

.. code-block:: console

   >3ded31e2-a211-4480-9d0c-101cad9b4584 chrIV 798194 843752 rev
   798209  800210  0.00667184      BrdU
   800213  802215  5.36575 BrdU
   802218  804219  8.20312 BrdU
   804224  806225  9.23163 BrdU
   806228  808236  7.1194  BrdU
   808237  810238  7.62854 BrdU
   810239  812240  8.0111  BrdU
   812241  814248  2.75055 BrdU
   814252  816256  2.08421 BrdU
   816261  818264  -0.285431       Thym

Note that the region width may sometimes vary slightly from the value specified. The region width is designated as the coordinate of the first thymidine greater than the window width (2 kb by default) from the starting coordinate.  In order to guard against assigning a score to regions with very few thymidines, ``DNAscent regions`` will also extend the region until at least 30 calls are considered.
