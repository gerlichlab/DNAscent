# DNAscent Test - HMM False Positives

This test checks whether false positives are correlated with the number of events passed to the HMM, properties of the reference sequence at that point, or the span on the query sequence.

## Files

`testFalsePositives.py`

## Running

For a sample with no BrdU, run DNAscent detect on a single thread with #define TEST_LL 1 and stderr redirected to a file (call it eventTest.txt).  Run `python testFalsePositives.py eventTest.txt`. 
