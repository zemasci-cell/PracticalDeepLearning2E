#
#  file:  compare_embeddings.py
#
#  Compare model embeddings
#
#  RTK, 25-Feb-2024
#  Last update:  25-Feb-2024
#
################################################################

import numpy as np
import os
import sys

def cosine(a,b):
    """Calculate the cosine distance between two vectors"""
    num = np.dot(a,b)
    den = np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b))
    return 1.0 - num/den

if (len(sys.argv) == 1):
    print()
    print("compare_embeddings <emb0> <emb1> [<outfile>]")
    print()
    print("  <emb0> - a set of embeddings assumed to be related (.npy)")
    print("  <emb1> - a second set of embeddings")
    print("  <outfile> - name for the output cosine distance matrix (.npy)")
    print()
    exit(0)

emb0 = np.load(sys.argv[1])
emb1 = np.load(sys.argv[2])
outfile = sys.argv[3] if (len(sys.argv) == 4) else ""

dist = np.zeros((len(emb0), len(emb1)))
for i in range(len(emb0)):
    for j in range(len(emb1)):
        dist[i,j] = cosine(emb0[i], emb1[j])

if (outfile != ""):
    np.save(outfile, dist)

print("Distance matrix:")
print()
print(dist)
print()
print("Mean cosine distance = %0.8f +/- %0.8f" % (dist.mean(), dist.std(ddof=1)/np.sqrt(len(dist))))
print()

