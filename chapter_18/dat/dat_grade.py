#
#  file:  dat_grade.py
#
#  Calculate the DAT for a text file of word lists
#  (based on examples.py in the DAT repo)
#
#  RTK, 26-Feb-2024
#  Last update:  26-Feb-2024
#
################################################################

import os
import sys
import dat  # DAT module
import numpy as np

if (len(sys.argv) == 1):
    print()
    print("dat_grade <words>")
    print()
    print("  <words> - file of words (one run per line)")
    print()
    exit(0)

#  Configure DAT (from examples.py):

# GloVe model from https://nlp.stanford.edu/projects/glove/
model = dat.Model("glove.840B.300d.txt", "words.txt")

#  Load and score each list
lines = [i[:-1] for i in open(sys.argv[1])]
scores = []
allwords = []

for line in lines:
    if (line == ""):
        continue
    words = line.split(",")
    words = [i.strip() for i in words]
    if (len(words) != 10):
        print("error: encountered an error in: %s" % line)
    else:
        allwords += words
        try:
            sc = model.dat(words)
        except:
            sc = 0.0
        if (sc is None):
            sc = 0.0
        scores.append(sc)

scores = np.array(scores)
allwords = list(set(allwords))

for score in scores:
    print("%0.2f " % score, end="")
print("%0.4f %0.4f" % (scores.mean(), scores.std(ddof=1) / np.sqrt(len(scores))), end="")
print(" (%d unique words)" % len(allwords))

