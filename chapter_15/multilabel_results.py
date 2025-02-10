#
#  file:  multilabel_results.py
#
#  Interpret the output of multilabel.py
#
#  RTK, 03-Jan-2024
#  Last update:  04-Jan-2024
#
################################################################

import os
import sys
import numpy as np
import matplotlib.pylab as plt

if (len(sys.argv)==1):
    print()
    print("multilabel_results <pred> <outdir>")
    print()
    print("  <pred>   - predictions")
    print("  <outdir> - output directory")
    print()
    exit(0)

ytest = np.load("../data/mnist/mnist_multilabel_ytest.npy")
pred = np.load(sys.argv[1])
outdir = sys.argv[2]

os.system("mkdir %s 2>/dev/null" % outdir)
os.system("rm -rf %s/plots 2>/dev/null" % outdir)
os.system("mkdir %s/plots 2>/dev/null" % outdir)

np.random.seed(6502)

threshold = 0.1  # accept above this as digit present

#  Generate select plots
for i in range(len(ytest)):
    #  Plot about 2 percent
    if (np.random.random() < 0.02):
        for j in range(10):
            if (ytest[i,j]):
                plt.bar(j,pred[i,j], fill=True, color='k')
            else:
                plt.bar(j,pred[i,j], fill=False, color='k')
        plt.ylim((0,1))
        plt.xticks(range(10), [str(i) for i in range(10)])
        ax = plt.gca()
        for j,label in enumerate(ax.get_xticklabels()):
            if (ytest[i,j]):
                label.set_weight('bold')
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
        plt.savefig(outdir+("/plots/sample_%04d.png" % i), dpi=300)
        plt.close()
    
#  Gather stats
per_digit = np.zeros((10,2))  # row=digit, 0=correctly detected count, 1=presence count
correct = 0
for i in range(len(ytest)):
    pdigits = sorted(np.where(pred[i] > threshold)[0])
    digits = sorted(np.where(ytest[i] > 0)[0])
    for digit in digits:
        if (digit in pdigits):
            per_digit[digit,0] += 1
        per_digit[digit,1] += 1
    if (pdigits == digits):
        correct += 1

print("Correct detection of all digits: %0.5f (threshold=%0.2f)" % (correct/len(ytest),threshold))
print()
print("Correct detection by digit:")
v = 0.0
for i in range(10):
    v += per_digit[i,0] / per_digit[i,1]
    print("    %d: %0.5f" % (i, per_digit[i,0] / per_digit[i,1]))
print(" mean: %0.5f" % (v/10,))
print()

