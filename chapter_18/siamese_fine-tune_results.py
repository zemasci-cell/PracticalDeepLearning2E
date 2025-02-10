#
#  file: siamese_fine-tune_results.py
#
#  Assess the output of go_siamese_fine-tune
#
#  RTK, 24-Jan-2024
#  Last update:  26-Jan-2024
#
################################################################

import numpy as np
import matplotlib.pylab as plt
from scipy.stats import ttest_ind, mannwhitneyu

def Load(fname):
    t = open(fname).read()[:-1]
    return float(t.split()[-3][:-1])

def Stats(path):
    acc = np.zeros(6)
    for i in range(6):
        acc[i] = Load(path+("%d" % i)+"/accuracy_mcc.txt")
    return acc

#  100 percent
b3_100 = Stats("results/rot_blk3_128_12_100_run")
s1_100 = Stats("results/siamese_blk3_128_12_100_run")
s2_100 = Stats("results/siamese2_blk3_128_12_100_run")
sc_100 = Stats("results/rot_scratch_128_12_100_run")

#  10 percent
b3_010 = Stats("results/rot_blk3_128_120_010_run")
s1_010 = Stats("results/siamese_blk3_128_120_010_run")
s2_010 = Stats("results/siamese2_blk3_128_120_010_run")
sc_010 = Stats("results/rot_scratch_128_120_010_run")

#  5 percent
b3_005 = Stats("results/rot_blk3_128_240_005_run")
s1_005 = Stats("results/siamese_blk3_128_240_005_run")
s2_005 = Stats("results/siamese2_blk3_128_240_005_run")
sc_005 = Stats("results/rot_scratch_128_240_005_run")

#  1 percent
b3_001 = Stats("results/rot_blk3_128_240_001_run")
s1_001 = Stats("results/siamese_blk3_128_240_001_run")
s2_001 = Stats("results/siamese2_blk3_128_240_001_run")
sc_001 = Stats("results/rot_scratch_128_240_001_run")

#  Plot
x = [1,2,3,4]
y = [b3_001.mean(),b3_005.mean(),b3_010.mean(),b3_100.mean()]
e = np.array([b3_001.std(ddof=1),b3_005.std(ddof=1),b3_010.std(ddof=1),b3_100.std(ddof=1)])/np.sqrt(6)
plt.errorbar(x,y,e, marker='^', fillstyle='none', linewidth=0.7, color='k', label='block 3')

y = [s1_001.mean(),s1_005.mean(),s1_010.mean(),s1_100.mean()]
e = np.array([s1_001.std(ddof=1),s1_005.std(ddof=1),s1_010.std(ddof=1),s1_100.std(ddof=1)])/np.sqrt(6)
plt.errorbar(x,y,e, marker='o', fillstyle='none', linewidth=0.7, color='k', label='siamese')

y = [s2_001.mean(),s2_005.mean(),s2_010.mean(),s2_100.mean()]
e = np.array([s2_001.std(ddof=1),s2_005.std(ddof=1),s2_010.std(ddof=1),s2_100.std(ddof=1)])/np.sqrt(6)
plt.errorbar(x,y,e, marker='d', fillstyle='none', linewidth=0.7, color='k', label='siamese2')

y = [sc_001.mean(),sc_005.mean(),sc_010.mean(),sc_100.mean()]
e = np.array([sc_001.std(ddof=1),sc_005.std(ddof=1),sc_010.std(ddof=1),sc_100.std(ddof=1)])/np.sqrt(6)
plt.errorbar(x,y,e, marker='s', fillstyle='none', linewidth=0.7, color='k', label='scratch')

plt.xticks(x,["1","5","10","100"])
plt.xlabel("Training percent")
plt.ylabel("Mean accuracy ($n=6$)")
plt.legend(loc='best')
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("siamese_fine-tune_results_plot.png", dpi=300)
plt.savefig("siamese_fine-tune_results_plot.eps", dpi=300)
plt.show()

