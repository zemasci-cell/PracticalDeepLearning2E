#
#  file: rot_fine-tune_results.py
#
#  Assess the output of go_rot_fine-tune
#
#  RTK, 08-Jan-2024
#  Last update:  09-Jan-2024
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
blk2_100 = Stats("results/rot_blk2_128_12_100_run")
blk3_100 = Stats("results/rot_blk3_128_12_100_run")
sgd3_100 = Stats("results/rot_blk3_sgd_128_12_100_run")
scrt_100 = Stats("results/rot_scratch_128_12_100_run")
print("100 percent:")
print("    blk2: %9.6f +/- %9.6f" % (blk2_100.mean(), blk2_100.std(ddof=1)/np.sqrt(6)))
print("    blk3: %9.6f +/- %9.6f" % (blk3_100.mean(), blk3_100.std(ddof=1)/np.sqrt(6)))
print(" scratch: %9.6f +/- %9.6f" % (scrt_100.mean(), scrt_100.std(ddof=1)/np.sqrt(6)))
t,p = ttest_ind(blk2_100, blk3_100); _,u = mannwhitneyu(blk2_100, blk3_100)
print(" blk2 vs blk3   : (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
t,p = ttest_ind(blk2_100, scrt_100); _,u = mannwhitneyu(blk2_100, scrt_100)
print(" blk2 vs scratch: (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
t,p = ttest_ind(blk3_100, scrt_100); _,u = mannwhitneyu(blk3_100, scrt_100)
print(" blk3 vs scratch: (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
print()

#  10 percent
blk2_010 = Stats("results/rot_blk2_128_120_010_run")
blk3_010 = Stats("results/rot_blk3_128_120_010_run")
sgd3_010 = Stats("results/rot_blk3_sgd_128_120_010_run")
scrt_010 = Stats("results/rot_scratch_128_120_010_run")
print("010 percent:")
print("    blk2: %9.6f +/- %9.6f" % (blk2_010.mean(), blk2_010.std(ddof=1)/np.sqrt(6)))
print("    blk3: %9.6f +/- %9.6f" % (blk3_010.mean(), blk3_010.std(ddof=1)/np.sqrt(6)))
print(" scratch: %9.6f +/- %9.6f" % (scrt_010.mean(), scrt_010.std(ddof=1)/np.sqrt(6)))
t,p = ttest_ind(blk2_010, blk3_010); _,u = mannwhitneyu(blk2_010, blk3_010)
print(" blk2 vs blk3   : (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
t,p = ttest_ind(blk2_010, scrt_010); _,u = mannwhitneyu(blk2_010, scrt_010)
print(" blk2 vs scratch: (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
t,p = ttest_ind(blk3_010, scrt_010); _,u = mannwhitneyu(blk3_010, scrt_010)
print(" blk3 vs scratch: (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
print()

#  5 percent
blk2_005 = Stats("results/rot_blk2_128_240_005_run")
blk3_005 = Stats("results/rot_blk3_128_240_005_run")
sgd3_005 = Stats("results/rot_blk3_sgd_128_240_005_run")
scrt_005 = Stats("results/rot_scratch_128_240_005_run")
print("005 percent:")
print("    blk2: %9.6f +/- %9.6f" % (blk2_005.mean(), blk2_005.std(ddof=1)/np.sqrt(6)))
print("    blk3: %9.6f +/- %9.6f" % (blk3_005.mean(), blk3_005.std(ddof=1)/np.sqrt(6)))
print(" scratch: %9.6f +/- %9.6f" % (scrt_005.mean(), scrt_005.std(ddof=1)/np.sqrt(6)))
t,p = ttest_ind(blk2_005, blk3_005); _,u = mannwhitneyu(blk2_005, blk3_005)
print(" blk2 vs blk3   : (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
t,p = ttest_ind(blk2_005, scrt_005); _,u = mannwhitneyu(blk2_005, scrt_005)
print(" blk2 vs scratch: (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
t,p = ttest_ind(blk3_005, scrt_005); _,u = mannwhitneyu(blk3_005, scrt_005)
print(" blk3 vs scratch: (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
print()

#  1 percent
blk2_001 = Stats("results/rot_blk2_128_240_001_run")
blk3_001 = Stats("results/rot_blk3_128_240_001_run")
sgd3_001 = Stats("results/rot_blk3_sgd_128_240_001_run")
scrt_001 = Stats("results/rot_scratch_128_240_001_run")
print("001 percent:")
print("    blk2: %9.6f +/- %9.6f" % (blk2_001.mean(), blk2_001.std(ddof=1)/np.sqrt(6)))
print("    blk3: %9.6f +/- %9.6f" % (blk3_001.mean(), blk3_001.std(ddof=1)/np.sqrt(6)))
print(" scratch: %9.6f +/- %9.6f" % (scrt_001.mean(), scrt_001.std(ddof=1)/np.sqrt(6)))
t,p = ttest_ind(blk2_001, blk3_001); _,u = mannwhitneyu(blk2_001, blk3_001)
print(" blk2 vs blk3   : (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
t,p = ttest_ind(blk2_001, scrt_001); _,u = mannwhitneyu(blk2_001, scrt_001)
print(" blk2 vs scratch: (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
t,p = ttest_ind(blk3_001, scrt_001); _,u = mannwhitneyu(blk3_001, scrt_001)
print(" blk3 vs scratch: (t=%9.6f, p=%9.6f, u=%9.6f)" % (t,p,u))
print()

#  Plot
x = [1,2,3,4]
y = [blk2_001.mean(),blk2_005.mean(),blk2_010.mean(),blk2_100.mean()]
e = np.array([blk2_001.std(ddof=1),blk2_005.std(ddof=1),blk2_010.std(ddof=1),blk2_100.std(ddof=1)])/np.sqrt(6)
plt.errorbar(x,y,e, marker='o', fillstyle='none', linewidth=0.7, color='k', label='block 2')

y = [blk3_001.mean(),blk3_005.mean(),blk3_010.mean(),blk3_100.mean()]
e = np.array([blk3_001.std(ddof=1),blk3_005.std(ddof=1),blk3_010.std(ddof=1),blk3_100.std(ddof=1)])/np.sqrt(6)
plt.errorbar(x,y,e, marker='^', fillstyle='none', linewidth=0.7, color='k', label='block 3')

y = [scrt_001.mean(),scrt_005.mean(),scrt_010.mean(),scrt_100.mean()]
e = np.array([scrt_001.std(ddof=1),scrt_005.std(ddof=1),scrt_010.std(ddof=1),scrt_100.std(ddof=1)])/np.sqrt(6)
plt.errorbar(x,y,e, marker='s', fillstyle='none', linewidth=0.7, color='k', label='scratch')

y = [sgd3_001.mean(),sgd3_005.mean(),sgd3_010.mean(),sgd3_100.mean()]
e = np.array([sgd3_001.std(ddof=1),sgd3_005.std(ddof=1),sgd3_010.std(ddof=1),sgd3_100.std(ddof=1)])/np.sqrt(6)
plt.errorbar(x,y,e, marker='^', linewidth=0.7, color='k', label='block 3 (SGD)')

#  1 percent augmented - Adam optimizer - 32 minibatch, 60 epochs -- go_rot_fine-tune_augment
adam = np.array([0.4885,0.4795,0.4873,0.5021,0.4896,0.4850])
plt.plot([1],[adam.mean()], marker='*', linestyle='none', fillstyle='none', color='k', label='aug 3 (Adam)')

#  1 percent augmented - SGD, lr=0.005 - 32 minibatch, 60 epochs -- go_rot_fine-tune_augment
sgd = np.array([0.5158,0.5136,0.5181,0.5138,0.5201,0.5163])
plt.plot([1],[sgd.mean()], marker='*', linestyle='none', color='k', label='aug 3 (SGD)')

plt.xticks(x,["1","5","10","100"])
plt.xlabel("Training percent")
plt.ylabel("Mean accuracy ($n=6$)")
plt.legend(loc='best')
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("rot_fine-tune_results_plot.png", dpi=300)
plt.savefig("rot_fine-tune_results_plot.eps", dpi=300)
plt.show()

