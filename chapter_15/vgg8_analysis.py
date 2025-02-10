#
#  file:  vgg8_analysis.py
#
#  Analysis of vgg8_runs results
#
#  RTK, 17-Aug-2023
#  Last update:  17-Aug-2023
#
################################################################

import numpy as np
import matplotlib.pylab as plt
import pickle
from scipy.stats import ttest_ind, mannwhitneyu

def Summarize(title, bname):
    """Summarize a test"""
    terr = np.zeros((5,10))
    verr = np.zeros((5,10))
    cm = np.zeros((5,10,10))
    acc = np.zeros(5)
    mcc = np.zeros(5)

    for i in range(5):
        base = "runs/vgg_b128_e10_%s_run%d" % (bname,i)
        _, _, terr[i,:], verr[i,:] = pickle.load(open(base+"/results.pkl","rb"))
        cm[i,:,:] = np.load(base+"/confusion_matrix.npy")
        t = open(base+"/accuracy_mcc.txt").read().split()
        acc[i], mcc[i] = float(t[3][:-1]), float(t[5])  # [:-1] removes the ','

    cc = np.round(cm.mean(axis=0)).astype("uint16")
    ac,ae = acc.mean(), acc.std(ddof=1) / np.sqrt(5)
    mc,me = mcc.mean(), mcc.std(ddof=1) / np.sqrt(5)
    te, tstd = terr.mean(axis=0), terr.std(ddof=1, axis=0) / np.sqrt(5)
    ve, vstd = verr.mean(axis=0), verr.std(ddof=1, axis=0) / np.sqrt(5)
    print("%s:" % title)
    print("ACC: %0.5f+/-%0.5f,  MCC: %0.5f+/-%0.5f" % (ac,ae,mc,me))
    print(cc)
    print()
    n = list(range(len(te)))
    plt.errorbar(n,te,tstd, marker='s', linewidth=0.7, fillstyle='none', color='k', label='train')
    plt.errorbar(n,ve,vstd, marker='o', linewidth=0.7, fillstyle='none', color='k', label='test')
    plt.legend(loc='best')
    plt.ylim((0,0.8))
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig("vgg8_%s_plot.png" % bname, dpi=300)
    plt.savefig("vgg8_%s_plot.eps" % bname, dpi=300)
    plt.close()
    return acc, mcc, ve,vstd

a0,m0,v0,vs0 = Summarize("Batch normalization", "batchnorm")
a1,m1,v1,vs1 = Summarize("No dropout", "dropout0")
a2,m2,v2,vs2 = Summarize("Standard dropout", "dropout1")
a3,m3,v3,vs3 = Summarize("Spatial dropout", "dropout2")

t,p = ttest_ind(a0,a1);  _,u = mannwhitneyu(a0,a1, method='exact')
print("Batch vs no dropout   : (t=%0.5f,p=%0.5f)(MW=%0.5f)" % (t,p,u))
t,p = ttest_ind(a1,a2);  _,u = mannwhitneyu(a1,a2, method='exact')
print("No dropout vs standard: (t=%0.5f,p=%0.5f)(MW=%0.5f)" % (t,p,u))
t,p = ttest_ind(a1,a3);  _,u = mannwhitneyu(a1,a3, method='exact')
print("No dropout vs spatial : (t=%0.5f,p=%0.5f)(MW=%0.5f)" % (t,p,u))

n = np.arange(len(v0))
plt.errorbar(n+0.00,v0,vs0, marker='o', linewidth=0.7, fillstyle='none', color='k', label='batch')
plt.errorbar(n+0.05,v1,vs1, marker='s', linewidth=0.7, fillstyle='none', color='k', label='no drop')
plt.errorbar(n+0.10,v2,vs2, marker='^', linewidth=0.7, fillstyle='none', color='k', label='classic')
plt.errorbar(n+0.15,v3,vs3, marker='p', linewidth=0.7, fillstyle='none', color='k', label='spatial')
plt.ylim((0,0.7))
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("vgg8_summary_plot.png", dpi=300)
plt.savefig("vgg8_summary_plot.eps", dpi=300)
plt.close()

