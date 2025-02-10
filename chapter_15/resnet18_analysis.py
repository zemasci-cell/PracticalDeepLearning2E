#
#  file:  resnet18_analysis.py
#
#  Analysis of resne18_runs results
#
#  RTK, 19-Aug-2023
#  Last update:  19-Aug-2023
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
        base = "runs/resnet_b128_e10_%s_run%d" % (bname,i)
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
    plt.savefig("resnet18_%s_plot.png" % bname, dpi=300)
    plt.savefig("resnet18_%s_plot.eps" % bname, dpi=300)
    plt.close()
    return acc, mcc, ve,vstd

a0,m0,v0,vs0 = Summarize("No batch normalization", "no_bn")
a1,m1,v1,vs1 = Summarize("Batch normalization", "bn")

t,p = ttest_ind(a1,a0);  _,u = mannwhitneyu(a1,a0)
print("Batch vs no batch: (t=%0.5f,p=%0.5f)(MW=%0.5f)" % (t,p,u))

n = np.arange(len(v0))
plt.errorbar(n+0.00,v1,vs1, marker='o', linewidth=0.7, fillstyle='none', color='k', label='batch')
plt.errorbar(n+0.00,v0,vs0, marker='s', linewidth=0.7, fillstyle='none', color='k', label='no batch')
plt.ylim((0,0.7))
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("resnet18_summary_plot.png", dpi=300)
plt.savefig("resnet18_summary_plot.eps", dpi=300)
plt.close()

