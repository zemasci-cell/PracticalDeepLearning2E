#
#  file:  all_analysis.py
#
#  Compare the best version of each model
#
#  RTK, 21-Aug-2023
#  Last update:  21-Aug-2023
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
        base = "%s%d" % (bname,i)
        _, _, terr[i,:], verr[i,:] = pickle.load(open(base+"/results.pkl","rb"))
        cm[i,:,:] = np.load(base+"/confusion_matrix.npy")
        t = open(base+"/accuracy_mcc.txt").read().split()
        acc[i], mcc[i] = float(t[3][:-1]), float(t[5])

    cc = np.round(cm.mean(axis=0)).astype("uint16")
    ac,ae = acc.mean(), acc.std(ddof=1) / np.sqrt(5)
    mc,me = mcc.mean(), mcc.std(ddof=1) / np.sqrt(5)
    te, tstd = terr.mean(axis=0), terr.std(ddof=1, axis=0) / np.sqrt(5)
    ve, vstd = verr.mean(axis=0), verr.std(ddof=1, axis=0) / np.sqrt(5)
    print("%s:" % title)
    print("ACC: %0.5f+/-%0.5f,  MCC: %0.5f+/-%0.5f" % (ac,ae,mc,me))
    print(cc)
    print()
    return acc, mcc, ve,vstd

a0,m0,v0,vs0 = Summarize("ResNet-18", "runs/resnet_b128_e10_bn_run")
a1,m1,v1,vs1 = Summarize("VGG8", "runs/vgg_b128_e10_batchnorm_run")
a2,m2,v2,vs2 = Summarize("MobileNet", "runs/mobilenet_b128_e10_run")

t,p = ttest_ind(a0,a1); _,u = mannwhitneyu(a0,a1)
print("ResNet-18 vs VGG8     : (t=%0.5f,p=%0.5f,MW=%0.5f)" % (t,p,u))
t,p = ttest_ind(a0,a2); _,u = mannwhitneyu(a0,a2)
print("ResNet-18 vs MobileNet: (t=%0.5f,p=%0.5f,MW=%0.5f)" % (t,p,u))
t,p = ttest_ind(a1,a2); _,u = mannwhitneyu(a1,a2)
print("VGG8 vs MobileNet     : (t=%0.5f,p=%0.5f,MW=%0.5f)" % (t,p,u))

n = np.arange(len(v0))
plt.errorbar(n+0.06,v2,vs2, marker='^', linewidth=0.7, fillstyle='none', color='k', label='MobileNet')
plt.errorbar(n+0.00,v0,vs0, marker='o', linewidth=0.7, fillstyle='none', color='k', label='ResNet-18')
plt.errorbar(n+0.03,v1,vs1, marker='s', linewidth=0.7, fillstyle='none', color='k', label='VGG8')
plt.ylim((0,1.0))
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("all_summary_plot.png", dpi=300)
plt.savefig("all_summary_plot.eps", dpi=300)
plt.close()

