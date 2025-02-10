#
#  file:  mobilenet_analysis.py
#
#  Analysis of mobilenet_runs results
#
#  RTK, 21-Aug-2023
#  Last update:  21-Aug-2023
#
################################################################

import numpy as np
import matplotlib.pylab as plt
import pickle

terr = np.zeros((5,10))
verr = np.zeros((5,10))
cm = np.zeros((5,10,10))
acc = np.zeros(5)
mcc = np.zeros(5)

for i in range(5):
    base = "runs/mobilenet_b128_e10_run%d/" % i
    _, _, terr[i,:], verr[i,:] = pickle.load(open(base+"results.pkl","rb"))
    cm[i,:,:] = np.load(base+"confusion_matrix.npy")
    t = open(base+"accuracy_mcc.txt").read().split()
    acc[i], mcc[i] = float(t[3][:-1]), float(t[5])

cc = np.round(cm.mean(axis=0)).astype("uint16")
ac,ae = acc.mean(), acc.std(ddof=1) / np.sqrt(5)
mc,me = mcc.mean(), mcc.std(ddof=1) / np.sqrt(5)
te, tstd = terr.mean(axis=0), terr.std(ddof=1, axis=0) / np.sqrt(5)
ve, vstd = verr.mean(axis=0), verr.std(ddof=1, axis=0) / np.sqrt(5)
print("Mobilenet:")
print("ACC: %0.5f+/-%0.5f,  MCC: %0.5f+/-%0.5f" % (ac,ae,mc,me))
print(cc)
print()
n = list(range(len(te)))
plt.errorbar(n,ve,vstd, marker='o', linewidth=0.7, fillstyle='none', color='k')
plt.ylim((0,0.7))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("mobilenet_plot.png", dpi=300)
plt.savefig("mobilenet_plot.eps", dpi=300)
plt.close()

