#
#  file:  rot_tsne_plots.py
#
#  Compare rotation features and raw features
#
#  RTK, 14-Jan-2024
#  Last update:  15-Jan-2024
#
################################################################

import os
import sys
import numpy as np
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

haveDim = True
try:
    from skdim.id import KNN, MLE
except:
    haveDim = False


if (len(sys.argv) == 1):
    print()
    print("rot_tsne_plots <base_dir> [show]")
    print()
    print("  <base_dir> - base directory with raw.keras and model.keras")
    print("  show       - show the plots (plots saved regardless)")
    print()
    exit(0)

bdir = sys.argv[1]
show = True if (len(sys.argv) == 3) else False

#  Extract the features
os.system("python3 rot_extract_features.py %s/raw.keras 3 /tmp/raw 2>/dev/null" % bdir)
os.system("python3 rot_extract_features.py %s/model.keras 3 /tmp/pre 2>/dev/null" % bdir)

#  Load the extracted features
raw = np.load("/tmp/raw_cifar10_features_test.npy")
pre = np.load("/tmp/pre_cifar10_features_test.npy")
lbl = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

#  If skdim available, estimate the ID of the datasets
if (haveDim):
    z = np.load("../data/cifar10/cifar10_test_images.npy").reshape(10000,32*32*3) / 255
    org_id = int(MLE().fit_transform(z) + 0.5)
    pre_id = int(MLE().fit_transform(pre) + 0.5)
    raw_id = int(MLE().fit_transform(raw) + 0.5)

#  Select a subset for plotting (with labels)
np.random.seed(68040)  # fixed here and for t-SNE and k-means !!!
idx = np.argsort(np.random.random(len(lbl)))[:1000]
raw, pre, lbl = raw[idx], pre[idx], lbl[idx]

raw_t = TSNE(n_components=2).fit_transform(raw)
pre_t = TSNE(n_components=2).fit_transform(pre)

#  Clusters
kr = KMeans(n_clusters=10).fit_predict(raw_t)
kp = KMeans(n_clusters=10).fit_predict(pre_t)

markers = ['o','s','^','v','P','<','>','*','D','X']*2

for i in range(len(raw_t)):
    plt.plot(raw_t[i,0], raw_t[i,1], marker=markers[lbl[i]], fillstyle='none', linestyle='none', color='k')
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("rot_tsne_raw_plot.eps", dpi=300)
plt.savefig("rot_tsne_raw_plot.png", dpi=300)
if (show):  plt.show()
plt.close()

for i in range(len(pre_t)-10):
    plt.plot(pre_t[i,0], pre_t[i,1], marker=markers[lbl[i]], fillstyle='none', linestyle='none', color='k')
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("rot_tsne_pretrained_plot.eps", dpi=300)
plt.savefig("rot_tsne_pretrained_plot.png", dpi=300)
if (show):  plt.show()
plt.close()

#  Calculate the purity
sr,sp = [],[]

#  Labels for each cluster
for i in range(10):
    idx = np.where(kr==i)[0];  sr.append(lbl[idx])
    idx = np.where(kp==i)[0];  sp.append(lbl[idx])

#  Number of samples in each cluster belonging to the dominant class
scr = scp = 0
for i in range(10):
    scr += np.bincount(sr[i], minlength=10).max()
    scp += np.bincount(sp[i], minlength=10).max()

#  Purity is the fraction of total samples correctly assigned
#  to the dominant class
N = len(raw_t)
print()
print("Purity:")
print("    raw features        : %0.4f" % (scr / N))
print("    pretrained features : %0.4f" % (scp / N))
print()

if (haveDim):
    print("Intrinsic dimension:")
    print("    raw features        : %d" % raw_id)
    print("    pretrained features : %d" % pre_id)
    print("    original features   : %d" % org_id)
    print()

