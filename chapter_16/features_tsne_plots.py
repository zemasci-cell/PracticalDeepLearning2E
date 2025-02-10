#
#  files: features_tsne_plots.py
#
#  Visualize the embedded features
#
#  RTK, 14-Dec-2023
#  Last update:  17-Dec-2023
#
################################################################

import sys
import numpy as np
import matplotlib.pylab as plt 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

x_rtrain = np.load("../data/cifar10/cifar10_train_images.npy").reshape((50000,3072))
x_rtest = np.load("../data/cifar10/cifar10_test_images.npy").reshape((10000,3072))

x_btrain = np.load("features_cifar10_block_train.npy")
x_btest = np.load("features_cifar10_block_test.npy")

x_etrain = np.load("features_cifar10_train.npy")
x_etest = np.load("features_cifar10_test.npy")

y_train = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()
y_test = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

#  Choose M test samples per class
M = 20
np.random.seed(8080)
rsamples = np.zeros((M*10,3072))
esamples = np.zeros((M*10,512))
bsamples = np.zeros((M*10,4096))
for i in range(10):
    idx = np.where(y_test==i)[0]
    t = np.argsort(np.random.random(len(idx)))[:M]
    idx = idx[t]
    rsamples[(i*M):(i*M+M),:] = x_rtest[idx]
    esamples[(i*M):(i*M+M),:] = x_etest[idx]
    bsamples[(i*M):(i*M+M),:] = x_btest[idx]

ysamples = []
for i in range(10):
    ysamples = ysamples + [i]*M
ysamples = np.array(ysamples)

#  t-SNE plots
markers = ['o','s','^','v','P','<','>','*','D','X']
rr = TSNE(n_components=2).fit_transform(rsamples)
ee = TSNE(n_components=2).fit_transform(esamples)
bb = TSNE(n_components=2).fit_transform(bsamples)

for i in range(10):
    plt.plot(rr[(i*M):(i*M+M),0], rr[(i*M):(i*M+M),1], marker=markers[i], fillstyle='none', color='k', linestyle='none', label=str(i))
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("features_tsne_raw_plot.eps", dpi=300)
plt.savefig("features_tsne_raw_plot.png", dpi=300)
plt.close()

for i in range(10):
    plt.plot(ee[(i*M):(i*M+M),0], ee[(i*M):(i*M+M),1], marker=markers[i], fillstyle='none', color='k', linestyle='none', label=str(i))
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("features_tsne_embedded_plot.eps", dpi=300)
plt.savefig("features_tsne_embedded_plot.png", dpi=300)
plt.close()

for i in range(10):
    plt.plot(bb[(i*M):(i*M+M),0], bb[(i*M):(i*M+M),1], marker=markers[i], fillstyle='none', color='k', linestyle='none', label=str(i))
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("features_tsne_block_plot.eps", dpi=300)
plt.savefig("features_tsne_block_plot.png", dpi=300)
plt.close()

#  Clustering
krr = KMeans(n_clusters=10).fit_predict(rr)
kee = KMeans(n_clusters=10).fit_predict(ee)
kbb = KMeans(n_clusters=10).fit_predict(bb)

#  Calculate the purity
srr,see,sbb = [],[],[]

#  Labels for each cluster
for i in range(10):
    idx = np.where(krr==i)[0];  srr.append(ysamples[idx])
    idx = np.where(kee==i)[0];  see.append(ysamples[idx])
    idx = np.where(kbb==i)[0];  sbb.append(ysamples[idx])

#  Number of samples in each cluster belonging to the dominant class
scrr = scee = scbb = 0
for i in range(10):
    scrr += np.bincount(srr[i], minlength=10).max()
    scee += np.bincount(see[i], minlength=10).max()
    scbb += np.bincount(sbb[i], minlength=10).max() 

#  Purity is the fraction of total samples correctly assigned
#  to the dominant class
N = len(ee)
print()
print("Purity:")
print("    raw image features: %0.4f" % (scrr / N))
print("    embedded features : %0.4f" % (scee / N))
print("    layer 10 features : %0.4f" % (scbb / N))
print()

