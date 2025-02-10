import numpy as np
import matplotlib.pylab as plt
x = np.load("../data/iris/iris2_train.npy")
y = np.load("../data/iris/iris2_train_labels.npy")
i = np.argsort(np.random.random(len(y)))[:400]
x = x[i]
y = y[i]
i0 = np.where(y==0)[0]
i1 = np.where(y==1)[0]
x0 = x[i0]
x1 = x[i1]
plt.plot(x0[:,0],x0[:,1],marker='o',color='k',fillstyle='none',linestyle='none')
plt.plot(x1[:,0],x1[:,1],marker='^',color='k',fillstyle='none',linestyle='none')
plt.xlabel("$x_0$")
plt.xlabel("$x_1$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("nn_iris_dataset_plot.eps", dpi=300)
plt.savefig("nn_iris_dataset_plot.png", dpi=300)

