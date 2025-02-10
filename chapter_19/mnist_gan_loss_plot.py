#  Plot the G and D loss by epoch
import numpy as np
import matplotlib.pylab as plt

lines = [i[:-1].split() for i in open('mnist_gan_run.txt')]
G = np.array([float(line[2].split("=")[1][:-1]) for line in lines])
D = np.array([float(line[3].split("=")[1]) for line in lines])
X = range(1,len(G)+1)

N = 200
plt.plot(X[::N], G[::N], linestyle='solid', color='k', label='$G$')
plt.plot(X[::N], D[::N], linestyle='dashed', color='k', label='$D$')
plt.plot(X[::N], [0.69315]*len(G[::N]), linestyle='dotted', color='k')

plt.xlabel("Minibatch")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("mnist_gan_loss_plot.eps", dpi=300)
plt.savefig("mnist_gan_loss_plot.png", dpi=300)
plt.show()

