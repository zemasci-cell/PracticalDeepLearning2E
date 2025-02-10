#
#  file:  fine-tune_vgg16_fraction_plot.py
#
#  RTK, 03-Dec-2023
#  Last update:  10-Dec-2023
#
################################################################

import numpy as np
import matplotlib.pylab as plt

def GetResults(fraction, sgd=False):
    sgd = "_sgd" if (sgd) else ""
    f = "results/fine-tune_vgg16_fraction%s_%03d_run0/accuracy_mcc.txt" % (sgd,fraction)
    t = open(f).read()[:-1].split()
    acc0, mcc0 = float(t[3][:-1]), float(t[5])
    f = "results/fine-tune_vgg16_fraction%s_%03d_run1/accuracy_mcc.txt" % (sgd,fraction)
    t = open(f).read()[:-1].split()
    acc0 = 0.5*(acc0 + float(t[3][:-1]))
    mcc0 = 0.5*(mcc0 + float(t[5]))
    f = "results/fine-tune_vgg16_fraction%s_%03d_none_run0/accuracy_mcc.txt" % (sgd,fraction)
    t = open(f).read()[:-1].split()
    acc1, mcc1 = float(t[3][:-1]), float(t[5])
    f = "results/fine-tune_vgg16_fraction%s_%03d_none_run1/accuracy_mcc.txt" % (sgd,fraction)
    t = open(f).read()[:-1].split()
    acc1 = 0.5*(acc1 + float(t[3][:-1]))
    mcc1 = 0.5*(mcc1 + float(t[5]))
    return acc0, acc1

def GetVGG8(fraction, sgd=False):
    sgd = "_sgd" if (sgd) else ""
    f = 'results/vgg8_fraction%s_%03d_run0/accuracy_mcc.txt' % (sgd,fraction)
    t = open(f).read()[:-1].split()
    acc = float(t[3][:-1])
    f = 'results/vgg8_fraction%s_%03d_run1/accuracy_mcc.txt' % (sgd,fraction)
    t = open(f).read()[:-1].split()
    acc = 0.5*(acc + float(t[3][:-1]))
    return acc

x = [1,3,5,10]
ac0 = []
ac1 = []
for fraction in x:
    a0,a1 = GetResults(fraction)
    ac0.append(a0)
    ac1.append(a1)
ac2 = []
for fraction in x:
    ac2.append(GetVGG8(fraction))

sc0 = []
sc1 = []
for fraction in x:
    s0,s1 = GetResults(fraction, sgd=True)
    sc0.append(s0)
    sc1.append(s1)
sc2 = []
for fraction in x:
    sc2.append(GetVGG8(fraction, sgd=True))

plt.plot(x,ac0, marker='o', linewidth=0.7, fillstyle='none', color='k', label='ImageNet')
plt.plot(x,ac1, marker='s', linewidth=0.7, fillstyle='none', color='k', label='none')
plt.plot(x,ac2, marker='^', linewidth=0.7, fillstyle='none', color='k', label='VGG8')
plt.plot(x,sc0, marker='o', linewidth=0.7, color='k')
plt.plot(x,sc1, marker='s', linewidth=0.7, color='k')
plt.plot(x,sc2, marker='^', linewidth=0.7, color='k')
plt.xlabel("Fraction of full CIFAR-10 training set")
plt.ylabel("Accuracy (test set, $n=2$)")
plt.legend(loc='best')
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("fine-tune_vgg16_fraction_plot.eps", dpi=300)
plt.savefig("fine-tune_vgg16_fraction_plot.png", dpi=300)
plt.show()

