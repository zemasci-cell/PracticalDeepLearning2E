#
#  file: fine-tune_vgg16_plot.py
#
#  Plot the fine-tune VGG16 results
#
#  RTK, 01-Dec-2023
#  Last update:  01-Dec-2023
#
################################################################

import matplotlib.pylab as plt

def GetResults(level):
    f = "results/fine-tune_vgg16_128_12_%d_run0/accuracy_mcc.txt" % level
    t = open(f).read()[:-1].split()
    acc, mcc = float(t[3][:-1]), float(t[5])
    f = "results/fine-tune_vgg16_128_12_%d_run1/accuracy_mcc.txt" % level
    t = open(f).read()[:-1].split()
    acc = 0.5*(acc + float(t[3][:-1]))
    mcc = 0.5*(mcc + float(t[5]))
    return acc,mcc

#  by number of frozen blocks
acc,mcc = [],[]
for i in range(6):
    a,m = GetResults(i)
    acc.append(a)
    mcc.append(m)

#  from scratch - copied by hand
scratch = 0.5*(0.7763+0.7699)

plt.plot(range(6),acc, marker='o', fillstyle='none', color='k', linewidth=0.7, label='pretrained')
plt.plot(0,scratch, marker='*', color='k', linestyle='none', label='from scratch')
plt.xlabel("Number of Frozen Blocks")
plt.ylabel("Mean Accuracy ($n=2$)")
plt.ylim((0.69,0.84))
plt.legend(loc='best')
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("fine-tune_vgg16_plot.eps")
plt.show()

