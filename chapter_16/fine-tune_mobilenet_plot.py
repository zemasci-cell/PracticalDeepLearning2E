#
#  file: fine-tune_mobilenet_plot.py
#
#  Plot the fine-tune MobileNet results
#
#  RTK, 01-Dec-2023
#  Last update:  03-Dec-2023
#
################################################################

import matplotlib.pylab as plt

def GetResults(level):
    if (level == 13):
        level = 'all'
    f = "results/fine-tune_mobilenet_128_12_%s_run0/accuracy_mcc.txt" % str(level)
    t = open(f).read()[:-1].split()
    acc, mcc = float(t[3][:-1]), float(t[5])
    f = "results/fine-tune_mobilenet_128_12_%s_run1/accuracy_mcc.txt" % str(level)
    t = open(f).read()[:-1].split()
    acc = 0.5*(acc + float(t[3][:-1]))
    mcc = 0.5*(mcc + float(t[5]))
    return acc,mcc

#  by number of unfrozen blocks
acc,mcc = [],[]
x = [0,2,4,6,8,10,13]
for i in x:
    a,m = GetResults(i)
    acc.append(a)
    mcc.append(m)

#  from scratch - copied by hand
scratch = 0.5*(0.6156+0.6137)

plt.plot(x,acc, marker='o', fillstyle='none', color='k', linewidth=0.7, label='pretrained')
plt.plot(13,scratch, marker='*', color='k', linestyle='none', label='from scratch')
plt.xlabel("Number of Unfrozen Blocks")
plt.ylabel("Mean Accuracy ($n=2$)")
plt.legend(loc='best')
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("fine-tune_mobilenet_plot.eps")
plt.show()

