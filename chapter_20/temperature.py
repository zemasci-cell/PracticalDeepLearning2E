#
#  file: temperature.py
#
#  Show the effect of temperature on a probability
#  distribution
#
#  RTK, 13-Mar-2024
#  Last update:  13-Mar-2024
#
################################################################

import sys
import numpy as np
import matplotlib.pylab as plt

def Adjusted(p, T):
    """Adjust by the given temperature"""
    s = np.exp(np.log(p+1e-8)/T)
    return s / s.sum()


if (len(sys.argv) == 1):
    print()
    print("temperature <left> <middle> <right>")
    print()
    print("  <left>, <middle>, <right> -- temperatures [0,1]")
    print()
    exit(0)

t0 = float(sys.argv[1]) + 1e-4
t1 = float(sys.argv[2]) + 1e-4
t2 = float(sys.argv[3]) + 1e-4

#  A hypothetical distribution (grayscale image histogram)
orig = np.load("probability_distribution.npy")

#  Temperature adjusted versions
a0 = Adjusted(orig, t0)
a1 = Adjusted(orig, t1)
a2 = Adjusted(orig, t2)

#  Side by side plots
x = np.linspace(0,1,len(orig))
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9,5))
ax0.plot(x, a0, color='k')
ax1.plot(x, a1, color='k')
ax2.plot(x, a2, color='k')
ax0.set_title("$T=%0.1f$" % t0)
ax1.set_title("$T=%0.1f$" % t1)
ax2.set_title("$T=%0.1f$" % t2)
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("temperature_plot.png", dpi=300)
plt.savefig("temperature_plot.eps", dpi=300)
plt.show()

