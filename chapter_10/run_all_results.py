import numpy as np

lines = [i[:-1] for i in open("run_all_results.txt") if (i != "")]
d = 100.0*np.array([float(i.split()[-1]) for i in lines]).reshape((11,10))

for i in range(11):
    print("%2d: %0.5f +/- %0.5f" % (i, d[i].mean(), d[i].std(ddof=1) / np.sqrt(d.shape[1])))

