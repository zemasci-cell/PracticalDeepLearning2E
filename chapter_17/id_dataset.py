#  Estimate the intrinsic dimensionality of user-supplied datasets
import sys
import numpy as np
from skdim.id import MLE

if (len(sys.argv) == 1):
    print()
    print("id_dataset <dataset>")
    print()
    print("  <dataset> - dataset (.npy)")
    print()
    exit(0)

x = np.load(sys.argv[1])
nsamp, rest = x.shape[0], x.shape[1:]
x = x.reshape((nsamp, np.prod(rest)))

ID = int(MLE().fit_transform(x) + 0.5)

print("Intrinsic dimensionality (MLE): %d" % ID)

