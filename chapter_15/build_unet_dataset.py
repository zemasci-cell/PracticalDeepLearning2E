#
#  file:  build_unet_dataset.py
#
#  RTK, 21-Dec-2023
#  Last update:  21-Dec-2023
#
###########################################################

import numpy as np
x = np.load("../data/m2nist/images.npy")
y = np.load("../data/m2nist/labels.npy")
i = np.argsort(np.random.random(len(y)))
x,y = x[i], y[i]
n = int(0.8*len(y))
xtrn,ytrn = x[:n], y[:n]
xtst,ytst = x[n:], y[n:]
np.save("../data/m2nist/xtrain.npy", xtrn)
np.save("../data/m2nist/ytrain.npy", ytrn)
np.save("../data/m2nist/xtest.npy", xtst)
np.save("../data/m2nist/ytest.npy", ytst)

