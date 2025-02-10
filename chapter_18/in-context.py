#
#  file:  in-context.py
#
#  Generate a dataset for in-context learning example
#
#  RTK, 29-Feb-2024
#  Last update:  29-Feb-2024
#
################################################################

import numpy as np
from sklearn.datasets import make_classification

x,y = make_classification(n_samples=60, n_features=4, n_classes=3, 
        n_clusters_per_class=1, random_state=6502)

#  scale [-1,+1] and round to two decimals
x = x / np.abs(x).max()
x = (x*100).astype("int32") / 100.0

i = np.where(y==0)[0];  x0 = x[i]
i = np.where(y==1)[0];  x1 = x[i]
i = np.where(y==2)[0];  x2 = x[i]
np.random.seed(6809)

xtrn = np.vstack((x0[:12],x1[:12],x2[:12]))
ytrn = np.array([0]*12 + [1]*12 + [2]*12)
i = np.argsort(np.random.random(len(ytrn)))
xtrn, ytrn = xtrn[i], ytrn[i]

xtst = np.vstack((x0[12:],x1[12:],x2[12:]))
ytst = np.array([0]*8 + [1]*8 + [2]*8)
i = np.argsort(np.random.random(len(ytst)))
xtst, ytst = xtst[i], ytst[i]

print(xtrn)
print(ytrn)
print()
print(xtst)
print(ytst)

np.save("in-context_xtrain.npy", xtrn)
np.save("in-context_ytrain.npy", ytrn)
np.save("in-context_xtest.npy", xtst)
np.save("in-context_ytest.npy", ytst)

