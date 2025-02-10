#
#  file:  build_multilabel_dataset.py
#
#  Build a 1-3 MNIST digit per image dataset
#
#  RTK, 03-Jan-2024
#  Last update:  03-Jan-2024
#
################################################################

import os
import numpy as np
from PIL import Image

def SelectRandomPatch(fnames):
    """Return a randomly selected 128x128 UCMerced patch"""
    f = fnames[np.random.randint(0,len(fnames))]
    im = np.array(Image.open(f).convert("L"))
    rows,cols = im.shape
    xlo = np.random.randint(0,110)
    ylo = np.random.randint(0,110)
    return im[xlo:(xlo+128),ylo:(ylo+128)]//3

#  List of UCMerced images
fnames = []
for d in os.listdir("../data/UCMerced_LandUse/Images"):
    fbase = "../data/UCMerced_LandUse/Images/" + d + "/"
    for k in os.listdir(fbase):
        fnames.append(fbase + k)

#  Existing MNIST
M = 6000
xtrn = np.load("../data/mnist/mnist_train_images.npy")[:M]
ytrn = np.load("../data/mnist/mnist_train_labels.npy")[:M]
xtst = np.load("../data/mnist/mnist_test_images.npy")[:M]
ytst = np.load("../data/mnist/mnist_test_labels.npy")[:M]

#  Embed in a larger image
np.random.seed(1313)
xtrain = np.zeros((M,128,128), dtype="uint8")
ytrain = np.zeros((M,10), dtype="uint8")

for i in range(M):
    n = np.random.randint(1,4)
    y = []
    for j in range(np.random.randint(1,4)):
        r,c = np.random.randint(1,101,size=2)
        while (xtrain[i,r:(r+28),c:(c+28)].sum() > 0):
            r,c = np.random.randint(1,101,size=2)
        idx = np.random.randint(0,M)
        xtrain[i,r:(r+28),c:(c+28)] = xtrn[idx]
        y.append(ytrn[idx])
    cl = [0]*10
    for k in y:
        cl[k] = 1
    ytrain[i,:] = cl

np.save("../data/mnist/mnist_multilabel_xtrain.npy", xtrain)
np.save("../data/mnist/mnist_multilabel_ytrain.npy", ytrain)

xtest = np.zeros((M,128,128), dtype="uint8")
ytest = np.zeros((M,10), dtype="uint8")

for i in range(M):
    n = np.random.randint(1,4)
    y = []
    for j in range(np.random.randint(1,4)):
        r,c = np.random.randint(1,101,size=2)
        while (xtest[i,r:(r+28),c:(c+28)].sum() > 0):
            r,c = np.random.randint(1,101,size=2)
        idx = np.random.randint(0,M)
        xtest[i,r:(r+28),c:(c+28)] = xtst[idx]
        y.append(ytst[idx])
    cl = [0]*10
    for k in y:
        cl[k] = 1
    ytest[i,:] = cl

np.save("../data/mnist/mnist_multilabel_xtest.npy", xtest)
np.save("../data/mnist/mnist_multilabel_ytest.npy", ytest)

#  UCMerced backgrounds added
for i in range(len(xtrain)):
    img = SelectRandomPatch(fnames)
    img[np.where(xtrain[i] != 0)] = 200
    xtrain[i] = img

for i in range(len(xtest)):
    img = SelectRandomPatch(fnames)
    img[np.where(xtest[i] != 0)] = 200
    xtest[i] = img

np.save("../data/mnist/mnist_multilabel_land_xtrain.npy", xtrain)
np.save("../data/mnist/mnist_multilabel_land_xtest.npy", xtest)

