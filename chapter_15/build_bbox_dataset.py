#
#  file:  build_bbox_dataset.py
#
#  Build the 128x128 bounding box datasets
#
#  RTK, 27-Dec-2023
#  Last update:  28-Dec-2023
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
    xlo = np.random.randint(0,120)
    ylo = np.random.randint(0,120)
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
xtst = np.load("../data/mnist/mnist_test_images.npy")
ytst = np.load("../data/mnist/mnist_test_labels.npy")

#  Embed in a larger image
np.random.seed(1313)
xtrain = np.zeros((M,128,128), dtype="uint8")
ytrainb = np.zeros((M,4), dtype="float32")
ytrainc = np.zeros((M,10), dtype="uint8")

for i in range(M):
    r,c = np.random.randint(28,101,size=2)
    xtrain[i,r:(r+28),c:(c+28)] = xtrn[i]
    cl = [0]*10; cl[ytrn[i]] = 1
    ytrainc[i,:] = cl
    ytrainb[i,:] = np.array([r,c,28,28]) / 128

np.save("../data/mnist/mnist_bbox_xtrain.npy", xtrain)
np.save("../data/mnist/mnist_bbox_ytrainb.npy", ytrainb)
np.save("../data/mnist/mnist_bbox_ytrainc.npy", ytrainc)

M = 10_000
xtest = np.zeros((M,128,128), dtype="uint8")
ytestb = np.zeros((M,4), dtype="float32")
ytestc = np.zeros((M,10), dtype="uint8")

for i in range(M):
    r,c = np.random.randint(28,101,size=2)
    xtest[i,r:(r+28),c:(c+28)] = xtst[i]
    cl = [0]*10; cl[ytst[i]] = 1
    ytestc[i,:] = cl
    ytestb[i,:] = np.array([r,c,28,28]) / 128

np.save("../data/mnist/mnist_bbox_xtest.npy", xtest)
np.save("../data/mnist/mnist_bbox_ytestb.npy", ytestb)
np.save("../data/mnist/mnist_bbox_ytestc.npy", ytestc)

#  UCMerced backgrounds added
for i in range(len(xtrain)):
    img = SelectRandomPatch(fnames)
    img[np.where(xtrain[i] != 0)] = 200
    xtrain[i] = img

for i in range(len(xtest)):
    img = SelectRandomPatch(fnames)
    img[np.where(xtest[i] != 0)] = 200
    xtest[i] = img

np.save("../data/mnist/mnist_bbox_land_xtrain.npy", xtrain)
np.save("../data/mnist/mnist_bbox_land_xtest.npy", xtest)

