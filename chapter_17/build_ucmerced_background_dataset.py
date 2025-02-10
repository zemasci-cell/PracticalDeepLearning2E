#
#  file:  build_ucmerced_background_dataset.py
#
#  Add random UCMerced backgrounds to M2NIST
#
#  RTK, 22-Dec-2023
#  Last update:  22-Dec-2023
#
################################################################

import os
import numpy as np
from PIL import Image

def SelectRandomPatch(fnames):
    """Return a randomly selected 64x84 UCMerced patch"""
    f = fnames[np.random.randint(0,len(fnames))]
    im = np.array(Image.open(f).convert("L"))
    rows,cols = im.shape
    xlo = np.random.randint(0,128)
    ylo = np.random.randint(0,128)
    return im[xlo:(xlo+64),ylo:(ylo+84)]

#  List of UCMerced images
fnames = []
for d in os.listdir("../data/UCMerced_LandUse/Images"):
    fbase = "../data/UCMerced_LandUse/Images/" + d + "/"
    for k in os.listdir(fbase):
        fnames.append(fbase + k)

#  Existing images
xtrn = np.load("../data/m2nist/xtrain.npy")
xtst = np.load("../data/m2nist/xtest.npy")

#  Replace zero pixels with randomly selected background
#  patch, 64x84 pixels
np.random.seed(68040)
xt = np.zeros((len(xtrn),64,84), dtype="uint8")
for i in range(len(xtrn)):
    xt[i,:,:] = SelectRandomPatch(fnames)
    idx = np.where(xtrn[i]!=0)
    xt[i,idx[0],idx[1]] = (xtrn[i,idx[0],idx[1]]/2.0).astype("uint8")
np.save("../data/m2nist/xtrain_land.npy", xt)

xt = np.zeros((len(xtst),64,84), dtype="uint8")
for i in range(len(xtst)):
    xt[i,:,:] = SelectRandomPatch(fnames)
    idx = np.where(xtst[i]!=0)
    xt[i,idx[0],idx[1]] = (xtst[i,idx[0],idx[1]]/2.0).astype("uint8")
np.save("../data/m2nist/xtest_land.npy", xt)

#  Subtle -- add to the existing pixel value
np.random.seed(68040)  # reset the seed to use the same backgrounds
xt = np.zeros((len(xtrn),64,84), dtype="uint8")
for i in range(len(xtrn)):
    xt[i,:,:] = SelectRandomPatch(fnames)
    idx = np.where(xtrn[i]!=0)
    xt[i,idx[0],idx[1]] = np.minimum(xt[i,idx[0],idx[1]]+20.0,255).astype("uint8")
np.save("../data/m2nist/xtrain_subtle.npy", xt)

xt = np.zeros((len(xtst),64,84), dtype="uint8")
for i in range(len(xtst)):
    xt[i,:,:] = SelectRandomPatch(fnames)
    idx = np.where(xtst[i]!=0)
    xt[i,idx[0],idx[1]] = np.minimum(xt[i,idx[0],idx[1]]+20.0,255).astype("uint8")
np.save("../data/m2nist/xtest_subtle.npy", xt)

