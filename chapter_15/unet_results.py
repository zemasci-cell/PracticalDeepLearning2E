#
#  file:  unet_results.py
#
#  Interpret the output of unet.py
#
#  RTK, 23-Dec-2023
#  Last update:  25-Dec-2023
#
################################################################

import os
import sys
import numpy as np
from PIL import Image

def IoU(x,y):
    """Calculate IoU scores"""

    def CalcIoU(x,y,k):
        """IoU for an image"""
        xk = np.where(x.ravel()==(k+1))[0]
        yk = np.where(y.ravel())[0]
        i = set(xk) & set(yk)
        u = set(xk) | set(yk)
        return len(i) / len(u)

    iou = np.zeros(10)  # IoU by class
    n = np.zeros(10)    # number of instances of that class, including FP
    for k in range(10):
        for i in range(len(x)):
            id0 = np.where(x[i]==(k+1))[0]
            id1 = np.where(y[i,:,:,k]==1)[0]
            if (id0.size > 0) or (id1.size > 0):
                iou[k] += CalcIoU(x[i], y[i,:,:,k], k)
                n[k] += 1
    return iou / n


if (len(sys.argv) == 1):
    print()
    print("unet_results <dir> blank|land|subtle [images]")
    print()
    print("  <dir>             - source directory (output of unet.py)")
    print("  blank|land|subtle - test image source")
    print("  images            - if present, create .png image pairs")
    print()
    exit(0)

sdir = sys.argv[1]
mode = sys.argv[2].lower()
mdir = "images" if (len(sys.argv) > 3) else ""

#  Load the test images
if (mode == "land"):
    xtst = np.load("../data/m2nist/xtest_land.npy")
elif (mode == "subtle"):
    xtst = np.load("../data/m2nist/xtest_subtle.npy")
else:
    xtst = np.load("../data/m2nist/xtest.npy")

#  Load the labels
ytst = np.load("../data/m2nist/ytest.npy")

#  Load the softmax predictions
soft = np.load(sdir + "/softmax.npy")

#  Turn the predictions into grayscale images
pred = np.zeros(soft.shape[:3], dtype="uint8")
for i in range(len(soft)):
    im = (1+np.argmax(soft[i], axis=2)).astype("uint8")
    im[np.where(im==11)] = 0
    pred[i,...] = im

#  Calculate the IoU
iou = IoU(pred, ytst)
np.save(sdir+"/iou.npy", iou)
print("Mean per class IoU:")
for k in range(10):
    print("    %d: %0.6f" % (k,iou[k]))
print()

#  Create the output directory and images
np.save(sdir+"/predictions.npy", pred)

if (mdir != ""):
    c = [[0,0,0],         # black
         [255, 0, 0],     # red
         [0, 255, 0],     # green
         [0, 0, 255],     # blue
         [255, 255, 0],   # yellow
         [0, 255, 255],   # cyan
         [255, 0, 255],   # magenta
         [255, 165, 0],   # orange
         [128, 0, 128],   # purple
         [50, 205, 50],   # lime green
         [255, 192, 203]] # pink
    
    os.system("rm -rf %s/images; mkdir %s/images" % (sdir,sdir))

    for i in range(len(pred)):
        im = np.zeros((64,2*84,3), dtype="uint8")
        im[:,:84,0] = xtst[i]
        im[:,:84,1] = xtst[i]
        im[:,:84,2] = xtst[i]
        for x in range(64):
            for y in range(84):
                t = pred[i,x,y]
                im[x,y+84,0] = c[t][0]
                im[x,y+84,1] = c[t][1]
                im[x,y+84,2] = c[t][2]
        Image.fromarray(im).save("%s/images/image_%04d.png" % (sdir,i))

