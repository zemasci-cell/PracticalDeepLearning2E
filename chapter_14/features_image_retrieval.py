#
#  file:  features_image_retrieval.py
#
#  Use the VGG16 embeddings to find training set images
#  similar to selected test set images.
#
#  Requires:
#   features_cifar10_train.npy  features_cifar10_block_train.npy
#   features_cifar10_test.npy   features_cifar10_block_test.npy
#
#  Run 'features_cifar10_vgg16.py' and
#      'features_cifar10_vgg16_block.py' first
#
#  RTK, 05-Dec-2023
#  Last update:  07-Dec-2023
#
################################################################

import os
import sys
import numpy as np
from PIL import Image

def cosine(a,b):
    """Calculate the cosine distance between a and b"""
    num = np.dot(a,b)
    den = np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b))
    return 1.0 - num/den

def euclidean(a,b):
    """Calculate the Euclidean distance between a and b"""
    return np.sqrt(((a-b)**2).sum())


#  Command line
if (len(sys.argv) == 1):
    print()
    print("features_image_retrieval vgg16|layer10 <classname> <ntest> <ntrain> <metric> <outdir>")
    print()
    print("  vgg16|layer10 - feature source")
    print("  <classname> - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
    print("  <ntest>     - number of randomly selected test images to use")
    print("  <ntrain>    - number of training set similar images per test image")
    print("  <metric>    - euclidean|cosine")
    print("  <outdir>    - output directory (overwritten)")
    print()
    exit(0)

mode = 'layer10' if (sys.argv[1].lower() == 'layer10') else 'vgg16'

classes = [ 
'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'
]
try:
    target = classes.index(sys.argv[2].lower())
except:
    print("%s is not a valid class name" % sys.argv[2].lower())
    exit(1)

ntest = int(sys.argv[3])
ntrain = int(sys.argv[4])
metric = cosine if (sys.argv[5].lower()=='cosine') else euclidean
outdir = sys.argv[6]

#  Source data (images, labels and embeddings)
xtrn_img = np.load("../data/cifar10/cifar10_train_images.npy")
xtst_img = np.load("../data/cifar10/cifar10_test_images.npy")
ytrain = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()
ytest = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

if (mode == 'layer10'):
    xtrn_emb = np.load("features_cifar10_block_train.npy")
    xtst_emb = np.load("features_cifar10_block_test.npy")
else:
    xtrn_emb = np.load("features_cifar10_train.npy")
    xtst_emb = np.load("features_cifar10_test.npy")

#  Pick ntest instances of target class from the test data
idx = np.where(ytest == target)[0]
np.random.seed(359)  # remove to select new images
k = np.argsort(np.random.random(len(idx)))
np.random.seed()
idx = idx[k][:ntest]
tst_img = xtst_img[idx]  # selected images of target class
tst_emb = xtst_emb[idx]  # corresponding embeddings

#  For each target sample, find the ntrain closest embeddings using the
#  selected metric
results = [None]*ntest
classes = [None]*ntest

for i in range(ntest):
    dist = []
    for j in range(len(ytrain)):
        dist.append(metric(tst_emb[i], xtrn_emb[j]))
    idx = np.argsort(dist)[:ntrain]
    results[i] = xtrn_img[idx]
    classes[i] = ytrain[idx]

#  Create the output directory and dump the respective images
os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
for i in range(ntest):
    im = Image.fromarray(tst_img[i])
    im.save(outdir+"/image_%d.png" % i)
    for j in range(ntrain):
        im = Image.fromarray(results[i][j])
        im.save(outdir+"/image_%d_%d.png" % (i,j))
    np.save(outdir+"/classes.npy", np.array(classes))

