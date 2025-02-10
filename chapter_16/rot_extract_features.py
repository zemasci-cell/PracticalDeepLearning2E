#
#  file:  rot_extract_features.py
#
#  Generate feature vectors from the pretrained rot models
#
#  RTK, 07-Jan-2024
#  Last update: 08-Jan-2024
#
################################################################

import sys
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input

if (len(sys.argv) == 1):
    print()
    print("rot_extract_features <model> <block> <outbase>")
    print()
    print("  <model>   - pretrained base model")
    print("  <block>   - output block, 1-3")
    print("  <outbase> - base output name")
    print()
    exit(0)

mname = sys.argv[1]
blk = int(sys.argv[2])
outdir = sys.argv[3]

#  Translate block to model layer
idx = [4,8,12][blk-1]

#  Load CIFAR-10 - remember to use only the first M train
#                  and test labels as well
M = 10000
xtrain = np.load("../data/cifar10/cifar10_train_images.npy")[:M] / 255.0
xtest  = np.load("../data/cifar10/cifar10_test_images.npy")[:M] / 255.0

#  Load the base model
model = load_model(mname)

#  Set up to extract the desired layer
if (idx == -1):
    #  entire base model output
    xtrn = model.predict(xtrain, verbose=0)
    xtst = model.predict(xtest, verbose=0)
else:
    #  Build an ancillary model to get the output
    #  of the desired convolutional block
    inp = Input(shape=(32,32,3))
    layer = model.layers[idx]
    cmodel = Model(inputs=model.input, outputs=layer.output)
    xtrn = cmodel.predict(xtrain, verbose=0)
    xtst = cmodel.predict(xtest, verbose=0)
    xtrn = xtrn.reshape((xtrn.shape[0], np.product(xtrn.shape[1:])))
    xtst = xtst.reshape((xtst.shape[0], np.product(xtst.shape[1:])))

#  And write them to disk
np.save(outdir+"_cifar10_features_train.npy", xtrn)
np.save(outdir+"_cifar10_features_test.npy", xtst)

