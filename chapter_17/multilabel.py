#
#  file:  multilabel.py
#
#  Multilabel model for MNIST
#
#  RTK, 03-Jan-2024
#  Last update:  03-Jan-2024
#
################################################################

import os
import sys
import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np


def ConvBlock(_, filters):
    """Build convolution-relu-maxpooling block"""

    _ = Conv2D(filters, (3,3), padding='same')(_)
    _ = ReLU()(_)
    _ = Conv2D(filters, (3,3), padding='same')(_)
    _ = ReLU()(_)
    return MaxPooling2D((2,2))(_)


def DenseBlock(_, nodes):
    """Build a Dense-ReLU-Dropout block"""
    
    _ = Dense(nodes)(_)
    _ = ReLU()(_)
    _ = Dropout(0.5)(_)
    return _


#  Command line
if (len(sys.argv) == 1):
    print()
    print("multilabel <minibatch> <epochs> blank|land <outdir>")
    print()
    print("  <minibatch>    -  minibatch size (e.g. 32)")
    print("  <epochs>       -  number of training epochs (e.g. 10)")
    print("  blank | land   -  dataset type")
    print("  <outdir>       -  output file directory (overwritten)")
    print()
    exit(0)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
mode = sys.argv[3].lower()
outdir = sys.argv[4]

#  Other parameters
num_classes = 10
img_rows, img_cols = 128,128
input_shape = (img_rows, img_cols, 1)

#  Load the multilabel MNIST dataset
if (mode == 'blank'):
    xtrain = np.load("../data/mnist/mnist_multilabel_xtrain.npy")[:,:,:,np.newaxis]
else:
    xtrain = np.load("../data/mnist/mnist_multilabel_land_xtrain.npy")[:,:,:,np.newaxis]
if (mode == 'blank'):
    xtest = np.load("../data/mnist/mnist_multilabel_xtest.npy")[:,:,:,np.newaxis]
else:
    xtest = np.load("../data/mnist/mnist_multilabel_land_xtest.npy")[:,:,:,np.newaxis]
ytrain = np.load("../data/mnist/mnist_multilabel_ytrain.npy")
ytest = np.load("../data/mnist/mnist_multilabel_ytest.npy")

#  Scale [0,1]
xtrain = xtrain.astype('float32') / 255
xtest = xtest.astype('float32') / 255

#  Model
inp = Input(input_shape)
_ = ConvBlock(inp, 32)
_ = ConvBlock(_,   64)
_ = ConvBlock(_,  128)
_ = Flatten()(_)
_ = DenseBlock(_, 128)
outp = Dense(num_classes, activation='sigmoid')(_)

model = Model(inputs=inp, outputs=outp)
model.summary()

#  Compile and train
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam())

model.fit(xtrain, ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

#  Results
os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
pred = model.predict(xtest, verbose=0)
np.save(outdir+"/predictions.npy", pred)
model.save(outdir+"/model.keras")

