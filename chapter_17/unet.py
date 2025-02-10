#
#  file:  unet2.py
#
#  Train a U-net to classify the M2NIST dataset
#
#  RTK, 22-Dec-2023
#  Last update:  24-Dec-2023
#
################################################################

import os
import sys
import pickle
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import concatenate, BatchNormalization
from tensorflow.keras.layers import Softmax, ReLU

def ConvBlock(inp, filters, useBN=False):
    """Add twin convolutional layers"""
    _ = Conv2D(filters, (3,3), padding='same')(inp)
    if (useBN):
        _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = Conv2D(filters, (3,3), padding='same')(_)
    if (useBN):
        _ = BatchNormalization()(_)
    return ReLU()(_)


#  Command line
if (len(sys.argv) == 1):
    print()
    print("unet <minibatch> <epochs> <useBN> blank|land|subtle <outdir>")
    print()
    print("  <minibatch>       - minibatch size (e.g. 32)")
    print("  <epochs>          - number of training epochs (e.g. 5)")
    print("  <useBN>           - 0=no, 1=yes")
    print("  blank|land|subtle - data source")
    print("  <outdir>          - output file directory (overwritten)")
    print()
    exit(0)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
useBN = True if (sys.argv[3]=="1") else False
mode = sys.argv[4].lower()
outdir = sys.argv[5]

#  Other parameters
num_classes = 11
img_rows, img_cols = 64, 84
input_shape = (img_rows, img_cols, 1)

#  Load the M2NIST dataset
if (mode == 'land'):
    x_train = np.load("../data/m2nist/xtrain_land.npy")
    x_test = np.load("../data/m2nist/xtest_land.npy")
elif (mode == 'subtle'):
    x_train = np.load("../data/m2nist/xtrain_subtle.npy")
    x_test = np.load("../data/m2nist/xtest_subtle.npy")
else:
    x_train = np.load("../data/m2nist/xtrain.npy")
    x_test = np.load("../data/m2nist/xtest.npy")

y_train = np.load("../data/m2nist/ytrain.npy")
y_test = np.load("../data/m2nist/ytest.npy")

#  Scale [0,1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#  Build the U-net model
inp = Input(input_shape)

#  Downsample
c1= ConvBlock(inp, 64, useBN=useBN)
_ = MaxPooling2D((2,2))(c1)
c2= ConvBlock(_, 128, useBN=useBN)
_ = MaxPooling2D((2,2))(c2)

#  Bottleneck
_ = ConvBlock(_, 256, useBN=useBN)

#  Upsample with skip connections
_ = UpSampling2D((2,2))(_)
_ = concatenate([c2, _], axis=3)
_ = ConvBlock(_, 128, useBN=useBN)
_ = UpSampling2D((2,2))(_)
_ = concatenate([c1,_], axis=3)
_ = ConvBlock(_, 64, useBN=useBN)

#  Softmax over each pixel with a 1x1 convolution
_ = Conv2D(num_classes, (1,1))(_)
outp = Softmax()(_)
model = Model(inputs=[inp], outputs=[outp])

#  Compile and train
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#  Results
tloss = history.history['loss']
vloss = history.history['val_loss']
terr = 1.0 - np.array(history.history['accuracy'])
verr = 1.0 - np.array(history.history['val_accuracy'])
d = [tloss,vloss,terr,verr]
os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
pickle.dump(d, open(outdir+"/results.pkl", "wb"))
prob = model.predict(x_test, verbose=0)
np.save(outdir+"/softmax.npy", prob)
model.save(outdir+"/model.keras")

