#
#  file:  vgg8_sgd.py
#
#  VGG8 for CIFAR-10
#
#  RTK, 14-Aug-2023
#  Last update:  16-Aug-2023
#
################################################################

import os
import sys
import pickle
from sklearn.metrics import matthews_corrcoef
import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.layers import Softmax, SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np

def ConfusionMatrix(pred, y):
    """Return a confusion matrix"""
    cm = np.zeros((10,10), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc


def ConvBlock(_, filters, dropout=0, pct=0.25, useBN=False):
    """Build a VGG convolution-relu-maxpooling block w/optional dropout"""

    if (useBN):
        _ = Conv2D(filters, (3,3), padding='same')(_)
        _ = BatchNormalization()(_)  # BN before activation
        _ = ReLU()(_)
        _ = Conv2D(filters, (3,3), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = ReLU()(_)
    else:
        _ = Conv2D(filters, (3,3), padding='same')(_)
        _ = ReLU()(_)
        if (dropout==1):             # Dropout after activation
            _ = Dropout(pct)(_)
        elif (dropout==2):
            _ = SpatialDropout2D(pct)(_)
        _ = Conv2D(filters, (3,3), padding='same')(_)
        _ = ReLU()(_)
        if (dropout==1):
            _ = Dropout(pct)(_)
        elif (dropout==2):
            _ = SpatialDropout2D(pct)(_)

    return MaxPooling2D((2,2))(_)


def DenseBlock(_, nodes, useBN=False):
    """Build a Dense-ReLU-Dropout block"""
    
    _ = Dense(nodes)(_)
    if (useBN):
        _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = Dropout(0.5)(_)
    return _


#  Command line
if (len(sys.argv) == 1):
    print()
    print("vgg8 <minibatch> <epochs> <dropout_type> <dropout_pct> <useBN> <outdir> [<fraction>]")
    print()
    print("  <minibatch>    -  minibatch size (e.g. 128)")
    print("  <epochs>       -  number of training epochs (e.g. 16)")
    print("  <dropout_type> -  0=none, 1=dropout/dropout, 2=spatial/dropout")
    print("  <dropout_pct>  -  conv layer dropout fraction (e.g. 0.5)")
    print("  <useBN>        -  0=no, 1=yes (ignores dropout settings yes)")
    print("  <outdir>       -  output file directory (overwritten)")
    print("  <fraction>  -  fraction of full training set to use (def=1.0)")
    print()
    exit(0)

batch_size = int(sys.argv[1])  # VGG paper uses 256
epochs = int(sys.argv[2])
dropout = int(sys.argv[3])
pct = float(sys.argv[4])
useBN = True if int(sys.argv[5]) else False
outdir = sys.argv[6]
fraction = float(sys.argv[7]) if (len(sys.argv)==8) else 1.0

#  Other parameters
num_classes = 10
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

#  Load the full RGB CIFAR-10 dataset (unaugmented)
x_train = np.load("../data/cifar10/cifar10_train_images.npy")
ytrain  = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()
x_test  = np.load("../data/cifar10/cifar10_test_images.npy")
ytest   = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

#  Scale [0,1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#  Select desired fraction
if (fraction != 1.0):
    np.random.seed(73939133)
    n = int(len(x_train)*fraction)
    idx = np.argsort(np.random.random(len(x_train)))[:n]
    x_train = x_train[idx]
    ytrain = ytrain[idx]
    np.random.seed()

#  Convert labels to one-hot vectors
y_train = keras.utils.to_categorical(ytrain, num_classes)
y_test = keras.utils.to_categorical(ytest, num_classes)

#  Model suitable for 32x32 CIFAR-10 (VGG8)
inp = Input(input_shape)
_ = ConvBlock(inp, 64, dropout=dropout, pct=pct, useBN=useBN)
_ = ConvBlock(_,  128, dropout=dropout, pct=pct, useBN=useBN)
_ = ConvBlock(_,  256, dropout=dropout, pct=pct, useBN=useBN)
_ = Flatten()(_)
_ = DenseBlock(_, 2048, useBN=useBN)
_ = DenseBlock(_, 2048, useBN=useBN)
_ = Dense(num_classes)(_)
outp = Softmax()(_)

model = Model(inputs=inp, outputs=outp)
model.summary()

#  Compile and train
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.005),
              metrics=['accuracy'])

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
pred = model.predict(x_test, verbose=0)
plabel = np.argmax(pred, axis=1)
cm, acc = ConfusionMatrix(plabel, ytest)
mcc = matthews_corrcoef(ytest, plabel)
s = 'Test set accuracy: %0.4f, MCC: %0.4f' % (acc,mcc)
with open(outdir+"/accuracy_mcc.txt", "w") as f:
    f.write(s+"\n")
np.save(outdir+"/confusion_matrix.npy", cm)
np.save(outdir+"/predictions.npy", pred)
model.save(outdir+"/model.keras")
print(s)
print(cm)

