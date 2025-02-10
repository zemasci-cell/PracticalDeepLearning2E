#
#  file:  resnet18.py
#
#  ResNet-18 applied to CIFAR-10
#
#  RTK, 17-Aug-2023
#  Last update:  07-Sep-2023
#
################################################################

import os
import sys
import pickle
from sklearn.metrics import matthews_corrcoef
import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.layers import Softmax, Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np

def ConfusionMatrix(pred, y):
    """Return a confusion matrix"""
    cm = np.zeros((10,10), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc


def ResidualBlock(x, filters, downsample=False, useBN=False):
    """Build a residual block"""
    if (downsample):
        strides = (2,2)
        inp = Conv2D(filters, (1,1), strides=strides, padding='same')(x)
    else:
        strides = (1,1)
        inp = x

    _ = Conv2D(filters, (3,3), strides=strides, padding='same')(x)
    if (useBN):
        _ = BatchNormalization()(_)
    _ = ReLU()(_)

    _ = Conv2D(filters, (3,3), strides=(1,1), padding='same')(_)
    if (useBN):
        _ = BatchNormalization()(_)

    _ = Add()([_, inp])
    return ReLU()(_)


#  Command line
if (len(sys.argv) == 1):
    print()
    print("resnet18 <minibatch> <epochs> <useBN> <outdir>")
    print()
    print("  <minibatch>  -  minibatch size (e.g. 128)")
    print("  <epochs>     -  number of training epochs (e.g. 16)")
    print("  <useBN>      -  0=no, 1=yes")
    print("  <outdir>     -  output file directory (overwritten)")
    print()
    exit(0)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
useBN = True if (int(sys.argv[3])) else False
outdir = sys.argv[4]

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

#  Convert labels to one-hot vectors
y_train = keras.utils.to_categorical(ytrain, num_classes)
y_test = keras.utils.to_categorical(ytest, num_classes)

#  ResNet-18 model
inp = Input(input_shape)
_ = Conv2D(64, (3,3), strides=(1,1), padding='same')(inp)
_ = ResidualBlock(_,  64, downsample=False, useBN=useBN)
_ = ResidualBlock(_,  64, downsample=False, useBN=useBN)
_ = ResidualBlock(_, 128, downsample=True,  useBN=useBN)
_ = ResidualBlock(_, 128, downsample=False, useBN=useBN)
_ = ResidualBlock(_, 256, downsample=True,  useBN=useBN)
_ = ResidualBlock(_, 256, downsample=False, useBN=useBN)
_ = ResidualBlock(_, 512, downsample=True,  useBN=useBN)
_ = ResidualBlock(_, 512, downsample=False, useBN=useBN)
_ = GlobalAveragePooling2D()(_)
_ = Dense(num_classes)(_)
outp = Softmax()(_)

model = Model(inputs=inp, outputs=outp)
model.summary()

#  Compile and train
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
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


