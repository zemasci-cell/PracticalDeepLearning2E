#
#  file:  vgg13.py
#
#  VGG13 with CIFAR-10
#
#  RTK, 14-Aug-2023
#  Last update:  17-Aug-2023
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
from PIL import Image

def ConfusionMatrix(pred, y):
    """Return a confusion matrix"""
    cm = np.zeros((10,10), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc


def ConvBlock(_, filters):
    """Build a VGG convolution-relu-maxpooling block w/optional dropout"""

    _ = Conv2D(filters, (3,3), padding='same')(_)
    _ = ReLU()(_)
    _ = Conv2D(filters, (3,3), padding='same')(_)
    _ = ReLU()(_)
    return MaxPooling2D((2,2))(_)


def DenseBlock(_, nodes, useBN=False):
    """Build a Dense-ReLU-Dropout block"""
    
    _ = Dense(nodes)(_)
    _ = ReLU()(_)
    _ = Dropout(0.5)(_)
    return _


#  Settings
batch_size = 128
epochs = 10
outdir = "vgg13"
num_classes = 10
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

#  Load the full RGB CIFAR-10 dataset (unaugmented)
if (not os.path.exists("cifar10_scaled_xtrain.npy")):
    idx = np.argsort(np.random.random(6000))
    x_train = np.load("../data/cifar10/cifar10_train_images.npy")[idx]
    ytrain  = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()[idx]
    np.save("cifar10_scaled_ytrain.npy", ytrain)

    idx = np.argsort(np.random.random(600))
    x_test  = np.load("../data/cifar10/cifar10_test_images.npy")[idx]
    ytest   = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()[idx]
    np.save("cifar10_scaled_ytest.npy", ytest)

    x = np.zeros((x_train.shape[0],224,224,3), dtype="float16")
    for i in range(x_train.shape[0]):
        im = Image.fromarray(x_train[i]).resize((224,224))
        x[i,:,:,:] = (np.array(im)/255.0).astype("float16")
    np.save("cifar10_scaled_xtrain.npy",x)

    x = np.zeros((x_test.shape[0],224,224,3), dtype="float16")
    for i in range(x_test.shape[0]):
        im = Image.fromarray(x_test[i]).resize((224,224))
        x[i,:,:,:] = (np.array(im)/255.0).astype("float16")
    np.save("cifar10_scaled_xtest.npy",x)
    exit(0)
else:
    x_train = np.load("cifar10_scaled_xtrain.npy")
    x_test = np.load("cifar10_scaled_xtest.npy")
    ytrain = np.load("cifar10_scaled_ytrain.npy")
    ytest = np.load("cifar10_scaled_ytest.npy")

#  Convert labels to one-hot vectors
y_train = keras.utils.to_categorical(ytrain, num_classes)
y_test = keras.utils.to_categorical(ytest, num_classes)

#  The original VGG13 model using the Keras functional API
inp = Input(input_shape)
_ = ConvBlock(inp, 64)
_ = ConvBlock(_,  128)
_ = ConvBlock(_,  256)
_ = ConvBlock(_,  512)
_ = ConvBlock(_,  512)
_ = Flatten()(_)
_ = DenseBlock(_, 4096)
_ = DenseBlock(_, 4096)
_ = Dense(1000)(_)  # no dropout on third layer, see paper
_ = ReLU()(_)
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

