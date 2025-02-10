#
#  file:  LeNet5.py
#
#  Implement LeNet5 using both the sequential and functional APIs
#
#  RTK, 26-Aug-2023
#  Last update:  26-Aug-2023
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
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pylab as plt


def ConfusionMatrix(pred, y):
    """Return a confusion matrix"""
    cm = np.zeros((10,10), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

#  Command line
if (len(sys.argv) == 1):
    print()
    print("lenet5 <minibatch> <epochs> <useBN> <type>")
    print()
    print("  <minibatch>  -  minibatch size (e.g. 128)")
    print("  <epochs>     -  number of training epochs (e.g. 16)")
    print("  <useBN>      -  0=no batch norm, 1=use batch norm (keep bias to simplify code)")
    print("  <type>       -  0=sequential, 1=functional")
    print()
    exit(0)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
useBN = True if (int(sys.argv[3])) else False
mtype = int(sys.argv[4])

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

if (mtype):
    #  functional API
    inp = Input(input_shape)
    _ = Conv2D(6, (3,3))(inp)
    if (useBN): _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = MaxPooling2D((2,2))(_)
    _ = Conv2D(16, (3,3))(_)
    if (useBN): _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = MaxPooling2D((2,2))(_)
    _ = Conv2D(120, (3,3))(_)
    if (useBN): _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = Flatten()(_)
    _ = Dense(84)(_)
    if (useBN): _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = Dropout(0.5)(_)
    _ = Dense(num_classes)(_)
    outp = Softmax()(_)
    model = Model(inputs=inp, outputs=outp)
else:
    #  sequential API
    model = Sequential()
    model.add(Conv2D(6, (3,3), input_shape=input_shape, activation='relu'))
    if (useBN): model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(16, (3,3), activation='relu'))
    if (useBN): model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(120, (3,3), activation='relu'))
    if (useBN): model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(84))
    if (useBN): model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

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

#  Test
pred = model.predict(x_test, verbose=0)
plabel = np.argmax(pred, axis=1)
cm, acc = ConfusionMatrix(plabel, ytest)
mcc = matthews_corrcoef(ytest, plabel)
print(cm)
print('Test set accuracy: %0.4f, MCC: %0.4f' % (acc,mcc))
print()

terr = 1.0 - np.array(history.history['accuracy'])
verr = 1.0 - np.array(history.history['val_accuracy'])
x = list(range(epochs))
plt.plot(x, terr, linestyle='solid', linewidth=0.5, color='k', label='train')
plt.plot(x, verr, linestyle='solid', color='k', label='test')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("lenet5_plot.png", dpi=300)
plt.savefig("lenet5_plot.eps", dpi=300)

