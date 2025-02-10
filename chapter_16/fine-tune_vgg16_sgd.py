#
#  file:  fine-tune_vgg16_sgd.py
#
#  Fine-tune CIFAR-10 using pretrained ImageNet
#  weights and VGG16
#
#  RTK, 28-Nov-2023
#  Last update: 10-Dec-2023
#
################################################################

import os
import sys
import pickle
from sklearn.metrics import matthews_corrcoef
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, ReLU, Softmax
from tensorflow.keras.layers import Dropout, Flatten
from PIL import Image

def Augment(im):
    img = Image.fromarray(im)
    if (np.random.random() < 0.5):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if (np.random.random() < 0.3333):
        r = 3*np.random.random()-3
        img = img.rotate(r, resample=Image.BILINEAR)
    else:
        i = np.array(img)
        x = np.random.randint(-3,4)
        y = np.random.randint(-3,4)
        i = np.roll(i, np.random.randint(-3,4), axis=1)
        i = np.roll(i, np.random.randint(-3,4), axis=0)
        img = Image.fromarray(i)
    return np.array(img)

def AugmentDataset(x,y):
    """Augment the dataset 10x"""
    factor = 10
    newx = np.zeros((x.shape[0]*factor, 32,32,3), dtype="uint8")
    newy = np.zeros(y.shape[0]*factor, dtype="uint8")
    k=0 
    for i in range(x.shape[0]):
        im = Image.fromarray(x[i,:])
        newx[k,...] = np.array(im)
        newy[k] = y[i]
        k += 1
        for j in range(factor-1):
            newx[k,...] = Augment(x[i,:])
            newy[k] = y[i]
            k += 1
    idx = np.argsort(np.random.random(newx.shape[0]))
    return newx[idx], newy[idx]

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
    print("fine-tune_vgg16 <minibatch> <epochs> <weights> <freeze> <outdir> [<fraction> <augment>]")
    print()
    print("  <minibatch> -  minibatch size (e.g. 128)")
    print("  <epochs>    -  number of fine-tuning epochs (e.g. 12)")
    print("  <weights>   -  'none' | 'imagenet' (use pretrained weights)")
    print("  <freeze>    -  freeze blocks through <freeze>, e.g. 5 for all")
    print("  <outdir>    -  output file directory (overwritten)")
    print("  <fraction>  -  fraction of full training set to use (def=1.0)")
    print("  augment     -  if present, 10x augmentation of <fraction>")
    print()
    exit(0)

#  Training parameters
batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
weights = None if (sys.argv[3].lower()=='none') else 'imagenet'
freeze = int(sys.argv[4])
outdir = sys.argv[5]
fraction = float(sys.argv[6]) if (len(sys.argv)>=7) else 1.0
augment = len(sys.argv) == 8

#  Load CIFAR-10
x_train = np.load("../data/cifar10/cifar10_train_images.npy")
ytrain  = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()
x_test  = np.load("../data/cifar10/cifar10_test_images.npy")
ytest   = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

#  Choose desired fraction, augment 10x if selected
if (fraction != 1.0):
    np.random.seed(73939133)
    n = int(len(x_train)*fraction)
    idx = np.argsort(np.random.random(len(x_train)))[:n]
    x_train = x_train[idx]
    ytrain = ytrain[idx]
    if (augment):
        x_train, ytrain = AugmentDataset(x_train, ytrain)
    np.random.seed()

#  Preprocess the inputs
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#  Convert labels to one-hot vectors
num_classes = ytrain.max() + 1
y_train = keras.utils.to_categorical(ytrain, num_classes)
y_test = keras.utils.to_categorical(ytest, num_classes)

#  Build the base model
inp = Input(shape=(32,32,3))
if (weights == 'imagenet'):
    base = VGG16(input_tensor=inp, include_top=False, weights='imagenet')

    #  Freeze the base model weights according to <freeze>
    idx = [0,4,7,11,15,19][freeze]
    if (idx > 0):
        for layer in base.layers[:idx]:
            layer.trainable = False
else:
    base = VGG16(input_tensor=inp, include_top=False, weights=None)

#  Add the trainable top layers
_ = Flatten()(base.output)
_ = Dense(1024)(_)
_ = BatchNormalization()(_)
_ = ReLU()(_)
_ = Dropout(0.5)(_)
_ = Dense(1024)(_)
_ = BatchNormalization()(_)
_ = ReLU()(_)
_ = Dropout(0.5)(_)
_ = Dense(num_classes)(_)
outp = Softmax()(_)

#  The complete VGG16 model
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
          validation_data=(x_test[:1000], y_test[:1000]))

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

