#
#  file:  rot_fine-tune.py
#
#  Fine-tune the self-supervised rot model on CIFAR-10
#
#  RTK, 08-Jan-2024
#  Last update: 08-Jan-2024
#
################################################################

import os
import sys
import pickle
from sklearn.metrics import matthews_corrcoef
import numpy as np
import tensorflow.keras as keras
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization, ReLU, Softmax
from tensorflow.keras.layers import Dropout, Flatten

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
    print("rot_fine-tune <base> blk2|blk3|scratch <minibatch> <epochs> <outdir> [<fraction> [augment]]")
    print()
    print("  <base>            -  rot self-supervised pretrained model")
    print("  blk2|blk3|scratch -  freeze through block 2 or block 3 or from scratch")
    print("  <minibatch>       -  minibatch size (e.g. 128)")
    print("  <epochs>          -  number of fine-tuning epochs (e.g. 12)")
    print("  <outdir>          -  output file directory (overwritten)")
    print("  <fraction>        -  fraction of full training set to use (def=1.0)")
    print("  augment           -  if present, 10x augment training set (fraction must be given)")
    print()
    exit(0)

#  Training parameters
bname = sys.argv[1]
blk = sys.argv[2].lower()
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
outdir = sys.argv[5]
fraction = float(sys.argv[6]) if (len(sys.argv)>=7) else 1.0
augment = True if (len(sys.argv)==8) else False

#  Load CIFAR-10 and scale
x_train = np.load("../data/cifar10/cifar10_train_images.npy")
ytrain  = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()
x_test  = np.load("../data/cifar10/cifar10_test_images.npy") / 255.0
ytest   = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

#  Select desired fraction
if (fraction != 1.0):
    np.random.seed(73939133)
    n = int(len(x_train)*fraction)
    idx = np.argsort(np.random.random(len(x_train)))[:n]
    x_train = x_train[idx]
    ytrain = ytrain[idx]
    np.random.seed()

#  Augment the dataset
if (augment):
    x_train, ytrain = AugmentDataset(x_train, ytrain)

#  Now scale training set
x_train = x_train / 255.0

#  Convert labels to one-hot vectors
num_classes = ytrain.max() + 1
y_train = keras.utils.to_categorical(ytrain, num_classes)
y_test = keras.utils.to_categorical(ytest, num_classes)

if (blk == 'scratch'):
    #  base portion:
    inp = Input((32,32,3))
    _ = Conv2D(32, (3,3), padding='same')(inp)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = MaxPooling2D((2,2))(_)
    _ = Conv2D(64, (3,3), padding='same')(_)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = MaxPooling2D((2,2))(_)
    _ = Conv2D(128, (3,3), padding='same')(_)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = MaxPooling2D((2,2))(_)

    #  top portion:
    _ = Flatten()(_)
    _ = Dense(1024)(_)
    _ = BatchNormalization(name="bn1")(_)
    _ = ReLU(name="relu1")(_)
    _ = Dropout(0.5)(_)
    _ = Dense(1024)(_)
    _ = BatchNormalization(name="bn2")(_)
    _ = ReLU(name="relu2")(_)
    _ = Dropout(0.5)(_)
    _ = Dense(num_classes)(_)
    outp = Softmax()(_)
    model = Model(inputs=inp, outputs=outp)
else:
    #  Load the base model and freeze through selected block
    base = load_model(bname)

    limit = 8 if (blk == 'blk2') else 12 
    for layer in base.layers[:limit]:
        layer.trainable = False

    _ = Flatten()(base.layers[limit].output)
    _ = Dense(1024)(_)
    _ = BatchNormalization(name="bn1")(_)
    _ = ReLU(name="relu1")(_)
    _ = Dropout(0.5)(_)
    _ = Dense(1024)(_)
    _ = BatchNormalization(name="bn2")(_)
    _ = ReLU(name="relu2")(_)
    _ = Dropout(0.5)(_)
    _ = Dense(num_classes)(_)
    outp = Softmax()(_)

    model = Model(inputs=base.input, outputs=outp)
model.summary()

#  Compile and train
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
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
#model.save(outdir+"/model.keras")  # uncomment if you want the model
print(s)
print(cm)

