#
#  file:  fine-tune_mobilenet.py
#
#  Fine-tune CIFAR-10 using pretrained ImageNet
#  weights and MobileNet
#
#  RTK, 02-Dec-2023
#  Last update: 02-Dec-2023
#
################################################################

import os
import sys
import pickle
from sklearn.metrics import matthews_corrcoef
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, ReLU, Softmax
from tensorflow.keras.layers import Dropout, Flatten

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
    print("fine-tune_mobilenet <minibatch> <epochs> <weights> <unfreeze> <outdir>")
    print()
    print("  <minibatch> -  minibatch size (e.g. 128)")
    print("  <epochs>    -  number of fine-tuning epochs (e.g. 12)")
    print("  <weights>   -  'none' | 'imagenet' (use pretrained weights)")
    print("  <unfreeze>  -  unfreeze the top <unfreeze> depthwise blocks")
    print("                 0=freeze all blocks, -1=freeze nothing")
    print("  <outdir>    -  output file directory (overwritten)")
    print()
    exit(0)

#  Training parameters
batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
weights = None if (sys.argv[3].lower()=='none') else 'imagenet'
unfreeze = int(sys.argv[4])
outdir = sys.argv[5]

#  Load CIFAR-10
x_train = np.load("../data/cifar10/cifar10_train_images.npy")
ytrain  = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()
x_test  = np.load("../data/cifar10/cifar10_test_images.npy")
ytest   = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

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
    base = MobileNet(input_tensor=inp, include_top=False, weights=weights)

    if (unfreeze != -1):
        #  freeze all layers
        for layer in base.layers:
            layer.trainable = False

        #  unfreeze top-n requested blocks (n < 13)
        for i in range(1,6*unfreeze+1):
            base.layers[-i].trainable = True
else:
    base = MobileNet(input_tensor=inp, include_top=False, weights=None)

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

#  The complete model
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

