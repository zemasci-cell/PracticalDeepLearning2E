#
#  file:  sentiment_train.npy
#
#  Train a model using the sentiment embeddings
#
#  RTK, 27-Feb-2024
#  Last update:  28-Feb-2024
#
#  Results (nomic-embed-text) (64 mb 30 ep)
#   Test set accuracy: 0.8085, MCC: 0.6167
#   [[782 194]
#    [189 835]]
#
################################################################

import os
import sys
import pickle
import numpy as np
from sklearn.metrics import matthews_corrcoef
import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, ReLU, Dropout
from tensorflow.keras.layers import LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import Reshape, Conv2D, GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Flatten
import numpy as np

def ConfusionMatrix(pred, y):
    """Return a confusion matrix"""
    cm = np.zeros((2,2), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc


#  Command line
if (len(sys.argv) == 1):
    print()
    print("sentiment_train <minibatch> <epochs> <outdir>")
    print()
    print("  <minibatch>    -  minibatch size (e.g. 128)")
    print("  <epochs>       -  number of training epochs (e.g. 16)")
    print("  <outdir>       -  output file directory (overwritten)")
    print()
    exit(0)

batch_size = int(sys.argv[1]) 
epochs = int(sys.argv[2])
outdir = sys.argv[3]

#  Load the datasets -- run sentiment_dataset.py first
x_train = np.load("sentiment140_xtrain.npy")
ytrain  = np.load("sentiment140_ytrain.npy")
x_test  = np.load("sentiment140_xtest.npy")
ytest   = np.load("sentiment140_ytest.npy")

#  Scale [0,1]-ish
x_train = (x_train + 7) / 20
x_test = (x_test + 7) / 20

#  Build the classifier
inp = Input((x_train.shape[1],))
_ = Dense(512)(inp)
_ = LeakyReLU(0.2)(_)
_ = Dropout(0.3)(_)
_ = Dense(256)(_)
_ = LeakyReLU(0.2)(_)
_ = Dropout(0.3)(_)
outp = Dense(1, activation='sigmoid')(_)

model = Model(inputs=inp, outputs=outp)
model.summary()

#  Compile and train
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, ytest))

#  Results
tloss = history.history['loss']
vloss = history.history['val_loss']
terr = 1.0 - np.array(history.history['accuracy'])
verr = 1.0 - np.array(history.history['val_accuracy'])
d = [tloss,vloss,terr,verr]
os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
pickle.dump(d, open(outdir+"/results.pkl", "wb"))
prob = model.predict(x_test, verbose=0)
plabel = np.zeros(len(prob), dtype="uint8")
plabel[np.where(prob >= 0.5)[0]] = 1
cm, acc = ConfusionMatrix(plabel, ytest)
mcc = matthews_corrcoef(ytest, plabel)
s = 'Test set accuracy: %0.4f, MCC: %0.4f' % (acc,mcc)
with open(outdir+"/accuracy_mcc.txt", "w") as f:
    f.write(s+"\n")
np.save(outdir+"/confusion_matrix.npy", cm)
np.save(outdir+"/predictions.npy", plabel)
model.save(outdir+"/model.keras")
print(s)
print(cm)

