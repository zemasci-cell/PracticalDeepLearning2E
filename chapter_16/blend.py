#
#  file:  blend.py
#
#  Self-supervised learning via blending and rotation
#
#  RTK, 06-Jan-2024
#  Last update:  10-Jan-2024
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
import numpy as np

def ConfusionMatrix(pred, y):
    """Return a confusion matrix"""
    cm = np.zeros((3,3), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

def ProcessDataset(b, seed=359):
    """Use blending and rotation to build a dataset"""
    np.random.seed(seed)
    x, y, k = [], [], 0
    while (k < len(b)-1):
        l = np.random.randint(0,3)
        ans = np.zeros(b.shape[1:], dtype="uint8")
        if (l==0):
            for c in range(3):
                ans[:,:,c] = (b[k,:,:,c]/2 + b[k+1,:,:,c]/2).astype("uint8")
            k += 2
        elif (l==1):
            ans[:,:,:] = b[k,:,:,:]
            k += 1
        else:
            for c in range(3):
                ans[:,:,c] = np.rot90(b[k,:,:,c],2)
            k += 1
        x.append(ans)
        t = [0]*3; t[l] = 1
        y.append(t)
    np.random.seed()
    return np.array(x) / 255.0, np.array(y)


#  Command line
if (len(sys.argv) == 1):
    print()
    print("blend <minibatch> <epochs> <outdir>")
    print()
    print("  <minibatch> - minibatch size (e.g. 128)")
    print("  <epochs>    - number of training epochs (e.g. 10)")
    print("  <outdir>    - output file directory (overwritten)")
    print()
    exit(0)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
outdir = sys.argv[3]

#  Other parameters
num_classes = 3
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

#  Construct a dataset -- [0,1], one-hot encoded labels
b = np.load("../data/cifar100/xtrain.npy")
x,y = ProcessDataset(b, seed=8675309)
n = int(len(x)*0.9)
xtrn,ytrn = x[:n],y[:n]
xtst,ytst = x[n:],y[n:]

#  Base model
inp = Input(input_shape)
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
outp = Flatten()(_)
base = Model(inputs=inp, outputs=outp)
base.summary()

#  Store the untrained base model
os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
base.save(outdir+"/raw.keras")

#  Classification head to predict rotation class
_ = Dense(num_classes)(base.output)
outp = Softmax()(_)
model = Model(inputs=base.input, outputs=outp)
model.summary()

#  Compile and train
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(xtrn, ytrn,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xtst, ytst))

#  Results
tloss = history.history['loss']
vloss = history.history['val_loss']
terr = 1.0 - np.array(history.history['accuracy'])
verr = 1.0 - np.array(history.history['val_accuracy'])
d = [tloss,vloss,terr,verr]
pickle.dump(d, open(outdir+"/results.pkl", "wb"))
pred = model.predict(xtst, verbose=0)
plabel = np.argmax(pred, axis=1)
ytest = np.argmax(ytst, axis=1)
cm, acc = ConfusionMatrix(plabel, ytest)
mcc = matthews_corrcoef(ytest, plabel)
s = 'Test set accuracy: %0.4f, MCC: %0.4f' % (acc,mcc)
with open(outdir+"/accuracy_mcc.txt", "w") as f:
    f.write(s+"\n")
np.save(outdir+"/confusion_matrix.npy", cm)
np.save(outdir+"/predictions.npy", pred)
base.save(outdir+"/model.keras")  # save the trained base model
print(s)
print(cm)

