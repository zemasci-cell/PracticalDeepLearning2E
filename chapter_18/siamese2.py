#
#  file:  siamese2.py
#
#  Self-supervised learning with a Siamese network
#
#  RTK, 12-Jan-2024
#  Last update:  14-Jan-2024
#
################################################################

import os
import sys
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.layers import Add 
from tensorflow.keras.layers import BatchNormalization
import numpy as np
from PIL import Image, ImageEnhance

def ConfusionMatrix(p,y):
    cm = np.zeros((2,2), dtype="uint16")
    for i in range(len(y)):
        cm[y[i],p[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

def Augment(im):
    """Augment the given image"""
    img = Image.fromarray(im)
    choice = np.random.randint(0,8)
    if (choice == 0):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif (choice == 1):
        r = -3 + 6*np.random.random()
        img = img.rotate(r, resample=Image.BILINEAR)
    elif (choice == 2):
        i = np.array(img)
        i = np.roll(i, np.random.randint(-3,4), axis=1)
        i = np.roll(i, np.random.randint(-3,4), axis=0)
        img = Image.fromarray(i)
    elif (choice == 3):
        r = 0.7 + 0.5*np.random.random()
        t = np.zeros(im.shape, dtype="uint8")
        t[:,:,0] = np.clip(np.array(img)[:,:,0]**r, 0, 255)
        t[:,:,1] = np.clip(np.array(img)[:,:,1]**r, 0, 255)
        t[:,:,2] = np.clip(np.array(img)[:,:,2]**r, 0, 255)
        img = Image.fromarray(t) 
    elif (choice == 4):
        r = 0.5 + 1.5*np.random.random()
        img = ImageEnhance.Brightness(img).enhance(r)
    elif (choice == 5):
        r = 0.5 + 1.5*np.random.random()
        img = ImageEnhance.Contrast(img).enhance(r)
    elif (choice == 6):
        r = 0.5 + 3.0*np.random.random()
        img = ImageEnhance.Sharpness(img).enhance(r)
    elif (choice == 7):
        r = 0.0 + 2.0*np.random.random()
        img = ImageEnhance.Color(img).enhance(r)
    return np.array(img)

def SiameseDataset(b, seed=359):
    """Create a Siamese dataset from unlabeled CIFAR-100 images"""
    np.random.seed(seed)
    x0, x1, y = [], [], []
    for i in range(len(b)):
        if (np.random.random() < 0.5):
            #  positive pair
            x0.append(Augment(b[i]))
            x1.append(Augment(b[i]))
            y.append(1)
        else:
            #  negative pair
            x0.append(Augment(b[i]))
            k = np.random.randint(0,len(b))
            x1.append(Augment(b[k]))
            y.append(0)
    np.random.seed()
    return np.array(x0)/255, np.array(x1)/255, np.array(y)

def EncoderModel():
    """Encoder model -- rotation base"""
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
    outp = Flatten()(_)
    return Model(inputs=inp, outputs=outp, name="encoder")


#  Command line
if (len(sys.argv) == 1):
    print()
    print("siamese2 <minibatch> <epochs> <outdir>")
    print()
    print("  <minibatch>    -  minibatch size (e.g. 128)")
    print("  <epochs>       -  number of training epochs (e.g. 16)")
    print("  <outdir>       -  output file directory (overwritten)")
    print()
    exit(0)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
outdir = sys.argv[3]

#  Construct the Siamese dataset
b = np.load("../data/cifar100/xtrain.npy")
x0,x1,y = SiameseDataset(b, seed=8675309)
n = int(len(x0)*0.9)
x0trn,x1trn,ytrn = x0[:n],x1[:n],y[:n]
x0tst,x1tst,ytst = x0[n:],x1[n:],y[n:]

#  Create the model
encoder = EncoderModel()  # keep this for downstream tasks

inp_a, inp_b = Input((32,32,3)), Input((32,32,3))
proc_a, proc_b = encoder(inp_a), encoder(inp_b)

_ = Add()([proc_a, proc_b])
_ = Dense(1024)(_)
_ = BatchNormalization()(_)
_ = ReLU()(_)
outp = Dense(1, activation='sigmoid')(_)
model = Model([inp_a, inp_b], outp)
model.summary()

# Compile and train
model.compile(loss='binary_crossentropy', 
              optimizer=keras.optimizers.Adam(), 
              metrics=['accuracy'])

history = model.fit([x0trn, x1trn], ytrn,
          batch_size=batch_size,
          epochs=epochs, verbose=1,
          validation_data=([x0tst,x1tst], ytst))

#  Results
os.system("mkdir %s 2>/dev/null" % outdir)
tloss = history.history['loss']
vloss = history.history['val_loss']
terr = 1.0 - np.array(history.history['accuracy'])
verr = 1.0 - np.array(history.history['val_accuracy'])
d = [tloss,vloss,terr,verr]
pickle.dump(d, open(outdir+"/results.pkl", "wb"))
pred = model.predict([x0tst,x1tst], verbose=0).squeeze()
plabel = np.zeros(len(ytst), dtype="uint8")
plabel[np.where(pred > 0.5)] = 1
cm, acc = ConfusionMatrix(plabel, ytst)
mcc = matthews_corrcoef(ytst, plabel)
s = 'Test set accuracy: %0.4f, MCC: %0.4f' % (acc,mcc)
with open(outdir+"/accuracy_mcc.txt", "w") as f:
    f.write(s+"\n")
np.save(outdir+"/confusion_matrix.npy", cm)
np.save(outdir+"/predictions.npy", pred)
encoder.save(outdir+"/model.keras")  # save the trained encoder
print(s)
print(cm)

