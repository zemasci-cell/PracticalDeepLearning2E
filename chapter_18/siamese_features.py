#
#  file:  siamese_features.py
#
#  Visualize the Siamese network features
#
#  RTK, 14-Jan-2024
#  Last update:  25-Jan-2024
#
################################################################

import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance

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

#  Construct the Siamese dataset
b = np.load("../data/cifar100/xtrain.npy")
x0,x1,y = SiameseDataset(b, seed=8675309)
n = int(len(x0)*0.9)
x0,x1,y = x0[n:],x1[n:],y[n:]

#  Create two sets of images -- concatenated and summed
os.system("mkdir feature_images 2>/dev/null")
os.system("mkdir feature_images/concat 2>/dev/null")
os.system("mkdir feature_images/sum 2>/dev/null")

#
#  Concatenated:
#

#  Load the output of siamese_128_10/model.keras
model = load_model("results/siamese_128_10/model.keras")

#  Generate the features
f0 = model.predict(x0, verbose=0)
f1 = model.predict(x1, verbose=0)

#  Reshape the features to present them as images
f0 = f0.reshape((5000,32,64))
f1 = f1.reshape((5000,32,64))

#  Keep a random subset
np.random.seed(271828)
idx = np.argsort(np.random.random(len(y)))[:100]
f0, f1, y = f0[idx], f1[idx], y[idx]

for i in range(len(y)):
    #  concatenate
    i0 = (255*f0[i]/f0[i].max()).astype("uint8")
    i1 = (255*f1[i]/f1[i].max()).astype("uint8")
    img = np.zeros((32,2*64), dtype="uint8")
    img[:,:64] = 255-i0
    img[:,64:] = 255-i1
    if (y[i]):
        Image.fromarray(img).save("feature_images/concat/same_%04d.png" % i)
    else:
        Image.fromarray(img).save("feature_images/concat/diff_%04d.png" % i)

#
#  Summed:
#

#  Load the output of siamese2_128_10/model.keras
model = load_model("results/siamese2_128_10/model.keras")

#  Generate the features
f0 = model.predict(x0, verbose=0)
f1 = model.predict(x1, verbose=0)

#  Reshape the features to present them as images
f0 = f0.reshape((5000,32,64))
f1 = f1.reshape((5000,32,64))

#  Keep a random subset -- same subset as concatenate
np.random.seed(271828)
idx = np.argsort(np.random.random(len(y)))[:100]
f0, f1, y = f0[idx], f1[idx], y[idx]

for i in range(len(y)):
    img = f0[i] + f1[i]
    img = 255-(255*img/img.max()).astype("uint8")
    if (y[i]):
        Image.fromarray(img).save("feature_images/sum/same_%04d.png" % i)
    else:
        Image.fromarray(img).save("feature_images/sum/diff_%04d.png" % i)

