#
#  file:  features_cifar10_vgg16.py
#
#  Use pretrained VGG16 to create CIFAR-10 features for 
#  downstream models
#
#  RTK, 01-Dec-2023
#  Last update: 01-Dec-2023
#
################################################################

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten

#  Load CIFAR-10
x_train = np.load("../data/cifar10/cifar10_train_images.npy")
x_test  = np.load("../data/cifar10/cifar10_test_images.npy")

#  Preprocess the inputs
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#  Build the model using pretrained ImageNet weights
inp = Input(shape=(32,32,3))
base = VGG16(input_tensor=inp, include_top=False, weights='imagenet')
outp = Flatten()(base.output)
model = Model(inputs=inp, outputs=outp)

#  Generate output feature vectors
xtrn = model.predict(x_train, verbose=0)
xtst = model.predict(x_test, verbose=0)

#  And write them to disk
np.save("features_cifar10_train.npy", xtrn)
np.save("features_cifar10_test.npy", xtst)

