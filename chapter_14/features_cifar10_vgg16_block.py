#
#  file:  features_cifar10_vgg16_block.py
#
#  Use pretrained VGG16 to create CIFAR-10 features for 
#  downstream models from layer 10 
#
#  RTK, 06-Dec-2023
#  Last update: 06-Dec-2023
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

#  Ancillary model to get the activations from layer 10
layer = base.layers[10]
model = Model(inputs=base.input, outputs=layer.output)

#  Generate output feature vectors
xtrn = model.predict(x_train, verbose=0).reshape((50000,4*4*256))
xtst = model.predict(x_test, verbose=0).reshape((10000,4*4*256))

#  And write them to disk
np.save("features_cifar10_block_train.npy", xtrn)
np.save("features_cifar10_block_test.npy", xtst)

