#  Estimate the intrinsic dimensionality of MNIST and FMNIST
import numpy as np
import tensorflow.keras as keras
from skdim.id import MLE

(_,_), (x,_) = keras.datasets.mnist.load_data()
m_id = int(MLE().fit_transform(x.reshape((10000,28*28))) + 0.5)

(_,_), (x,_) = keras.datasets.fashion_mnist.load_data()
f_id = int(MLE().fit_transform(x.reshape((10000,28*28))) + 0.5)

print("Intrinsic dimensionality (MLE):")
print("    MNIST: %d" % m_id)
print("   FMNIST: %d" % f_id)
print()

