#  Train a 512x256 MLP on raw CIFAR-10 features
#
#  Run: MLP(512,256) on raw CIFAR-10 score = 0.4311
#
import numpy as np
from sklearn.neural_network import MLPClassifier

#  Load the raw features
M = 10_000
x_train = np.load("../data/cifar10/cifar10_train_images.npy").reshape((50000,32*32*3))[:M]
x_test = np.load("../data/cifar10/cifar10_test_images.npy").reshape((10000,32*32*3))[:M]
y_train = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()[:M]
y_test = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()[:M]

x_train = x_train / 255
x_test = x_test / 255

#  Define, train, score
clf = MLPClassifier(hidden_layer_sizes=(512,256))
clf.fit(x_train, y_train)
print("MLP(512,256) on raw CIFAR-10 score = %0.4f" % clf.score(x_test, y_test))
print()

