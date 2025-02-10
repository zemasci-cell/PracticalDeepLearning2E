#
#  files: features_mlp.py
#
#  Train using the full CIFAR-10 dataset
#
#  RTK, 17-Dec-2023
#  Last update:  17-Dec-2023
#
################################################################

import numpy as np
from sklearn.neural_network import MLPClassifier

x_train = np.load("features_cifar10_block_train.npy")
x_test = np.load("features_cifar10_block_test.npy")

y_train = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()
y_test = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

m = x_train.mean(axis=0)
s = x_train.std(ddof=1, axis=0)
x_train = (x_train - m) / s
x_test = (x_test - m) / s

clf = MLPClassifier(hidden_layer_sizes=(512,256))
clf.fit(x_train, y_train)
print("score = %0.4f" % clf.score(x_test, y_test))

