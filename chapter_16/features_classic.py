#
#  files: features_classic.py
#
#  Use the VGG16 embedded CIFAR-10 features with classical models
#
#  RTK, 01-Dec-2023
#  Last update:  02-Dec-2023
#
################################################################

import sys
import numpy as np
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def run(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    print("    score = %0.4f" % clf.score(x_test, y_test))
    print()

if (len(sys.argv) == 1):
    print()
    print("features_classic raw|vgg16|layer10")
    print()
    print("  raw     = raveled RGB")
    print("  vgg16   = VGG16 embedded features")
    print("  layer10 = VGG16 layer 10 features")
    print()
    exit(0)

if (sys.argv[1].lower() == 'raw'):
    x_train = np.load("../data/cifar10/cifar10_train_images.npy").reshape((50000,3072))
    x_test = np.load("../data/cifar10/cifar10_test_images.npy").reshape((10000,3072))
elif (sys.argv[1].lower() == 'layer10'):
    x_train = np.load("features_cifar10_block_train.npy")
    x_test = np.load("features_cifar10_block_test.npy")
else:
    x_train = np.load("features_cifar10_train.npy")
    x_test = np.load("features_cifar10_test.npy")

y_train = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()
y_test = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

np.random.seed(6502)
N = 20000
idx = np.argsort(np.random.random(len(x_train)))[:N]
x_train = x_train[idx]
y_train = y_train[idx]
N = 5000
idx = np.argsort(np.random.random(len(x_test)))[:N]
x_test = x_test[idx]
y_test = y_test[idx]
np.random.seed()

print("Naive Bayes classifier (Gaussian):")
run(x_train, y_train, x_test, y_test, GaussianNB())
print("Nearest centroid:")
run(x_train, y_train, x_test, y_test, NearestCentroid())
print("k-NN classifier (k=3):")
run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
print("k-NN classifier (k=7):")
run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=7))
print("Random forest classifier (estimators=100):")
run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=100))
print("Random forest classifier (estimators=400):")
run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=400))

#  MLP wants standardized inputs
m = x_train.mean(axis=0)
s = x_train.std(ddof=1, axis=0)
x_train = (x_train - m) / s
x_test = (x_test - m) / s

print("MLP classifier (512,256):")
run(x_train, y_train, x_test, y_test, MLPClassifier(hidden_layer_sizes=(512,256)))

