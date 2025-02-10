#
#  files: classic_models.py
#
#  Use the CIFAR-10 features with classical models
#
#  RTK, 07-Jan-2024
#  Last update:  07-Jan-2024
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
    print("classic_models <train> <test>")
    print()
    print("  <train> - CIFAR-10 training features (.npy)")
    print("  <test>  - CIFAR-10 testing features (.npy)")
    print()
    exit(0)

x_train = np.load(sys.argv[1])
x_test = np.load(sys.argv[2])

M = len(x_train)
y_train = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()[:M]
M = len(x_test)
y_test = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()[:M]

print("Nearest centroid:")
run(x_train, y_train, x_test, y_test, NearestCentroid())
print("Naive Bayes classifier (Gaussian):")
run(x_train, y_train, x_test, y_test, GaussianNB())
print("k-NN classifier (k=3):")
run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
print("k-NN classifier (k=7):")
run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=7))
print("Random forest classifier (estimators=100):")
run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=100))
print("Random forest classifier (estimators=400):")
run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=400))
print("MLP classifier (512,256):")
run(x_train, y_train, x_test, y_test, MLPClassifier(hidden_layer_sizes=(512,256)))

