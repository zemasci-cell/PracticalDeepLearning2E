#
#  file:  features_anomaly.py
#
#  Use CIFAR-10 features for anomaly detection with
#  a one-class SVM
#
#  RTK, 18-Dec-2023
#  Last update:  18-Dec-2023
#
################################################################

import numpy as np
from sklearn.svm import OneClassSVM

def KeepAnimals(x,y):
    animal, frog, airplane = [], [], []
    for i in range(len(y)):
        if (y[i] in [2,3,4,5,7]):
            animal.append(x[i])
        elif (y[i] == 6):
            frog.append(x[i])
        elif (y[i] == 0):
            airplane.append(x[i])
    return np.array(animal), np.array(frog), np.array(airplane)


#  Load the CIFAR-10 raw and layer 10 features
xtrn0 = np.load("../data/cifar10/cifar10_train_images.npy").reshape((50000,3072))
xtst0 = np.load("../data/cifar10/cifar10_test_images.npy").reshape((10000,3072))
xtrn1 = np.load("features_cifar10_block_train.npy")
xtst1 = np.load("features_cifar10_block_test.npy")
ytrn = np.load("../data/cifar10/cifar10_train_labels.npy").squeeze()
ytst = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

#  Keep only the animals setting frogs and airplanes aside
xtrn0, xtstf0, xtsta0 = KeepAnimals(xtrn0,ytrn)
xtrn1, xtstf1, xtsta1 = KeepAnimals(xtrn1,ytrn)
xtst0, _, _ = KeepAnimals(xtst0,ytst)
xtst1, _, _ = KeepAnimals(xtst1,ytst)

#  Scramble the training data and keep a fraction
np.random.seed(1066)
idx = np.argsort(np.random.random(len(xtrn0)))
xtrn0 = xtrn0[idx][:10000]
xtrn1 = xtrn1[idx][:10000]
np.random.seed()

#  Learn a one-class SVM using non-frogs as normal
clf0 = OneClassSVM(nu=0.03)
clf0.fit(xtrn0)
clf1 = OneClassSVM(nu=0.03)
clf1.fit(xtrn1)

#  Pass frogs, airplanes, and animals through the models
frog0 = clf0.predict(xtstf0)
airp0 = clf0.predict(xtsta0)
anim0 = clf0.predict(xtst0)
frog1 = clf1.predict(xtstf1)
airp1 = clf1.predict(xtsta1)
anim1 = clf1.predict(xtst1)

#  Compare the performance of the raw and layer 10 features
bfr0 = np.bincount(frog0+1, minlength=3)
bai0 = np.bincount(airp0+1, minlength=3)
ban0 = np.bincount(anim0+1, minlength=3)

bfr1 = np.bincount(frog1+1, minlength=3)
bai1 = np.bincount(airp1+1, minlength=3)
ban1 = np.bincount(anim1+1, minlength=3)

print()
print("Raw features:")
print("    animal anomaly  : %4d (%0.5f)" % (ban0[0], ban0[0] / ban0.sum()))
print("    frog anomaly    : %4d (%0.5f)" % (bfr0[0], bfr0[0] / bfr0.sum()))
print("    airplane anomaly: %4d (%0.5f)" % (bai0[0], bai0[0] / bai0.sum()))
print()
print("Layer 10 features:")
print("    animal anomaly  : %4d (%0.5f)" % (ban1[0], ban1[0] / ban1.sum()))
print("    frog anomaly    : %4d (%0.5f)" % (bfr1[0], bfr1[0] / bfr1.sum()))
print("    airplane anomaly: %4d (%0.5f)" % (bai1[0], bai1[0] / bai1.sum()))
print()

