#
#  file:  fine-tune_vgg16_1_percent_results.py
#
#  Evaluate 1 percent models (tested against full CIFAR-10 test set)
#
#  RTK, 11-Dec-2023
#  Last update:  11-Dec-2023
#
################################################################

import numpy as np
from sklearn.metrics import matthews_corrcoef

def ConfusionMatrix(pred, y):
    """Return a confusion matrix"""
    cm = np.zeros((10,10), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

def GetResults(run, aug=False):
    if (aug):
        f = "results/fine-tune_vgg16_fraction_sgd_32_60_001_aug_run%d/accuracy_mcc.txt" % run
    else:
        f = "results/fine-tune_vgg16_fraction_sgd_32_60_001_run%d/accuracy_mcc.txt" % run
    t = open(f).read()[:-1].split()
    return float(t[3][:-1]), float(t[5])

def gmean(z):
    return np.prod(z)**(1/len(z))


#  Average performance - unaugmented
acc,mcc = [],[]
for i in range(5):
    a,m = GetResults(i)
    acc.append(a)
    mcc.append(m)
acc = np.array(acc)
mcc = np.array(mcc)

am,ae = acc.mean(), acc.std(ddof=1) / np.sqrt(len(acc))
mm,me = mcc.mean(), mcc.std(ddof=1) / np.sqrt(len(mcc))

print()
print("Average model performance (unaugmented):")
print("    ACC: %0.5f +/- %0.5f, MCC: %0.5f +/- %0.5f" % (am,ae,mm,me))
print()

#  Ensemble performance - unaugmented
p0 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_run0/predictions.npy")
p1 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_run1/predictions.npy")
p2 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_run2/predictions.npy")
p3 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_run3/predictions.npy")
p4 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_run4/predictions.npy")

gm = np.zeros((10000,10))
for k in range(10):
    for i in range(10000):
        p = np.array([p0[i,k],p1[i,k],p2[i,k],p3[i,k],p4[i,k]])
        gm[i,k] = gmean(p)

pred = np.argmax(gm, axis=1)
ytest = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()
cm, acc = ConfusionMatrix(pred, ytest)
mcc = matthews_corrcoef(ytest, pred)

print("Ensemble model performance (unaugmented):")
print("    ACC: %0.5f, MCC: %0.5f" % (acc,mcc))
print()
print(cm)

#  Average performance - augmented
acc,mcc = [],[]
for i in range(5):
    a,m = GetResults(i, aug=True)
    acc.append(a)
    mcc.append(m)
acc = np.array(acc)
mcc = np.array(mcc)

am,ae = acc.mean(), acc.std(ddof=1) / np.sqrt(len(acc))
mm,me = mcc.mean(), mcc.std(ddof=1) / np.sqrt(len(mcc))

print()
print("Average model performance (augmented):")
print("    ACC: %0.5f +/- %0.5f, MCC: %0.5f +/- %0.5f" % (am,ae,mm,me))
print()

#  Ensemble performance - unaugmented
p0 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_aug_run0/predictions.npy")
p1 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_aug_run1/predictions.npy")
p2 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_aug_run2/predictions.npy")
p3 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_aug_run3/predictions.npy")
p4 = np.load("results/fine-tune_vgg16_fraction_sgd_32_60_001_aug_run4/predictions.npy")

gm = np.zeros((10000,10))
for k in range(10):
    for i in range(10000):
        p = np.array([p0[i,k],p1[i,k],p2[i,k],p3[i,k],p4[i,k]])
        gm[i,k] = gmean(p)

pred = np.argmax(gm, axis=1)
cm, acc = ConfusionMatrix(pred, ytest)
mcc = matthews_corrcoef(ytest, pred)

print("Ensemble model performance (augmented):")
print("    ACC: %0.5f, MCC: %0.5f" % (acc,mcc))
print()
print(cm)
print()


