#
#  file: cifar10_cnn_aug_results.py
#
#  Compare training with and without augmentations
#
#  RTK, 12-May-2023
#  Last update:  17-May-2023
#
################################################################

import numpy as np
import matplotlib.pylab as plt
import pickle
from sklearn.metrics import matthews_corrcoef

def ConfusionMatrix(y,p):
    """Calculate a confusion matrix"""
    cm = np.zeros((10,10), dtype="uint16")
    for i in range(len(y)):
        cm[y[i],p[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return acc, cm

#  Metrics
p0 = np.load("cifar10_cnn_crop_predictions.npy")
p0 = np.argmax(p0, axis=1)
p1 = np.load("cifar10_cnn_augmented_predictions.npy")
p1 = np.argmax(p1, axis=1)
p2 = np.load("cifar10_cnn_augmented_deep_predictions.npy")
p2 = np.argmax(p2, axis=1)
y  = np.load("../data/cifar10/cifar10_test_labels.npy")

acc0, cm0 = ConfusionMatrix(y,p0)
acc1, cm1 = ConfusionMatrix(y,p1)
acc2, cm2 = ConfusionMatrix(y,p2)
mcc0 = matthews_corrcoef(y,p0)
mcc1 = matthews_corrcoef(y,p1)
mcc2 = matthews_corrcoef(y,p2)

print("Original:\n%s"  % np.array2string(cm0, precision=0))
print("Accuracy: %0.5f, MCC: %0.5f\n" % (acc0, mcc0))
print("Augmented:\n%s"  % np.array2string(cm1, precision=0))
print("Accuracy: %0.5f, MCC: %0.5f\n" % (acc1, mcc1))
print("Augmented+Deep:\n%s"  % np.array2string(cm2, precision=0))
print("Accuracy: %0.5f, MCC: %0.5f\n" % (acc2, mcc2))

#  training plot
h0 = pickle.load(open("cifar10_cnn_crop_history.pkl","rb"))
h1 = pickle.load(open("cifar10_cnn_augmented_history.pkl","rb"))
h2 = pickle.load(open("cifar10_cnn_augmented_deep_history.pkl","rb"))

v0 = 100.0*(1.0 - np.array(h0.history['val_accuracy']))
v1 = 100.0*(1.0 - np.array(h1.history['val_accuracy']))
v2 = 100.0*(1.0 - np.array(h2.history['val_accuracy']))

x = range(1,13)
xs = ['%d' % i for i in x]

plt.plot(x,v0[::10], marker='s', fillstyle='none', linewidth=0.7, color='k', label='original')
plt.plot(x,v1, marker='d', fillstyle='none', linewidth=0.7, color='k', label='augmented')
plt.plot(x,v2, marker='^', fillstyle='none', linewidth=0.7, color='k', label='deep')
plt.legend(loc='upper right')
plt.xticks(x,xs)
plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig('cifar10_cnn_aug_results_plot.png', dpi=300)
plt.savefig('cifar10_cnn_aug_results_plot.eps', dpi=300)
plt.close()

