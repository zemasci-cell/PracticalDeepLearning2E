#
#  file:  ensemble.py
#
#  Merge the per model predictions using a single run
#  of each model type
#
#  RTK, 27-Aug-2023
#  Last update:  28-Aug-2023
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

ytest = np.load("../data/cifar10/cifar10_test_labels.npy").squeeze()

vprob = np.load("runs/vgg_b128_e10_batchnorm_run2/predictions.npy")
rprob = np.load("runs/resnet_b128_e10_bn_run2/predictions.npy")
mprob = np.load("runs/mobilenet_b128_e10_run2/predictions.npy")
aprob = (vprob + rprob + mprob) / 3.0
aprob0 = (vprob + rprob) / 2.0

#  geometric mean
gprob = (vprob*rprob*mprob)**(1/3.0)
gprob0= np.sqrt(vprob*rprob)
gpred = np.argmax(gprob, axis=1)
gpred0= np.argmax(gprob0, axis=1)
gcm, gac = ConfusionMatrix(gpred, ytest)
gcm0, gac0 = ConfusionMatrix(gpred0, ytest)
gmcc = matthews_corrcoef(ytest, gpred)
gmcc0 = matthews_corrcoef(ytest, gpred0)

vpred = np.argmax(vprob, axis=1)
rpred = np.argmax(rprob, axis=1)
mpred = np.argmax(mprob, axis=1)
apred = np.argmax(aprob, axis=1)
apred0 = np.argmax(aprob0, axis=1)

acm, aac = ConfusionMatrix(apred, ytest)
acm0, aac0 = ConfusionMatrix(apred0, ytest)
amcc = matthews_corrcoef(ytest, apred)
amcc0 = matthews_corrcoef(ytest, apred0)

#  weighted by model ACC
_, vacc = ConfusionMatrix(vpred, ytest)
_, racc = ConfusionMatrix(rpred, ytest)
_, macc = ConfusionMatrix(mpred, ytest)

waprob = (vacc * vprob + racc * rprob + macc * mprob) / (vacc + racc + macc)
wapred = np.argmax(waprob, axis=1)
waprob0 = (vacc * vprob + racc * rprob) / (vacc + racc)
wapred0 = np.argmax(waprob0, axis=1)

awmcc = matthews_corrcoef(ytest, wapred)
awcm, awac = ConfusionMatrix(wapred, ytest)
awmcc0 = matthews_corrcoef(ytest, wapred0)
awcm0, awac0 = ConfusionMatrix(wapred0, ytest)

#  weighted by model MCC
vmcc = matthews_corrcoef(ytest, vpred) 
rmcc = matthews_corrcoef(ytest, rpred) 
mmcc = matthews_corrcoef(ytest, mpred) 

wprob = (vmcc * vprob + rmcc * rprob + mmcc * mprob) / (vmcc + rmcc + mmcc)
wpred = np.argmax(wprob, axis=1)
wprob0 = (vmcc * vprob + rmcc * rprob) / (vmcc + rmcc)
wpred0 = np.argmax(wprob0, axis=1)

wmcc = matthews_corrcoef(ytest, wpred)
wcm, wac = ConfusionMatrix(wpred, ytest)
wmcc0 = matthews_corrcoef(ytest, wpred0)
wcm0, wac0 = ConfusionMatrix(wpred0, ytest)

print()
print("Average (VGG8 + ResNet-18 + MobileNet):")
print(acm)
print()
print("MCC Weighted average:")
print(wcm)
print()
print("ACC Weighted average:")
print(awcm)
print()
print("ACC Geometric mean:")
print(gcm)
print()

print()
print("Average (VGG8 + ResNet-18):")
print(acm0)
print()
print("MCC Weighted average:")
print(wcm0)
print()
print("ACC Weighted average:")
print(awcm0)
print()
print("ACC Geometric mean:")
print(gcm0)
print()

print("VGG8 + ResNet-18:")
print("Ensemble (geometric)   : ACC = %0.5f, MCC = %0.5f" % (gac0, gmcc0))
print("Ensemble (MCC weighted): ACC = %0.5f, MCC = %0.5f" % (wac0, wmcc0))
print("Ensemble (ACC weighted): ACC = %0.5f, MCC = %0.5f" % (awac0, awmcc0))
print("Ensemble (average)     : ACC = %0.5f, MCC = %0.5f" % (aac0, amcc0))
print()
print("VGG8 + ResNet-18 + MobileNet:")
print("Ensemble (geometric)   : ACC = %0.5f, MCC = %0.5f" % (gac, gmcc))
print("Ensemble (MCC weighted): ACC = %0.5f, MCC = %0.5f" % (wac, wmcc))
print("Ensemble (ACC weighted): ACC = %0.5f, MCC = %0.5f" % (awac, awmcc))
print("Ensemble (average)     : ACC = %0.5f, MCC = %0.5f" % (aac, amcc))
print()
print("VGG8                   : ACC = %0.5f, MCC = %0.5f" % (vacc, vmcc))
print("ResNet-18              : ACC = %0.5f, MCC = %0.5f" % (racc, rmcc))
print("MobileNet              : ACC = %0.5f, MCC = %0.5f" % (macc, mmcc))
print()

