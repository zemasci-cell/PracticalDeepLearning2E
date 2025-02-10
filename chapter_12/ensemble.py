import sys
import numpy as np

def Accuracy(y,p):
    cm = np.zeros((10,10))
    for i in range(len(y)): cm[y[i],p[i]] += 1
    return 100.0*np.diag(cm).sum()/cm.sum()

p0 = np.load("prob_run0.npy"); p1 = np.load("prob_run1.npy")
p2 = np.load("prob_run2.npy"); p3 = np.load("prob_run3.npy")
p4 = np.load("prob_run4.npy"); p5 = np.load("prob_run5.npy")
y = np.load("../data/audio/ESC-10/esc10_spect_test_labels.npy")

#  Average
prob = (p0+p1+p2+p3+p4+p5)/6.0
p = np.argmax(prob, axis=1)
print("Accuracy (average) = %0.2f%%" % Accuracy(y,p))

#  Maximum
p = np.zeros(len(y), dtype="uint8")
for i in range(len(y)):
    t = np.array([p0[i],p1[i],p2[i],p3[i],p4[i],p5[i]])
    p[i] = np.argmax(t.reshape(60)) % 10
print("Accuracy (maximum) = %0.2f%%" % Accuracy(y,p))

#  Voting
t = np.zeros((6,len(y)), dtype="uint32")
t[0,:] = np.argmax(p0,axis=1); t[1,:] = np.argmax(p1,axis=1)
t[2,:] = np.argmax(p2,axis=1); t[3,:] = np.argmax(p3,axis=1)
t[4,:] = np.argmax(p4,axis=1); t[5,:] = np.argmax(p5,axis=1)
p = np.zeros(len(y), dtype="uint8")
for i in range(len(y)):
    q = np.bincount(t[:,i])
    p[i] = np.argmax(q)
print("Accuracy (voting)  = %0.2f%%" % Accuracy(y,p))

