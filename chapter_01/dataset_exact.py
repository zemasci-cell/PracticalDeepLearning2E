import numpy as np
from sklearn.datasets import make_classification

# a will be a 10000(samples)x20(features) 2D array
# b will be the classification of each sample
a,b = make_classification(n_samples=10000, weights=[0.9,0.1])

# idx are the indexes of samples with a classification 0
idx = np.where(b == 0)[0]

# Store all samples in x0 and classification in y0
x0 = a[idx,:]
y0 = b[idx]

# Same thing for classification 1
idx = np.where(b == 1)[0]
x1 = a[idx,:]
y1 = b[idx]

# Randomize sample ordering of classification 0
idx = np.argsort(np.random.random(y0.shape))
# Classification update
y0 = y0[idx]
# Sample update
x0 = x0[idx]

# Randomize sample ordering of classification 1
idx = np.argsort(np.random.random(y1.shape))
# Classification update
y1 = y1[idx]
# Sample update
x1 = x1[idx]

# Calculate number of training samples of classification 0
ntrn0 = int(0.9 * x0.shape[0])

# Calculate number of training samples of classification 1
ntrn1 = int(0.9 * x1.shape[0])

# Data structures to hold training data
xtrn = np.zeros((int(ntrn0+ntrn1),20))
ytrn = np.zeros(int(ntrn0+ntrn1))

# Populate structures with the training data
xtrn[:ntrn0] = x0[:ntrn0]
xtrn[ntrn0:] = x1[:ntrn1]
ytrn[:ntrn0] = y0[:ntrn0]
ytrn[ntrn0:] = y1[:ntrn1]

# Calculate number of validation and test samples of classification 0
n0 = int(x0.shape[0]-ntrn0)

# Calculate number of validation and test samples of classification 1
n1 = int(x1.shape[0]-ntrn1)

# Using half the size calculated, create data structures to hold validation data
xval = np.zeros((int(n0/2+n1/2),20))
yval = np.zeros(int(n0/2+n1/2))

# Populate structures with the validation data
xval[:(n0//2)] = x0[ntrn0:(ntrn0+n0//2)]
xval[(n0//2):] = x1[ntrn1:(ntrn1+n1//2)]
yval[:(n0//2)] = y0[ntrn0:(ntrn0+n0//2)]
yval[(n0//2):] = y1[ntrn1:(ntrn1+n1//2)]

# Create test data
xtst = np.concatenate((x0[(ntrn0+n0//2):],x1[(ntrn1+n1//2):]))
ytst = np.concatenate((y0[(ntrn0+n0//2):],y1[(ntrn1+n1//2):]))