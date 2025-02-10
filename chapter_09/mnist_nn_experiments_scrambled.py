#
#  file:  mnist_nn_experiments_scrambled.py
#
#  Reduced MNIST + NN for Chapter 6.
#
#  RTK, 22-Oct-2018
#  Last update:  22-Oct-2018
#
###############################################################

import os
import numpy as np
import time
from sklearn.neural_network import MLPClassifier 

def epoch(x_train, y_train, x_test, y_test, clf):
    """Results for a single epoch"""

    clf.fit(x_train, y_train)
    train_loss = clf.loss_
    train_err = 1.0 - clf.score(x_train, y_train)
    val_err = 1.0 - clf.score(x_test, y_test)
    clf.warm_start = True
    return [train_loss, train_err, val_err]


def run(x_train, y_train, x_test, y_test, clf, max_iter):
    """Train and test"""

    train_loss = []
    train_err = []
    val_err = []

    clf.max_iter = 1  # one epoch at a time
    for i in range(max_iter):
        tl, terr, verr = epoch(x_train, y_train, x_test, y_test, clf)
        train_loss.append(tl)
        train_err.append(terr)
        val_err.append(verr)
        print "    %4d: val_err = %0.5f" % (i, val_err[-1])

    wavg = 0.0
    n = 0
    for w in clf.coefs_:
        wavg += w.sum()
        n += w.size
    wavg /= n

    return [train_loss, train_err, val_err, wavg]


def main():
    """Plot the training and validation losses."""

    outdir = "mnist_nn_experiments_scrambled"
    os.system("rm -rf %s" % outdir)
    os.system("mkdir %s" % outdir)

    #  Vector MNIST versions scaled [0,1)
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    xstrain = np.load("../data/mnist/mnist_train_scrambled_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    xstest = np.load("../data/mnist/mnist_test_scrambled_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    #  Reduce the size of the train dataset
    x_train = x_train[:6000]
    y_train = y_train[:6000]
    xstrain = xstrain[:6000]
    epochs = 6000 

    print "Unscrambled"
    nn = MLPClassifier(solver="sgd", verbose=False, tol=0,
           nesterovs_momentum=False, early_stopping=False, learning_rate_init=0.01,
           momentum=0.9, hidden_layer_sizes=(100,50), activation="relu",
           alpha=0.2, learning_rate="constant", batch_size=64, max_iter=1)
    train_loss, train_err, val_err, wavg = run(x_train, y_train, x_test, y_test, nn, epochs)
    print "    final: train error: %0.5f, val error: %0.5f, mean weight value = %0.8f"  % \
        (train_err[-1], val_err[-1], wavg)
    print
    np.save(outdir + ("/train_error.npy"), train_err)
    np.save(outdir + ("/train_loss.npy"), train_loss)
    np.save(outdir + ("/val_error.npy"), val_err)
    np.save(outdir + ("/mean_weight.npy"), np.array(wavg))

    print "Scrambled"
    nn = MLPClassifier(solver="sgd", verbose=False, tol=0,
           nesterovs_momentum=False, early_stopping=False, learning_rate_init=0.01,
           momentum=0.9, hidden_layer_sizes=(100,50), activation="relu",
           alpha=0.2, learning_rate="constant", batch_size=64, max_iter=1)
    train_loss, train_err, val_err, wavg = run(xstrain, y_train, xstest, y_test, nn, epochs)
    print "    final: train error: %0.5f, val error: %0.5f, mean weight value = %0.8f"  % \
        (train_err[-1], val_err[-1], wavg)
    print
    np.save(outdir + ("/train_error_scrambled.npy"), train_err)
    np.save(outdir + ("/train_loss_scrambled.npy"), train_loss)
    np.save(outdir + ("/val_error_scrambled.npy"), val_err)
    np.save(outdir + ("/mean_weight_scrambled.npy"), np.array(wavg))


if (__name__ == "__main__"):
    main()

