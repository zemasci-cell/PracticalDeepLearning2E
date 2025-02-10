import numpy as np
import matplotlib.pylab as plt
from scipy.stats import ttest_ind

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def main():
    c0 = np.load("mnist_nn_experiments_scrambled_run0/val_error.npy")
    c1 = np.load("mnist_nn_experiments_scrambled_run1/val_error.npy")
    c2 = np.load("mnist_nn_experiments_scrambled_run2/val_error.npy")
    c3 = np.load("mnist_nn_experiments_scrambled_run3/val_error.npy")
    c4 = np.load("mnist_nn_experiments_scrambled_run4/val_error.npy")
    c5 = np.load("mnist_nn_experiments_scrambled_run5/val_error.npy")
    c6 = np.load("mnist_nn_experiments_scrambled_run6/val_error.npy")
    c7 = np.load("mnist_nn_experiments_scrambled_run7/val_error.npy")
    c8 = np.load("mnist_nn_experiments_scrambled_run8/val_error.npy")
    c9 = np.load("mnist_nn_experiments_scrambled_run9/val_error.npy")
    c = np.array([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9])

    d0 = np.load("mnist_nn_experiments_scrambled_run0/val_error_scrambled.npy")
    d1 = np.load("mnist_nn_experiments_scrambled_run1/val_error_scrambled.npy")
    d2 = np.load("mnist_nn_experiments_scrambled_run2/val_error_scrambled.npy")
    d3 = np.load("mnist_nn_experiments_scrambled_run3/val_error_scrambled.npy")
    d4 = np.load("mnist_nn_experiments_scrambled_run4/val_error_scrambled.npy")
    d5 = np.load("mnist_nn_experiments_scrambled_run5/val_error_scrambled.npy")
    d6 = np.load("mnist_nn_experiments_scrambled_run6/val_error_scrambled.npy")
    d7 = np.load("mnist_nn_experiments_scrambled_run7/val_error_scrambled.npy")
    d8 = np.load("mnist_nn_experiments_scrambled_run8/val_error_scrambled.npy")
    d9 = np.load("mnist_nn_experiments_scrambled_run9/val_error_scrambled.npy")
    d = np.array([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])
    
    print ttest_ind(c[:,-1],d[:,-1])

    plt.plot(smooth(c.mean(axis=0),53,"flat"), color="k", linewidth=1, linestyle="-")
    plt.plot(smooth(d.mean(axis=0),53,"flat"), color="r", linewidth=1, linestyle="-")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Test Error", fontsize=16)
    plt.ylim((0.04,0.055))
    plt.xlim((75,4000))
    plt.tight_layout()
    plt.savefig("mnist_nn_experiments_scrambled_plot.png", type="png", dpi=600)
    plt.savefig("mnist_nn_experiments_scrambled_plot.pdf", type="pdf", dpi=600)
    plt.show()


main()

