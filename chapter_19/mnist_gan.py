#
#  file:  mnist_gan.py
#
#  An MLP-based GAN for MNIST
#
################################################################

import sys
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pylab as plt

def GenerateSamples(mb, generator, outdir):
    """Dump a 10x10 grid of sampled images"""
    N = 10
    noise = np.random.normal(size=(N*N,LATENT))
    imgs = generator.predict(noise, verbose=0).reshape((N*N,28,28))
    plt.figure(figsize=(N,N))
    for i in range(N*N):
        plt.subplot(N,N,i+1)
        plt.imshow(imgs[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
    plt.savefig(outdir+("/images/generated_mb_%d.png" % mb), dpi=300)
    plt.close()


if (len(sys.argv) == 1):
    print()
    print("mnist_gan.py <latent> <batch_size> <minibatches> <outdir>")
    print()
    print("  <latent>      - latent vector dimensionality (e.g. 40)")
    print("  <batch_size>  - minibatch size (e.g. 128)")
    print("  <minibatches> - minibatches (e.g. 10000)")
    print("  <outdir>      - output directory (overwritten)")
    print()
    exit(0)

LATENT = int(sys.argv[1])
batch_size = int(sys.argv[2])
minibatches = int(sys.argv[3])
outdir = sys.argv[4]

#  Use Adam - learning rate sensitive
adam = Adam(learning_rate=0.0002, beta_1=0.5)

#  Load the training data.  Labels are unnecessary.  Scale [-1,1]
xtrn = np.load("../data/mnist/mnist_train_images.npy").reshape((60000,28*28))
xtrn = (xtrn - 127.5) / 127.5

#  Build the generator
inp = Input((LATENT,))
_ = Dense(256)(inp)
_ = LeakyReLU(0.2)(_)
_ = Dense(512)(_)
_ = LeakyReLU(0.2)(_)
_ = Dense(1024)(_)
_ = LeakyReLU(0.2)(_)
outp = Dense(784, activation='tanh')(_)
generator = Model(inputs=inp, outputs=outp)

#  And the discriminator
inp = Input((784,))
_ = Dense(1024)(inp)
_ = LeakyReLU(0.2)(_)
_ = Dropout(0.3)(_)
_ = Dense(512)(_)
_ = LeakyReLU(0.2)(_)
_ = Dropout(0.3)(_)
_ = Dense(256)(_)
_ = LeakyReLU(0.2)(_)
_ = Dropout(0.3)(_)
outp = Dense(1, activation='sigmoid')(_)
discriminator = Model(inputs=inp, outputs=outp)
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

#  Freeze the discriminator
discriminator.trainable = False

#  Put the pieces together
inp = Input((LATENT,))
_ = generator(inp)
outp = discriminator(_)
gan = Model(inputs=inp, outputs=outp)
gan.compile(loss='binary_crossentropy', optimizer=adam)

#  Set up the output directory
os.system("mkdir %s 2>/dev/null" % outdir)
os.system("mkdir %s/images 2>/dev/null" % outdir)

for k in range(1, minibatches+1):
    #  Select a random half batch of real images
    idx = np.random.randint(0, xtrn.shape[0], batch_size//2)
    imgs = xtrn[idx]

    #  Sample noise and use the generator
    noise = np.random.normal(size=(batch_size//2, LATENT))
    gen_imgs = generator.predict(noise, verbose=0)

    #  Train the discriminator
    ones = np.ones((batch_size//2, 1))
    zeros = np.zeros((batch_size//2, 1))
    d_loss_real = discriminator.train_on_batch(imgs, ones)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, zeros)
    D = 0.5 * (d_loss_real + d_loss_fake)

    #  Now train the generator
    noise = np.random.normal(size=(batch_size, LATENT))
    ones = np.ones((batch_size, 1))
    G = gan.train_on_batch(noise, ones)

    print("Minibatch %5d: G=%0.8f, D=%0.8f" % (k, G, D), flush=True)
    if (k==1) or ((k % 200) == 0):
        GenerateSamples(k, generator, outdir)

#  Store the trained generator
generator.save(outdir+"/generator.keras")

