#
#  file:  mnist_cnn_gan.py
#
#  A CNN-based GAN for MNIST
#
################################################################

import sys
import os
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ReLU
from tensorflow.keras.layers import Dense, Reshape, Dropout
from tensorflow.keras.layers import LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import backend as K
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
    print("mnist_cnn_gan.py <latent> <batch_size> <minibatches> <outdir>")
    print()
    print("  <latent>      - latent vector dimensionality (e.g. 60)")
    print("  <batch_size>  - minibatch size (e.g. 128)")
    print("  <minibatches> - number of minibatches (e.g. 10000)")
    print("  <outdir>      - output directory (overwritten)")
    print()
    exit(0)

LATENT = int(sys.argv[1])
batch_size = int(sys.argv[2])
minibatches = int(sys.argv[3])
outdir = sys.argv[4]

#  Load the all MNIST training data, scale [0,1]
xtrn = np.load("../data/mnist/mnist_train_images.npy")
xtrn = np.vstack((xtrn, np.load("../data/mnist/mnist_test_images.npy")))
xtrn = (xtrn - 127.5) / 127.5

#  Build the generator
inp = Input((LATENT,))
_ = Dense(7*7*LATENT)(inp)
_ = LeakyReLU(0.2)(_)
_ = BatchNormalization(momentum=0.8)(_)
_ = Reshape((7,7,LATENT))(_)
_ = Conv2DTranspose(64, (3,3), strides=(2,2), padding="same")(_)
_ = BatchNormalization(momentum=0.8)(_)
_ = LeakyReLU(0.2)(_)
_ = Conv2DTranspose(64, (3,3), strides=(1,1), padding="same")(_)
_ = BatchNormalization(momentum=0.8)(_)
_ = LeakyReLU(0.2)(_)
outp = Conv2DTranspose(1, (5,5), strides=(2,2), padding="same", activation="tanh")(_)
generator = Model(inputs=inp, outputs=outp)

#  And the discriminator
image_inp = Input((28,28,1))
_ = Conv2D(32, (3,3), strides=(2,2), padding="same")(image_inp)
_ = LeakyReLU(0.2)(_)
_ = Conv2D(64, (3,3), strides=(2,2), padding="same")(_)
_ = LeakyReLU(0.2)(_)
_ = Flatten()(_)
_ = Dense(512)(_)
_ = LeakyReLU(0.2)(_)
outp = Dense(1, activation='sigmoid')(_)
discriminator = Model(inputs=image_inp, outputs=outp)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

#  Freeze the discriminator prior to compiling the
#  combined GAN model (discriminator.train_on_batch still works properly)
discriminator.trainable = False

#  Combined model
inp = Input((LATENT,))
fake_image = generator(inp)
outp = discriminator(fake_image)
gan = Model(inputs=inp, outputs=outp)
gan.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

os.system("mkdir %s 2>/dev/null" % outdir)
os.system("mkdir %s/images 2>/dev/null" % outdir)

#  Training
for k in range(1, minibatches+1):
    #  Select a random half batch of real images
    idx = np.random.randint(0, xtrn.shape[0], batch_size//2)
    imgs = xtrn[idx]

    #  Sample noise and use the generator
    noise = np.random.normal(size=(batch_size//2, LATENT))
    gen_imgs = generator.predict(noise, verbose=0)

    #  Train the discriminator using label smoothing
    ones = 0.8 + 0.4*np.random.random((batch_size//2, 1))
    zeros= 0.4*np.random.random((batch_size//2, 1))
    d_loss_real = discriminator.train_on_batch(imgs, ones)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, zeros)
    D = 0.5 * (d_loss_real + d_loss_fake)

    #  Now train the generator
    noise = np.random.normal(size=(batch_size, LATENT))
    ones = 0.8 + 0.4*np.random.random((batch_size, 1))
    G = gan.train_on_batch(noise, ones)

    print("Minibatch %5d: G=%0.8f, D=%0.8f" % (k, G, D))
    if (k==1) or ((k % 200) == 0):
        GenerateSamples(k, generator, outdir)

#  Store the generator
generator.save(outdir+"/generator.keras")

