#
#  file:  mnist_conditional_gan.py
#
#  A CNN-based conditional GAN for MNIST
#
################################################################

import sys
import os
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Dense, Reshape, Multiply
from tensorflow.keras.layers import LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import backend as K
import matplotlib.pylab as plt

def GenerateSamples(mb, generator, outdir):
    """Dump a 10x10 grid of sampled images"""
    N = 10
    plt.figure(figsize=(N,N))
    i = 1
    for l in range(N):
        for k in range(N):
            noise = np.random.normal(size=(1,LATENT))
            label = [0]*N; label[l] = 1
            label = np.array(label).reshape((1,N))
            img = generator.predict([noise, label], verbose=0)[0,:,:,0]
            plt.subplot(N,N,i); i += 1
            plt.imshow(img, interpolation='nearest', cmap='gray')
            plt.axis('off')
    plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
    plt.savefig(outdir+("/images/generated_mb_%d.png" % mb), dpi=300)
    plt.close()


if (len(sys.argv) == 1):
    print()
    print("mnist_conditional_gan.py <latent> <batch_size> <minibatches> <outdir>")
    print()
    print("  <latent>      - latent vector dimensionality (e.g. 100)")
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
num_classes = 10
xtrn = np.load("../data/mnist/mnist_train_images.npy")
xtrn = np.vstack((xtrn, np.load("../data/mnist/mnist_test_images.npy")))
xtrn = xtrn / 255
ytrn = np.load("../data/mnist/mnist_train_labels.npy")
ytrn = np.hstack((ytrn, np.load("../data/mnist/mnist_test_labels.npy")))
ytrn = keras.utils.to_categorical(ytrn, num_classes)

#  Build the generator
latent_inp = Input((LATENT,))
label_inp = Input((num_classes,))
merged_inp = Concatenate()([latent_inp, label_inp])
gen_channels = LATENT + num_classes

_ = Dense(7*7*gen_channels)(merged_inp)
_ = LeakyReLU(0.2)(_)
_ = Reshape((7,7,gen_channels))(_)
_ = Conv2DTranspose(64, (3,3), strides=(2,2), padding="same")(_)
_ = LeakyReLU(0.2)(_)
_ = Conv2DTranspose(64, (3,3), strides=(2,2), padding="same")(_)
_ = LeakyReLU(0.2)(_)
outp = Conv2D(1, (7,7), padding="same", activation="sigmoid")(_)
generator = Model(inputs=[latent_inp, label_inp], outputs=outp)

#  And the discriminator
image_shape = (28,28,1)
image_inp = Input((28,28,1))
label_embedding = Dense(np.prod(image_shape))(label_inp)
label_embedding = Reshape(image_shape)(label_embedding)
mod_image = Multiply()([image_inp, label_embedding])
_ = Conv2D(64, (3,3), strides=(2,2), padding="same")(mod_image)
_ = LeakyReLU(0.2)(_)
_ = Conv2D(128, (3,3), strides=(2,2), padding="same")(_)
_ = LeakyReLU(0.2)(_)
_ = GlobalMaxPooling2D()(_)
outp = Dense(1, activation="sigmoid")(_)
discriminator = Model([image_inp, label_inp], outp)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

#  Freeze the discriminator
discriminator.trainable = False

#  Combined model
gan_input = [Input((LATENT,)), Input((num_classes,))]
fake_image = generator(gan_input)
gan_output = discriminator([fake_image, gan_input[1]])
gan = Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

os.system("mkdir %s 2>/dev/null" % outdir)
os.system("mkdir %s/images 2>/dev/null" % outdir)

#  Training
for k in range(1, minibatches+1):
    #  Select a random half batch of real images
    idx = np.random.randint(0, xtrn.shape[0], batch_size//2)
    imgs, rlabels = xtrn[idx], ytrn[idx]

    #  Sample noise and use the generator
    noise = np.random.normal(size=(batch_size//2, LATENT))
    labels = np.random.randint(0, num_classes, batch_size//2)
    labels = keras.utils.to_categorical(labels, num_classes)
    gen_imgs = generator.predict([noise, labels], verbose=0)

    #  Train the discriminator
    ones = np.ones((batch_size//2, 1))
    zeros = np.zeros((batch_size//2, 1))
    d_loss_real = discriminator.train_on_batch([imgs, rlabels], ones)
    d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], zeros)
    D = 0.5 * (d_loss_real + d_loss_fake)

    #  Now train the generator conditioned on the labels
    noise = np.random.normal(size=(batch_size, LATENT))
    labels = np.random.randint(0, num_classes, batch_size)
    labels = keras.utils.to_categorical(labels, num_classes)
    ones = np.ones((batch_size, 1))
    G = gan.train_on_batch([noise, labels], ones)

    print("Minibatch %5d: G=%0.8f, D=%0.8f" % (k, G, D), flush=True)
    if (k==1) or ((k % 200) == 0):
        GenerateSamples(k, generator, outdir)

#  Store the generator
generator.save(outdir+"/generator.keras")

