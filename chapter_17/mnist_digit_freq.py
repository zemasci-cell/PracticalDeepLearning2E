#
#  file:  mnist_digit_freq.py
#
#  Use a well-trained CNN to classify generated MNIST digits
#  to determine their frequency.
#
#  RTK, 30-Jan-2024
#  Last update:  10-Feb-2024
#
################################################################

import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np
import sys
import os

def euclid(a,b):
    return np.sqrt(((a-b)**2).sum())

if (len(sys.argv) == 1):
    print()
    print("mnist_digit_freq <generator> <latent>")
    print()
    print("  <generator> - trained MNIST generator to test")
    print("  <latent>    - latent vector dimensionality")
    print()
    exit(0)

#  Define and train an MNIST classifier
batch_size = 128 
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28

x_train = np.load("../data/mnist/mnist_train_images.npy")
y_train = np.load("../data/mnist/mnist_train_labels.npy")

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes)

if (not os.path.exists("mnist_cnn_digit_freq_model.keras")):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), 
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    model.save("mnist_cnn_digit_freq_model.keras")
else:
    model = load_model("mnist_cnn_digit_freq_model.keras")

#  Actual MNIST training set digit frequencies
mnist = np.array([
    0.09871667, 0.11236667, 0.0993, 0.10218333, 0.09736667,
    0.09035, 0.09863333, 0.10441667, 0.09751667, 0.09915
])

#  Load the generator model
generator = load_model(sys.argv[1])
LATENT = int(sys.argv[2])

#  Generate 10,000 digits
N = 10_000
noise = np.random.normal(size=(N,LATENT))
t = generator.predict(noise, verbose=0)
x_test = np.zeros((N,28,28), dtype="uint8")
for i in range(N):
    x_test[i,:,:] = (((t[i]+1)/2)*255).astype("uint8").reshape((28,28))
x_test = x_test / 255

#  Classify the generated digits taking the model
#  at its word (> 99 percent accuracy)
prob = model.predict(x_test, verbose=0)
pred = np.argmax(prob, axis=1)
counts = np.bincount(pred, minlength=10)
freq = counts / counts.sum()

print("Digit frequency:")
for i in range(len(freq)):
    print("    %d: %0.6f" % (i, freq[i]))
print("(Euclidean distance = %0.5f)" % euclid(freq, mnist))
print()

