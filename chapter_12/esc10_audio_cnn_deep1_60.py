#
#  file:  esc10_audio_cnn_deeper.py
#
#  1D convolutional network, deeper depth
#
#  RTK, 13-Nov-2019
#  Last update:  21-May-2023
#
################################################################

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import backend as K
import matplotlib.pylab as plt
import numpy as np

batch_size = 32
num_classes = 10
epochs = 60  # more epochs
nsamp = (882,1)
z = 7 # fix to best kernel width

x_train = np.load("../data/audio/ESC-10/esc10_raw_train_audio.npy")
y_train = np.load("../data/audio/ESC-10/esc10_raw_train_labels.npy")
x_test  = np.load("../data/audio/ESC-10/esc10_raw_test_audio.npy")
y_test  = np.load("../data/audio/ESC-10/esc10_raw_test_labels.npy")

x_train = (x_train.astype('float32') + 32768) / 65536
x_test = (x_test.astype('float32') + 32768) / 65536

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv1D(32, kernel_size=z,
                 activation='relu',
                 input_shape=nsamp))

model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.25))

model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.25))

model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(x_test, y_test))

#  make the plots
tloss = history.history['loss']
vloss = history.history['val_loss']
terr = 1.0 - np.array(history.history['accuracy'])
verr = 1.0 - np.array(history.history['val_accuracy'])
x = range(1,epochs+1)

plt.plot(x, tloss, linestyle='solid', linewidth=0.7, color='k')
plt.plot(x[::5], tloss[::5], linestyle='none', marker='o', fillstyle='none', color='k', label='training loss')
plt.plot(x, vloss, linestyle='solid', linewidth=0.7, color='k')
plt.plot(x[::5], vloss[::5], linestyle='none', marker='^', fillstyle='none', color='k', label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower left')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig('deeper_epochs_60_loss.png', dpi=300)
plt.savefig('deeper_epochs_60_loss.eps', dpi=300)
plt.close()

plt.plot(x, terr, linestyle='solid', linewidth=0.7, color='k')
plt.plot(x[::5], terr[::5], linestyle='none', marker='o', fillstyle='none', color='k', label='training error')
plt.plot(x, verr, linestyle='solid', linewidth=0.7, color='k')
plt.plot(x[::5], verr[::5], linestyle='none', marker='^', fillstyle='none', color='k', label='validation error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig('deeper_epochs_60_error.png', dpi=300)
plt.savefig('deeper_epochs_60_error.eps', dpi=300)
plt.close()

