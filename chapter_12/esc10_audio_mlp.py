#
#  file:  esc10_audio_mlp.py
#
#  Traditional MLP
#
#  RTK, 13-Nov-2019
#  Last update:  20-May-2023
#
################################################################

from sklearn.neural_network import MLPClassifier
import numpy as np

num_classes = 10

x_train = np.load("../data/audio/ESC-10/esc10_raw_train_audio.npy")[:,:,0]
y_train = np.load("../data/audio/ESC-10/esc10_raw_train_labels.npy")
x_test  = np.load("../data/audio/ESC-10/esc10_raw_test_audio.npy")[:,:,0]
y_test  = np.load("../data/audio/ESC-10/esc10_raw_test_labels.npy")

x_train = (x_train.astype('float32') + 32768) / 65536
x_test = (x_test.astype('float32') + 32768) / 65536

model = MLPClassifier(hidden_layer_sizes=(512,128), 
                      max_iter=200,
                      solver='lbfgs')
model.fit(x_train, y_train)

score = 100.0*model.score(x_test, y_test)
print("score = %0.2f%%" % score)

