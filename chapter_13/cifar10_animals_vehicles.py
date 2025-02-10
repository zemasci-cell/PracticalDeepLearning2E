#
#  file:  cifar10_animal_vehicles.py
#
#  MNIST architecture applied to CIFAR-10
#
#  RTK, 20-Oct-2019
#  Last update:  11-May-2023
#
################################################################

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pylab as plt

def tally_predictions(model, x, y_test):
    y = np.argmax(y_test, axis=1)
    pp = model.predict(x)
    p = np.argmax(pp, axis=1)
    tp = tn = fp = fn = 0
    for i in range(len(y)):
        if (p[i] == 0) and (y[i] == 0):
            tn += 1
        elif (p[i] == 0) and (y[i] == 1):
            fn += 1
        elif (p[i] == 1) and (y[i] == 0):
            fp += 1
        else:
            tp += 1
    score = float(tp+tn) / float(tp+tn+fp+fn)
    return [tp, tn, fp, fn, score]

def basic_metrics(tally):
    tp, tn, fp, fn, _ = tally
    return {
        "TPR": tp / (tp + fn),
        "TNR": tn / (tn + fp),
        "PPV": tp / (tp + fp),
        "NPV": tn / (tn + fn),
        "FPR": fp / (fp + tn),
        "FNR": fn / (fn + tp)
    }

from math import sqrt
def advanced_metrics(tally, m): 
    tp, tn, fp, fn, _ = tally
    n = tp+tn+fp+fn
    po = (tp+tn)/n
    pe = (tp+fn)*(tp+fp)/n**2 + (tn+fp)*(tn+fn)/n**2

    return {
        "F1": 2.0*m["PPV"]*m["TPR"] / (m["PPV"] + m["TPR"]),
        "MCC": (tp*tn - fp*fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        "kappa": (po - pe) / (1.0 - pe),
        "informedness": m["TPR"] + m["TNR"] - 1.0,
        "markedness": m["PPV"] + m["NPV"] - 1.0 
    }

from sklearn.metrics import roc_auc_score, roc_curve
def roc_curve_area(model, x, y_test):
    y = np.argmax(y_test, axis=1)
    pp = model.predict(x)
    p = np.argmax(pp, axis=1)
    auc = roc_auc_score(y,p)
    roc = roc_curve(y,pp[:,1])
    return [auc, roc]

def false_positives_and_negatives(model, x, y):
    y_label= np.load("../data/cifar10/cifar10_test_labels.npy")[1000:]
    y_test = np.argmax(y, axis=1)
    pp = model.predict(x)
    p = np.argmax(pp, axis=1)
    hp = []; hn = []
    for i in range(len(y_test)):
        if (p[i] == 0) and (y_test[i] == 1):
            hn.append(y_label[i])
        elif (p[i] == 1) and (y_test[i] == 0):
            hp.append(y_label[i])
    hp = np.array(hp)
    hn = np.array(hn)
    a = np.histogram(hp, bins=10, range=[0,9])[0]
    b = np.histogram(hn, bins=10, range=[0,9])[0]
    print("vehicles as animals: %s" % np.array2string(a))
    print("animals as vehicles: %s" % np.array2string(b))


batch_size = 128
num_classes = 2
epochs = 12
img_rows, img_cols = 32, 32

x_train = np.load("../data/cifar10/cifar10_train_images.npy")
y_train = np.load("../data/cifar10/cifar10_train_animal_vehicle_labels.npy")

x_test = np.load("../data/cifar10/cifar10_test_images.npy")
y_test = np.load("../data/cifar10/cifar10_test_animal_vehicle_labels.npy")

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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

model.summary()

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test[:1000],y_test[:1000]))

#  evaluate the held-out test set
model.save("cifar10_cnn_animal_vehicles_model.keras") 
x = x_test[1000:]
y = y_test[1000:]

tally = tally_predictions(model, x, y)
tp,tn,fp,fn,acc = tally
basic = basic_metrics(tally)
adv = advanced_metrics(tally, basic)
auc,roc = roc_curve_area(model, x, y)
print()
false_positives_and_negatives(model, x, y)

plt.plot(roc[0],roc[1], color='k')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("cifar10_animal_vehicle_roc.png", dpi=300)
plt.savefig("cifar10_animal_vehicle_roc.eps", dpi=300)
plt.close()

print(f"""
TP {tp}
FP {fp}
TN {tn}
FN {fn}
TPR (sensitivity, recall) {basic["TPR"]}
TNR (specificity)         {basic["TNR"]}
PPV (precision)           {basic["PPV"]}
NPV                       {basic["NPV"]}
FPR                       {basic["FPR"]}
FNR                       {basic["FNR"]}
F1                        {adv['F1']}
MCC                       {adv['MCC']}
Cohen's kappa             {adv['kappa']}
Informedness              {adv['informedness']}
Markedness                {adv['markedness']}
Accuracy                  {acc}
AUC                       {auc}
""")

