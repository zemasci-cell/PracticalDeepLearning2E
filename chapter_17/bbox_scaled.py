#
#  file:  bbox_scaled.py
#
#  A simple dual-head model to localize an MNIST digit in a larger
#  image.  Run 'build_bbox_dataset.py' first, then run this
#  code followed by 'bbox_results.py' to calculate metrics and
#  generate result images.
#
#  Use the scaled datasets
#
#  RTK, 02-Jan-2024
#  Last update:  02-Jan-2024
#
################################################################

import os
import sys
import numpy as np
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import ReLU

def ConfusionMatrix(pred, y):
    """Return a confusion matrix"""
    cm = np.zeros((10,10), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc


if (len(sys.argv) == 1):
    print()
    print("bbox_scaled <batch> <epochs> blank|land <outdir>")
    print()
    print("  <batch>    - minibatch size (e.g. 32)")
    print("  <epochs>   - epochs (e.g. 10)")
    print("  blank|land - image source")
    print("  <outdir>   - output directory (overwritten)")
    print()
    exit(0)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
mode = sys.argv[3].lower()
outdir = sys.argv[4]

#  Load the datasets
if (mode == 'land'):
    xtrain = np.load("../data/mnist/mnist_bbox_scaled_land_xtrain.npy") / 255.0
    xtest = np.load("../data/mnist/mnist_bbox_scaled_land_xtest.npy") / 255.0
else:
    xtrain = np.load("../data/mnist/mnist_bbox_scaled_xtrain.npy") / 255.0
    xtest = np.load("../data/mnist/mnist_bbox_scaled_xtest.npy") / 255.0

ytrainc = np.load("../data/mnist/mnist_bbox_scaled_ytrainc.npy")
ytrainb = np.load("../data/mnist/mnist_bbox_scaled_ytrainb.npy")
ytestc = np.load("../data/mnist/mnist_bbox_scaled_ytestc.npy")
ytestb = np.load("../data/mnist/mnist_bbox_scaled_ytestb.npy")

#  Build the model
inp = Input(shape=(128,128,1))
_ = Conv2D( 32,(3,3))(inp); _ = ReLU()(_); _ = MaxPooling2D((2,2))(_)
_ = Conv2D( 64,(3,3))(_);   _ = ReLU()(_); _ = MaxPooling2D((2,2))(_)
_ = Conv2D(128,(3,3))(_);   _ = ReLU()(_); _ = MaxPooling2D((2,2))(_)
_ = Flatten()(_)
_ = Dense(128)(_)
_ = ReLU()(_)
_ = Dropout(0.5)(_)

#  Define the classification and bounding box heads
cout = Dense(10, activation='softmax', name='cout')(_)
bout = Dense(4, activation='sigmoid', name='bout')(_) 

#  Create the model and train
model = Model(inputs=inp, outputs=[cout, bout])
model.compile(optimizer='Adam', 
              loss={'cout': 'categorical_crossentropy', 'bout': 'mean_squared_error'},
              loss_weights={'cout': 1.0, 'bout': 50.0},
              metrics={'cout': 'accuracy'})
model.summary()

model.fit(xtrain, {'cout': ytrainc, 'bout': ytrainb},
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(xtest[:1000], {'cout': ytestc[:1000], 'bout': ytestb[:1000]}))

#  Results
os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
pred = model.predict(xtest, verbose=0)
plabel = np.argmax(pred[0], axis=1)
ytest = np.argmax(ytestc, axis=1)
cm, acc = ConfusionMatrix(plabel, ytest)
mcc = matthews_corrcoef(ytest, plabel)
s = 'Test set accuracy: %0.4f, MCC: %0.4f' % (acc,mcc)
with open(outdir+"/accuracy_mcc.txt", "w") as f:
    f.write(s+"\n")
np.save(outdir+"/confusion_matrix.npy", cm)
np.save(outdir+"/softmax.npy", pred[0])
np.save(outdir+"/bounding_box.npy", pred[1])
model.save(outdir+"/model.keras")
print(s)
print(cm)

