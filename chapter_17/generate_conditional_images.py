#
#  file:  generate_images.py
#
#  Use the trained generator model to create images
#
#  RTK, 10-Apr-2023
#  Last update:  03-Feb-2024
#
################################################################

import sys
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

if (len(sys.argv) == 1):
    print()
    print("generate_images <model> <label> <dim> <number> <outdir> [<seed>]")
    print()
    print("  <model>   -  a trained conditional GAN generator")
    print("  <label>   -  desiged class label (or 'random')")
    print("  <dim>     -  latent vector dimensionality")
    print("  <number>  -  number of images")
    print("  <outdir>  -  output directory, NumPy if '.npy'")
    print("  <seed>    -  PRNG seed (optional)")
    print()
    exit(0)

mname = sys.argv[1]
clabel = sys.argv[2]
latent_dim = int(sys.argv[3])
nimages = int(sys.argv[4])
outdir = sys.argv[5]
if (len(sys.argv) == 7):
    #  seed all the PRNGs
    seed = int(sys.argv[6])
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

model = load_model(mname)

if (outdir.find(".npy") == -1):
    os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
    for i in range(nimages):
        label = np.zeros(10, dtype="float32")
        if (clabel != "random"):
            label[int(clabel)] = 1
        else:
            label[np.random.randint(0,10)] = 1
        label = np.array(label).reshape((1,10))
        noise = np.random.normal(size=(1,latent_dim))
        fake = model.predict([noise, label], verbose=0)
        img = (255.0*fake[0,:,:,0]).astype("uint8").reshape((28,28))
        Image.fromarray(img).save("%s/image_%03d.png" % (outdir,i))
else:
    images = []
    for i in range(nimages):
        label = np.zeros(10, dtype="float32")
        if (clabel != "random"):
            label[int(clabel)] = 1
        else:
            label[np.random.randint(0,10)] = 1
        label = np.array(label).reshape((1,10))
        noise = np.random.normal(size=(1,latent_dim))
        fake = model.predict([noise, label], verbose=0)
        img = (255.0*fake[0,:,:,0]).astype("uint8").reshape((28,28))
        images.append(img)
    images = np.array(images)
    np.save(outdir, images)

