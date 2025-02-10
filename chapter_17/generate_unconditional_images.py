#
#  file:  generate_unconditional_images.py
#
#  Use the trained generator model to create images
#
#  RTK, 10-Apr-2023
#  Last update:  10-Feb-2024
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
    print("generate_unconditional_images <model> <dim> <number> <outdir> [<seed>]")
    print()
    print("  <model>   -  a trained unconditional GAN generator")
    print("  <dim>     -  latent vector dimensionality")
    print("  <number>  -  number of images")
    print("  <outdir>  -  output directory, NumPy if '.npy'")
    print("  <seed>    -  PRNG seed (optional)")
    print()
    exit(0)

mname = sys.argv[1]
latent_dim = int(sys.argv[2])
nimages = int(sys.argv[3])
outdir = sys.argv[4]
if (len(sys.argv) == 6):
    #  seed all the PRNGs
    seed = int(sys.argv[5])
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

model = load_model(mname)

if (outdir.find(".npy") == -1):
    os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
    for i in range(nimages):
        noise = np.random.normal(size=(1,latent_dim))
        fake = model.predict(noise, verbose=0)
        img = (255.0*fake[0,:]).astype("uint8").reshape((28,28))
        Image.fromarray(img).save("%s/image_%03d.png" % (outdir,i))
else:
    images = []
    for i in range(nimages):
        noise = np.random.normal(size=(1,latent_dim))
        fake = model.predict(noise, verbose=0)
        img = (255.0*fake[0,:]).astype("uint8").reshape((28,28))
        images.append(img)
    images = np.array(images)
    np.save(outdir, images)

