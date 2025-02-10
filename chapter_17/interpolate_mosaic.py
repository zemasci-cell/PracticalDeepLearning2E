#
#  file:  interpolate_mosaic.py
#
#  Use a trained generator model to interpolate images
#
#  RTK, 10-Apr-2023
#  Last update:  03-Feb-2024
#
################################################################

#  seeds 42, 359, 6502, 271828

import sys
import os
import random
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import random
from PIL import Image


if (len(sys.argv) == 1):
    print()
    print("interpolate_mosaic <model> <latent> <out_image> [<seed>]")
    print()
    print("  <model>     - trained conditional GAN generator")
    print("  <latent>    - latent vector dimensionality (e.g. 10)")
    print("  <out_image> - output image name")
    print("  <seed>      - PRNG seed (optional)")
    print()
    exit(0)

mname = sys.argv[1]
latent_dim = int(sys.argv[2])
oname = sys.argv[3]
if (len(sys.argv) == 5):
    #  seed all the PRNGs
    seed = int(sys.argv[4])
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

model = load_model(mname)

lo,hi = -5,5

#  iterate over the first 10 dimensions of the latent vector
image = np.zeros((280,28*16), dtype="uint8")

for d in range(10):
    label = np.zeros(10, dtype="float32")
    label[d] = 1
    label = np.array(label).reshape((1,10))
    base = np.random.normal(size=(1,latent_dim))

    for i,x in enumerate(np.linspace(lo,hi,16)):
        noise = base.copy()
        noise[0,d % latent_dim] = x
        fake = model.predict([noise, label], verbose=0)
        img = (255.0*fake[0,:,:,0]).astype("uint8").reshape((28,28))
        image[(d*28):(d*28+28), (i*28):(i*28+28)] = img

Image.fromarray(255-image).save(oname)

