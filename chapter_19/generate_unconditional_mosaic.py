#
#  file:  generate_unconditional_mosaic.py
#
#  Use a trained generator model to generate a mosaic
#
#  RTK, 08-Feb-2024
#  Last update:  10-Feb-2024
#
################################################################

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
    print("generate_unconditional_mosaic <model> <latent> <out_image> [<seed>]")
    print()
    print("  <model>     - trained unconditional GAN generator")
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

#  16 samples per row
image = np.zeros((280,28*16), dtype="uint8")

for d in range(10):
    for i in range(16):
        noise = np.random.normal(size=(1,latent_dim))
        fake = model.predict(noise, verbose=0)
        img = (255.0*(fake[0,:]+1)/2).astype("uint8").reshape((28,28))
        image[(d*28):(d*28+28), (i*28):(i*28+28)] = img

Image.fromarray(255-image).save(oname)

