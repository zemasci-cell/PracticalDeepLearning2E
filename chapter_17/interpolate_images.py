#
#  file:  interpolate_images.py
#
#  Use the trained generator model to interpolate images
#
#  RTK, 10-Apr-2023
#  Last update:  03-Feb-2024
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
    print("interpolate_images <model> <label> <dim> <outdir> [<seed>]")
    print()
    print("  <model>   -  trained conditional GAN generator")
    print("  <label>   -  class label")
    print("  <dim>     -  dimension to iterate over")
    print("  <outdir>  -  output directory (overwritten)")
    print("  <seed>    -  PRNG seed (optional)")
    print()
    exit(0)

mname = sys.argv[1]
clabel = int(sys.argv[2])
dim = int(sys.argv[3])
outdir = sys.argv[4]
if (len(sys.argv) == 6):
    #  seed all the PRNGs
    seed = int(sys.argv[5])
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

model = load_model(mname)

os.system("rm -rf %s; mkdir %s" % (outdir,outdir))

latent_dim = 10  # match what is the conditional GANs
label = np.zeros(10, dtype="float32")
label[clabel] = 1
label = np.array(label).reshape((1,10))

base = np.random.normal(size=(1,latent_dim))
lo,hi = -4,4

for i,x in enumerate(np.linspace(lo,hi,24)): 
    noise = base.copy()
    noise[0,dim] = x
    fake = model.predict([noise, label], verbose=0)
    img = (255.0*fake[0,:,:,0]).astype("uint8").reshape((28,28))
    Image.fromarray(img).save("%s/image_%03d.png" % (outdir,i))


