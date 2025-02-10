import os
import random

random.seed(8675309)

for i in range(100):
    seed = random.randint(1,100000)
    cmd = "python3 interpolate_random_mosaic.py results/mnist_conditional_v2_latent_160_128_12000/generator.keras 160 images/seed_160_%05d.png %d"
    os.system(cmd % (seed,seed))
    cmd = "python3 interpolate_random_mosaic.py results/mnist_conditional_v2_latent_10_128_12000/generator.keras 10 images/seed_10_%05d.png %d"
    os.system(cmd % (seed,seed))

