#
#  file:  position.py
#
#  Generate the position encoding matrix for the example LLM
#
#  RTK, 17-Mar-2024
#  Last update:  17-Mar-2024
#
################################################################

import numpy as np
import matplotlib.pylab as plt
from PIL import Image

#  For the LLM example
E,N = 768, 1024

#  Build the matrix element by element
P = np.zeros((N,E))

for p in range(N):
    for i in range(E//2):
        P[p,2*i] = np.sin(p/(10000**((2*i)/E)))
        P[p,2*i+1] = np.cos(p/(10000**((2*i)/E)))

#  Rescale and save as an image
im = (255*(P+1)/2).astype("uint8")
Image.fromarray(im).save("positional_encoding.png")

