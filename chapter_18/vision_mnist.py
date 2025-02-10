#
#  file:  vision_mnist.py
#
#  Classify MNIST digits with LLaVA
#
#  run first:
#    > pip3 install ollama
#
#  RTK, 02-Mar-2024
#  Last update:  02-Mar-2024
#
################################################################

import sys
import ollama
import io
import numpy as np
from PIL import Image

if (len(sys.argv) == 1):
    print()
    print("vision_mnist <model>")
    print()
    print("  <model> - vision model (e.g. llava:7b)")
    print()
    exit(0)

mname = sys.argv[1]

#  Prompt
prompt = """
Identify the digit present in this image. The image shows only a single, handwritten digit, 0 through 9, and nothing else.  Reply with only the digit as a number.
"""

#  Gather a small datset of MNIST digits -- two of each
x = np.load("../data/mnist/mnist_test_images.npy")
y = np.load("../data/mnist/mnist_test_labels.npy")
np.random.seed(6502)
idx = np.argsort(np.random.random(len(y)))
x = x[idx]
y = y[idx]

xtst, ytst = [], []
for i in range(10):
    idx = np.where(y==i)[0][:2]
    xtst.append(x[idx[0]])
    ytst.append(i)
    xtst.append(x[idx[1]])
    ytst.append(i)
xtst = np.array(xtst)
ytst = np.array(ytst)

#  Process each digit image
print("Classifying %d randomly selected MNIST digits:" % len(ytst))

for i,x in enumerate(xtst):
    #  Get the bytes of the image in PNG format
    image = Image.fromarray(x).resize((224,224), resample=Image.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    png = buf.getvalue()

    #  Ask the model to identify the digit
    resp = ollama.chat(
        model=mname,
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [png],
            },
        ]
    )
    y = resp['message']['content']
    print("    %2d:  actual: %d  model: %s" % (i+1, ytst[i], y), flush=True)

print()

