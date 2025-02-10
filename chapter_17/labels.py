import numpy as np
from PIL import Image

c = [[255, 0, 0],     # red
     [0, 255, 0],     # green
     [0, 0, 255],     # blue
     [255, 255, 0],   # yellow
     [0, 255, 255],   # cyan
     [255, 0, 255],   # magenta
     [255, 165, 0],   # orange
     [128, 0, 128],   # purple
     [50, 205, 50],   # lime green
     [255, 192, 203]] # pink

x = np.load("../data/mnist/mnist_test_images.npy")
y = np.load("../data/mnist/mnist_test_labels.npy")

img = np.zeros((28,280,3), dtype="uint8")

np.random.seed(42)
for i in range(10):
    idx = np.where(y==i)[0]
    idx = idx[np.random.randint(0,len(idx))]
    a,b = np.where(x[idx]!=0)
    a = np.array(a)
    b = 28*i + np.array(b)
    img[a,b,0] = c[i][0]
    img[a,b,1] = c[i][1]
    img[a,b,2] = c[i][2]

Image.fromarray(img).save("labels.png")
    
