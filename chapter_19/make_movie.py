import numpy as np
import os
from PIL import Image

fnames = [i[:-1] for i in open("filelist.txt")]

k = 0
for f in fnames:
    im = Image.open("images/%s" % f).resize((400,400)).convert("RGB")
    im.save("frames/frame_%04d.jpg" % k)
    k += 1

cmd = "cd frames; ffmpeg -framerate 10 -i frame_%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p mnist_gan_40.mp4"
os.system(cmd)
os.system("mv frames/mnist_gan_40.mp4 .")

