#  Group example image retrieval results
import numpy as np
from PIL import Image

classes = [ 
'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'
]

offset = 5

#  cosine
for cl in classes:
    base = "results/image/cosine/" + cl + "/"
    img = np.zeros(((32+offset)*5,32*11+offset,3), dtype="uint8")
    for j in range(5):
        img[j*(32+offset):(j*(32+offset)+32),:32,:] = np.array(Image.open(base+("image_%d.png" % j)))
        for k in range(10):
            im = np.array(Image.open(base+("image_%d_%d.png" % (j,k))))
            img[j*(32+offset):(j*(32+offset)+32),(offset+32+k*32):(offset+32+k*32+32),:] = im
    img[-offset:,:,:] = 255
    Image.fromarray(img).save(base+"montage.png")

for cl in classes:
    base = "results/image/cosine_layer10/" + cl + "/"
    img = np.zeros(((32+offset)*5,32*11+offset,3), dtype="uint8")
    for j in range(5):
        img[j*(32+offset):(j*(32+offset)+32),:32,:] = np.array(Image.open(base+("image_%d.png" % j)))
        for k in range(10):
            im = np.array(Image.open(base+("image_%d_%d.png" % (j,k))))
            img[j*(32+offset):(j*(32+offset)+32),(offset+32+k*32):(offset+32+k*32+32),:] = im
    img[-offset:,:,:] = 255
    Image.fromarray(img).save(base+"montage.png")

#  euclidean
for cl in classes:
    base = "results/image/euclidean/" + cl + "/"
    img = np.zeros(((32+offset)*5,32*11+offset,3), dtype="uint8")
    for j in range(5):
        img[j*(32+offset):(j*(32+offset)+32),:32,:] = np.array(Image.open(base+("image_%d.png" % j)))
        for k in range(10):
            im = np.array(Image.open(base+("image_%d_%d.png" % (j,k))))
            img[j*(32+offset):(j*(32+offset)+32),(offset+32+k*32):(offset+32+k*32+32),:] = im
    img[-offset:,:,:] = 255
    Image.fromarray(img).save(base+"montage.png")

for cl in classes:
    base = "results/image/euclidean_layer10/" + cl + "/"
    img = np.zeros(((32+offset)*5,32*11+offset,3), dtype="uint8")
    for j in range(5):
        img[j*(32+offset):(j*(32+offset)+32),:32,:] = np.array(Image.open(base+("image_%d.png" % j)))
        for k in range(10):
            im = np.array(Image.open(base+("image_%d_%d.png" % (j,k))))
            img[j*(32+offset):(j*(32+offset)+32),(offset+32+k*32):(offset+32+k*32+32),:] = im
    img[-offset:,:,:] = 255
    Image.fromarray(img).save(base+"montage.png")

#  grand montage by class
for cl in classes:
    i0 = np.array(Image.open("results/image/euclidean/" + cl + "/montage.png"))
    i1 = np.array(Image.open("results/image/cosine/" + cl + "/montage.png"))
    i2 = np.array(Image.open("results/image/euclidean_layer10/" + cl + "/montage.png"))
    i3 = np.array(Image.open("results/image/cosine_layer10/" + cl + "/montage.png"))
    img = np.zeros((2*185+5, 2*357+10, 3), dtype="uint8")
    img[...] = 255
    img[:185,:357,:] = i0
    img[:185,(357+10):,:] = i1
    img[(185+5):,:357,:] = i2
    img[(185+5):,(357+10):,:] = i3
    Image.fromarray(img).save("results/image/"+cl+".png")

