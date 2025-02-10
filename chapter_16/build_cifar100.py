#
#  file:  build_cifar100.py
#
#  Process the raw CIFAR-100 dataset that you must download
#  from here (the Python version):
#
#  https://www.cs.toronto.edu/~kriz/cifar.html
#
#  Once downloaded as 'cifar-100-python.tar.gz' expand with:
#
#    tar xzf cifar-100-python.tar.gz
#
#  Then run this code to produce train and test NumPy files
#  in the ../data/cifar100 directory.
#
#  You'll find the fine and coarse labels at the end of this file.
#
#  RTK, 06-Jan-2024
#  Last update:  06-Jan-2024
#
################################################################

import os
import numpy as np

#  Unpickle code from the website
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

#  Transform into 32x32x3 arrays
def Process(d):
    im = np.zeros((len(d),32,32,3), dtype='uint8')
    for i in range(len(d)):
        im[i,:,:,:] = np.transpose(d[i].reshape(3,32,32),(1,2,0))
    return im

np.random.seed(73939133)

d = unpickle('cifar-100-python/train')
xtrn = Process(d[b'data'])
ytrnf = np.array(d[b'fine_labels'])
ytrnc = np.array(d[b'coarse_labels'])

idx = np.argsort(np.random.random(len(xtrn)))
xtrn, ytrnf, ytrnc = xtrn[idx], ytrnf[idx], ytrnc[idx]

d = unpickle('cifar-100-python/test')
xtst = Process(d[b'data'])
ytstf = np.array(d[b'fine_labels'])
ytstc = np.array(d[b'coarse_labels'])

idx = np.argsort(np.random.random(len(xtst)))
xtst, ytstf, ytstc = xtst[idx], ytstf[idx], ytstc[idx]

os.system("mkdir ../data/cifar100 2>/dev/null")

np.save("../data/cifar100/xtrain.npy", xtrn)
np.save("../data/cifar100/xtest.npy", xtst)

np.save("../data/cifar100/ytrainf.npy", ytrnf)
np.save("../data/cifar100/ytrainc.npy", ytrnc)

np.save("../data/cifar100/ytestf.npy", ytstf)
np.save("../data/cifar100/ytestc.npy", ytstc)

# fine labels:
#
#  0 apple
#  1 aquarium fish
#  2 baby
#  3 bear
#  4 beaver
#  5 bed
#  6 bee
#  7 beetle
#  8 bicycle
#  9 bottle
# 10 bowl
# 11 boy
# 12 bridge
# 13 bus
# 14 butterfly
# 15 camel
# 16 can
# 17 castle
# 18 caterpillar
# 19 cattle
# 20 chair
# 21 chimpanzee
# 22 clock
# 23 cloud
# 24 cockroach
# 25 couch
# 26 crab
# 27 crocodile
# 28 cup
# 29 dinosaur
# 30 dolphin
# 31 elephant
# 32 flatfish
# 33 forest
# 34 fox
# 35 girl
# 36 hamster
# 37 house
# 38 kangaroo
# 39 computer keyboard
# 40 lamp
# 41 lawn_mower
# 42 leopard
# 43 lion
# 44 lizard
# 45 lobster
# 46 man
# 47 maple tree
# 48 motorcycle
# 49 mountain
# 50 mouse
# 51 mushroom
# 52 oak_tree
# 53 orange
# 54 orchid
# 55 otter
# 56 palm tree
# 57 pear
# 58 pickup truck
# 59 pine tree
# 60 plain
# 61 plate
# 62 poppy
# 63 porcupine
# 64 possum
# 65 rabbit
# 66 raccoon
# 67 ray
# 68 road
# 69 rocket
# 70 rose
# 71 sea
# 72 seal
# 73 shark
# 74 shrew
# 75 skunk
# 76 skyscraper
# 77 snail
# 78 snake
# 79 spider
# 80 squirrel
# 81 streetcar
# 82 sunflower
# 83 sweet pepper
# 84 table
# 85 tank
# 86 telephone
# 87 television
# 88 tiger
# 89 tractor
# 90 train
# 91 trout
# 92 tulip
# 93 turtle
# 94 wardrobe
# 95 whale
# 96 willow tree
# 97 wolf
# 98 woman
# 99 worm

#  coarse labels and associated fine labels:
#
#  0 aquatic mammals                beaver, dolphin, otter, seal, whale
#  1 fish                           aquarium fish, flatfish, ray, shark, trout
#  2 flowers                        orchids, poppies, roses, sunflowers, tulips
#  3 food containers                bottles, bowls, cans, cups, plates
#  4 fruit and vegetables           apples, mushrooms, oranges, pears, sweet peppers
#  5 household electrical devices   clock, computer keyboard, lamp, telephone, television
#  6 household furniture            bed, chair, couch, table, wardrobe
#  7 insects                        bee, beetle, butterfly, caterpillar, cockroach
#  8 large carnivores               bear, leopard, lion, tiger, wolf
#  9 large man-made outdoor things  bridge, castle, house, road, skyscraper
# 10 large natural outdoor scenes   cloud, forest, mountain, plain, sea
# 11 large omnivores and herbivores camel, cattle, chimpanzee, elephant, kangaroo
# 12 medium-sized mammals           fox, porcupine, possum, raccoon, skunk
# 13 non-insect invertebrates       crab, lobster, snail, spider, worm
# 14 people                         baby, boy, girl, man, woman
# 15 reptiles                       crocodile, dinosaur, lizard, snake, turtle
# 16 small mammals                  hamster, mouse, rabbit, shrew, squirrel
# 17 trees                          maple, oak, palm, pine, willow
# 18 vehicles 1                     bicycle, bus, motorcycle, pickup truck, train
# 19 vehicles 2                     lawn-mower, rocket, streetcar, tank, tractor

