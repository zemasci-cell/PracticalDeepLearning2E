#
#  file:  dogs_and_cats.py
#
#  Compare the embedding vectors for dogs, cats, and wolves
#
#  Run embeddings.py on dogs_cats_wolves.txt first
#
#  RTK, 10-Mar-2024
#  Last update:  11-Mar-2024
#
################################################################

import numpy as np
import matplotlib.pylab as plt
from sklearn.manifold import TSNE

def cosine(a,b):
    """Calculate the cosine distance between two vectors"""
    num = np.dot(a,b)
    den = np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b))
    return 1.0 - num/den

#  Get the embeddings
em = np.load("dogs_cats_wolves.npy")
dogs, cats, wolves = em[:6], em[6:12], em[12:]

#  Report mean within group cosine distances
d = []
for i in range(len(dogs)):
    for j in range(len(dogs)):
        if (i==j): continue
        d.append(cosine(dogs[i],dogs[j]))
mean_dogs = sum(d) / len(d)

d = []
for i in range(len(cats)):
    for j in range(len(cats)):
        if (i==j): continue
        d.append(cosine(cats[i],cats[j]))
mean_cats = sum(d) / len(d)

d = []
for i in range(len(wolves)):
    for j in range(len(wolves)):
        if (i==j): continue
        d.append(cosine(wolves[i],wolves[j]))
mean_wolves = sum(d) / len(d)

#  Now, mean between group distances
d = []
for i in range(len(dogs)):
    for j in range(len(cats)):
        d.append(cosine(dogs[i],cats[j]))
mean_dogs_cats = sum(d) / len(d)

d = []
for i in range(len(dogs)):
    for j in range(len(wolves)):
        d.append(cosine(dogs[i],wolves[j]))
mean_dogs_wolves = sum(d) / len(d)

d = []
for i in range(len(wolves)):
    for j in range(len(cats)):
        d.append(cosine(wolves[i],cats[j]))
mean_cats_wolves = sum(d) / len(d)

print("Mean cosine distances:")
print("    dogs          : %0.8f" % mean_dogs)
print("    cats          : %0.8f" % mean_cats)
print("    wolves        : %0.8f" % mean_wolves)
print()
print("    dogs vs cats  : %0.8f" % mean_dogs_cats)
print("    dogs vs wolves: %0.8f" % mean_dogs_wolves)
print("    cats vs wolves: %0.8f" % mean_cats_wolves)
print()

x = np.vstack((dogs,cats,wolves))
y = TSNE(n_components=2, perplexity=3, random_state=6502).fit_transform(x)
l = np.array([0]*6 + [1]*6 + [2]*4)
lbl = ['dog', 'cat', 'wolf']
pd, pc, pw = False, False, False
m = ['o','s','^']
for i in range(len(y)):
    if (not pd) and (l[i]==0):
        plt.plot([y[i,0]],[y[i,1]], marker=m[l[i]], fillstyle='none', linestyle='none', color='k', label=lbl[l[i]])
        pd = True
    elif (not pc) and (l[i]==1):
        plt.plot([y[i,0]],[y[i,1]], marker=m[l[i]], fillstyle='none', linestyle='none', color='k', label=lbl[l[i]])
        pc = True
    elif (not pw) and (l[i]==2):
        plt.plot([y[i,0]],[y[i,1]], marker=m[l[i]], fillstyle='none', linestyle='none', color='k', label=lbl[l[i]])
        pw = True
    else:
        plt.plot([y[i,0]],[y[i,1]], marker=m[l[i]], fillstyle='none', linestyle='none', color='k')
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.legend(loc='best')
plt.tight_layout(pad=0.25, w_pad=0, h_pad=0)
plt.savefig("dogs_cats_wolves.eps", dpi=300)
plt.savefig("dogs_cats_wolves.png", dpi=300)
plt.show()

    
