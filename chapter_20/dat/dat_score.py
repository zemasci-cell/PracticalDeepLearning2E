#
#  file:  dat_score.py
#
#  Calculate DAT scores using LLM embeddings
#
#  RTK, 26-Feb-2024
#  Last update:  26-Feb-2024
#
################################################################

import os
import sys
import numpy as np
import requests
import json

def cosine(a,b):
    """Calculate the cosine distance between two vectors"""
    num = np.dot(a,b)
    den = np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b))
    return 1.0 - num/den

def Score(words, model, url = "http://localhost:11434/api/embeddings"):
    """Score a collection of unique words"""
    #  Get the word embeddings
    embeddings = []
    for word in words:
        #  Get the word embedding
        msg = {"model":model, "prompt":word, "stream":False}
        resp = requests.post(url, json=msg)
        if (resp.status_code != 200):
            raise ValueError("Bad response from Ollama server")
        reply = json.loads(resp.text)
        embeddings.append(reply['embedding'])
    
    #  Pairwise cosine distances
    dist = []
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if (i==j):
                continue
            dist.append(cosine(embeddings[i], embeddings[j]))
    dist = np.array(dist) 
    return dist.mean()*100.0


if (len(sys.argv) == 1):
    print()
    print("dat_score <model> <words>")
    print()
    print("  <model> - model to use for embeddings (e.g. llama2)")
    print("  <words> - file of words (one run per line)")
    print()
    exit(0)

model = sys.argv[1]
wfile = sys.argv[2]

#  Load and score each list
lines = [i[:-1] for i in open(wfile)]
scores = []

for line in lines:
    if (line == ""):
        continue
    words = line.split(",")
    words = [i.strip() for i in words]
    words = list(set(words))  # unique words
    if (len(words) < 3):
        scores.append(0.0)
    else:
        scores.append(Score(words, model))

scores = np.array(scores)

for score in scores:
    print("%0.2f " % score, end="")
print("%0.4f %0.4f" % (scores.mean(), scores.std(ddof=1) / np.sqrt(len(scores))))

