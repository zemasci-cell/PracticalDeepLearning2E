#
#  file:  semantic.py
#
#  A simple example of semantic search using the nomic-embed-text
#  model
#
#  (see https://blog.nomic.ai/posts/nomic-embed-text-v1)
#
#  RTK, 03-Mar-2024
#  Last update: 03-Mar-2024
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

if (len(sys.argv) == 1):
    print()
    print("semantic <prompt>")
    print()
    print("  <prompt> - the semantic query")
    print()
    exit(0)

prompt = sys.argv[1]

#  Load Simpsons episode summaries and associated embeddings
text = [i[:-1] for i in open("simpsons_episodes.txt")]
embeddings = np.load("simpsons_embeddings.npy")

#  Ollama server interface for embeddings and model used
#  to generate the embeddings
url = "http://localhost:11434/api/embeddings"
model = "nomic-embed-text"

#  Get the prompt embedding
msg = {"model":model, "prompt":prompt, "stream":False}
resp = requests.post(url, json=msg)
if (resp.status_code != 200):
    raise ValueError("Bad response from Ollama server")
reply = json.loads(resp.text)
vec = np.array(reply['embedding'])

#  Find the cosine distances between episode summaries and the prompt
dist = []
for i in range(len(text)):
    dist.append(cosine(vec, embeddings[i]))
dist = np.array(dist)

#  Report the top-5 episodes (five smallest cosine distances)
print("Search results:")
idx = np.argsort(dist)
for i in idx[:5]:
    print("    (%0.6f) %s" % (dist[i], text[i]))
print()

