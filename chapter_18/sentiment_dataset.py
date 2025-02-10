#
#  file:  sentiment_dataset.py
#
#  Build a sentiment dataset
#
#  RTK, 26-Feb-2024
#  Last update:  13-Sep-2024
#
#  requires datasets:
#       > pip3 install datasets
#
################################################################

from datasets import load_dataset
import numpy as np
import requests
import json
import os
import sys

model = "nomic-embed-text"
nsamples = 20_000

ds = load_dataset('sentiment140', split='train', trust_remote_code=True)

text, sentiment = [], []
for sample in ds:
    text.append(sample['text'])
    sentiment.append(sample['sentiment'])

text = np.array(text)
labels = np.array(sentiment).astype("uint8")
labels[np.where(labels==4)] = 1

#  Randomize
np.random.seed(6502)
idx = np.argsort(np.random.random(len(labels)))
np.random.seed()
text = text[idx]
labels = labels[idx]

#  Keep samples and generate the text embeddings
text = list(text[:nsamples])
labels = labels[:nsamples]

#  Get the text embeddings using the selected model
url = "http://localhost:11434/api/embeddings"

embeddings = []
for prompt in text:
    #  Send the prompt and wait for the reply
    msg = {"model":model, "prompt":prompt, "stream":False}
    resp = requests.post(url, json=msg)
    if (resp.status_code != 200):
        raise ValueError("Bad response from Ollama server")
    
    #  Parse the reply
    reply = json.loads(resp.text)
    embeddings.append(reply['embedding'])
x = np.array(embeddings)

#  Store the embedding vectors and associated labels
n = int(len(labels)*0.1)  # 10 percent for test
xtst, ytst = x[:n], labels[:n]
xtrn, ytrn = x[n:], labels[n:]

np.save("sentiment140_xtrain.npy", xtrn.astype("float32"))
np.save("sentiment140_ytrain.npy", ytrn)
np.save("sentiment140_xtest.npy", xtst.astype("float32"))
np.save("sentiment140_ytest.npy", ytst)

