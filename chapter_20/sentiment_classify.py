#
#  file:  sentiment_classify.py
#
#  Use the sentiment classifier to label user text
#
#  RTK, 27-Feb-2024
#  Last update:  28-Feb-2024
#
################################################################

import os
import sys
import numpy as np
import requests
import json
from tensorflow.keras.models import load_model

if (len(sys.argv) == 1):
    print()
    print("sentiment_classify <model>")
    print()
    print("  <model>      - a trained sentiment classifier (.keras)")
    print()
    exit(0)

model = load_model(sys.argv[1])

prompt = input("]")
if (prompt == ""):
    exit(0)

while (prompt != ""):
    #  Send the prompt and wait for the reply
    msg = {"model":"nomic-embed-text", "prompt":prompt, "stream":False}
    resp = requests.post("http://localhost:11434/api/embeddings", json=msg)
    if (resp.status_code != 200):
        raise ValueError("Bad response from Ollama server")
    reply = json.loads(resp.text)

    #  Classify the embedding
    embedding = (np.array(reply['embedding']) + 7) / 20
    pred = model.predict(embedding.reshape((1,len(embedding))), verbose=0)[0]
    if (pred >= 0.5):
        print("  *** positive *** (%0.8f)" % pred)
    else:
        print("  *** negative *** (%0.8f)" % pred)

    #  Get next statement
    prompt = input("]")

