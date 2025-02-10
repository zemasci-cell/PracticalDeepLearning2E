#
#  file:  embeddings.py
#
#  Return model embeddings using Ollama
#
#  RTK, 25-Feb-2024
#  Last update:  25-Feb-2024
#
################################################################

import os
import sys
import numpy as np
import requests
import json

if (len(sys.argv) == 1):
    print()
    print("embeddings <prompts> <outfile>")
    print()
    print("  <prompts>     - file of prompts | a string")
    print("  <outfile>     - output file (.npy)")
    print()
    exit(0)

pfile = sys.argv[1]
outfile = sys.argv[2]

#  ollama server interface for embeddings
url = "http://localhost:11434/api/embeddings"

#  Act based on pfile
if (not os.path.exists(pfile)):
    #  pfile is a single prompt
    prompts = [pfile]
else:
    #  a file of prompts, one per line
    prompts = [i[:-1] for i in open(pfile)]

embeddings = []
for prompt in prompts:
    if (prompt == ""):
        continue

    #  Send the prompt and wait for the reply
    msg = {"model":"nomic-embed-text", "prompt":prompt, "stream":False}
    resp = requests.post(url, json=msg)
    if (resp.status_code != 200):
        raise ValueError("Bad response from Ollama server")
    
    #  Parse the reply
    reply = json.loads(resp.text)
    embeddings.append(reply['embedding'])

#  Store the embedding vectors
np.save(outfile, np.array(embeddings))

