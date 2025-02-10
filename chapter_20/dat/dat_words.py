#
#  file:  dat_words.py
#
#  Test models as a function of temperature
#
#  Run 'build_models.py' first
#
#  RTK, 13-Mar-2024
#  Last update: 13-Mar-2024
#
################################################################

import os
import sys
import requests
import json
import numpy as np

def GetResponse(model):
    """Get the model's response for a given temperature"""
    url = "http://localhost:11434/api/generate"
    msg = {
        "model":model, 
        "prompt":open("dat_prompt.txt","r").read(),
        "stream":False, 
    }
    resp = requests.post(url, json=msg)
    if (resp.status_code != 200):
        raise ValueError("Bad response from Ollama server")
    reply = json.loads(resp.text)
    return reply['response'].replace("\n"," ")


print("0.0 :  %s" % GetResponse("gemma_temp_0p0"))
print()
print("0.3 :  %s" % GetResponse("gemma_temp_0p3"))
print()
print("0.6 :  %s" % GetResponse("gemma_temp_0p6"))
print()
print("0.9 :  %s" % GetResponse("gemma_temp_0p9"))
print()

