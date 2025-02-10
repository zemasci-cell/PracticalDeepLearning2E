#
#  file:  chat.py
#
#  Emulate a chatbot
#
#  RTK, 25-Feb-2024
#  Last update:  10-Mar-2024
#
###############################################################3

import os
import sys
import requests
import json

if (len(sys.argv) == 1):
    print()
    print("chat <model> [<outfile>]")
    print()
    print("  <model>   - model name (e.g. llama2)")
    print("  <outfile> - output file (optional)")
    print()
    exit(0)

model = sys.argv[1]
outfile = sys.argv[2] if (len(sys.argv) == 3) else ""

#  Ollama server interface for generating responses
url = "http://localhost:11434/api/chat"

prompt = input("]")
text = "]" + prompt + "\n"
print()
messages = [{
    "role": "user",
    "content": prompt
}]

while (prompt != ""):
    #  Get the model's reply
    msg = {
        "model":model, 
        "messages": messages,
        "stream":False
    }
    resp = requests.post(url, json=msg)
    if (resp.status_code != 200):
        raise ValueError("Bad response from Ollama server")

    #  Parse it and get the user's next prompt
    reply = json.loads(resp.text)
    ans = reply['message']['content']
    text += ans + "\n\n"
    print(ans)
    print()
    messages.append({
        "role": "assistant",
        "content": ans
    })
    prompt = input("]")
    if (prompt != ""):
        text += "]" + prompt + "\n"
        messages.append({
            "role": "user",
            "content": prompt
        })

if (outfile != ""):
    with open(outfile,"w") as f:
        f.write(text)

