#
#  file:  dat_score_all.py
#
#  Test all the LLMs
#
#  RTK, 26-Feb-2024
#  Last update:  26-Feb-2024
#
################################################################

import os
import sys

if (len(sys.argv) == 1):
    print()
    print("dat_score_all <model>")
    print()
    print("  <model> -- embedding model name (e.g. llama2)")
    print()
    exit(0)

model = sys.argv[1]

dats = [
    "claude2", "gemini_1.0pro", "gemma_2b-instruct-fp16",
    "gemma_2b-instruct", "gemma_7b-instruct", "gpt4",
    "llama2_13b-chat", "llama2_7b-chat-fp16", "llama2_7b-chat",
    "mistral_7b-instruct", "llama2-unc_7b"
]

for d in dats:
    print("%-25s: " % d, end="", flush=True)
    cmd = "python3 dat_score.py %s dat_lists/dat_%s.txt" % (model, d)
    os.system(cmd)

