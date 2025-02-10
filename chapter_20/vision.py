#
#  file:  vision.py
#
#  Ollama vision with LLaVA
#
#  run first:
#    > pip3 install ollama
#
#  RTK, 02-Mar-2024
#  Last update:  02-Mar-2024
#
################################################################

import sys
import ollama

if (len(sys.argv) == 1):
    print()
    print("vision <model> <prompt> <file1> [<file2> ...]")
    print()
    print("  <model> - vision model (e.g. llava:7b)")
    print("  <prompt> - prompt (use double-quotes)")
    print("  <file1> ... image files")
    print()
    exit(0)

mname = sys.argv[1]
prompt = sys.argv[2]
files = sys.argv[3:]

for file in files:
    resp = ollama.chat(
        model=mname,
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [open(file,"rb").read()],
            },
        ]
    )

    print("File: %s\n" % file)
    print(resp['message']['content'])
    print()

