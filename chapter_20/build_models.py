#
#  file:  temp_build_models.py
#
#  Create new models with fixed temperatures
#
#  RTK, 13-Mar-2024
#  Last update:  13-Mar-2024
#
#################################################################

import ollama

# Create new models versions with different temperatures
modelfile="""
FROM gemma:7b-instruct
PARAMETER temperature 0.0
"""
ollama.create("gemma_temp_0p0", modelfile=modelfile)

modelfile="""
FROM gemma:7b-instruct
PARAMETER temperature 0.3
"""
ollama.create("gemma_temp_0p3", modelfile=modelfile)

modelfile="""
FROM gemma:7b-instruct
PARAMETER temperature 0.6
"""
ollama.create("gemma_temp_0p6", modelfile=modelfile)

modelfile="""
FROM gemma:7b-instruct
PARAMETER temperature 0.9
"""
ollama.create("gemma_temp_0p9", modelfile=modelfile)

