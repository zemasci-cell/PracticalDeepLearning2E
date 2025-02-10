Divergent Association Task
--------------------------(13-Mar-2024)

The files here experiment with the DAT method of scoring a type
of creativity.

The original paper is present: Naming_Unrelated_Words_Predicts_Creativity.pdf

The authors' GitHub repo is in `divergent-association-task', see the license
file for specifics on using their code.  Actual page:

https://github.com/jayolson/divergent-association-task

Read the source code in 'dat_grade.py' for the URL of the GLoVE data needed
to score a list of words.  You'll need to download it yourself (~5 GB expanded)

The directory 'dat_lists' contains sets of words returned by different LLMs, 
one set per line.  Use these with 'dat_grade.py'.

The file 'dat_score.py' uses LLM embeddings to score the words lists, for
comparison with the official DAT approach.

Use 'dat_temperature.py' to explore if there's a relationship between LLM
temperature and creativity on the DAT.

Have fun!

