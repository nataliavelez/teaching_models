import numpy as np
import pandas as pd
import sequential
import import_problems

filename = '/Users/aliciachen/Dropbox/teaching_models/teaching_stimuli - all_examples (9).csv'
all_problems = import_problems.df_from_csv(filename)

testprob = sequential.Problem(all_problems, 91)
testprob.view()
testprob.example_space()
testprob.literal()


testprob.exs = ((2, 1), (2, 1), (2, 1), (2, 1))

hGd_seq, dGh_seq = testprob.sequential(nIter=50)
hGd_prag, dGh_prag = testprob.pragmatic(nIter=50)
hGd_lit = testprob.hGd_lit
# Plot

import matplotlib.pyplot as plt

h1Gd_seq = [
    hGd_seq[0].xs(testprob.exs[0]).iloc[0, 0],
    hGd_seq[1].xs(testprob.exs[1]).iloc[0, 0], 
    hGd_seq[2].xs(testprob.exs[2]).iloc[0, 0],
    hGd_seq[3].xs(testprob.exs[3]).iloc[0, 0],  
    ]

h1Gd_prag = [
    hGd_prag[0].xs(testprob.exs[0]).iloc[0, 0],
    hGd_prag[1].xs((testprob.exs[:2]))[0], 
    hGd_prag[2].xs((testprob.exs[:3]))[0],
    hGd_prag[3].xs(testprob.exs)[0], 
]

h1Gd_lit = [
    hGd_lit[0].xs(testprob.exs[0]).iloc[0, 0],
    hGd_lit[1].xs((testprob.exs[:2]))[0], 
    hGd_lit[2].xs((testprob.exs[:3]))[0],
    hGd_lit[3].xs(testprob.exs)[0], 
]

plt.figure()
plt.plot(range(4), h1Gd_seq, label="sequential")
# plt.plot(range(4), h1Gd_prag, label="pragmatic")
# plt.plot(range(4), h1Gd_lit, label="literal")
plt.xticks(range(4), testprob.exs)
plt.xlabel('example')
plt.ylabel('p(h1|d)')
plt.ylim((-0.05, 1.05))
plt.legend()