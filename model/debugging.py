import examples_full


import numpy as np
import pandas as pd
import examples_full
import import_problems

filename = '/Users/aliciachen/Dropbox/teaching_models/teaching_stimuli - all_examples (9).csv'
all_problems = import_problems.df_from_csv(filename)


test = examples_full.Problem(all_problems, prob_idx=78)
test.view()
test.example_space()

#%%

def flat_idx_to_tuple(idx): 
    """Convert 0 to 35 index to tuple of coords"""
    row = (idx) // 6
    col = (idx) % 6
    return (row, col)

#%%
test.runModel(nIter=20)
# test.example_space()

# %%
test.hGd_prag[1].sort_values(by="h1")
# %%
