from examples_full import Problem


import numpy as np
import pandas as pd
import examples_full
import import_problems

filename = '/Users/aliciachen/Dropbox/teaching_models/teaching_stimuli - all_examples (9).csv'
all_problems = import_problems.df_from_csv(filename)


test = examples_full.Problem(all_problems, prob_idx=78)
test.view()
test.runModel(nIter=100)
# test.example_space()
