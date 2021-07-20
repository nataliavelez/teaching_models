import numpy as np
import pandas as pd
import import_problems

filename = '/Users/aliciachen/Dropbox/teaching_models/teaching_stimuli - all_examples (9).csv'
all_problems = import_problems.df_from_csv(filename)

# %% Functions for recursion and for helper functions 

# TODO: function for finding new ex. all we have to do is to make df_0 that contains only the 
# next psosible examples minus the indices of the past examples, 
# everything else is as is and pretty good and then you can recur

# add a function that loops over past examples, splits past and future examples 
# %% 
class Problem: 

    def __init__(self, prob_idx, selected_idxs):

        self.prob_idx = prob_idx
        self.selected_idxs = selected_idxs


