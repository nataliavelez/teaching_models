"""generate model predictions and plot heatmaps for the problems we're using"""

import numpy as np
import pandas as pd
import examples_full
import import_problems


filename = '/Users/aliciachen/Dropbox/teaching_models/teaching_stimuli - all_examples (9).csv'
all_problems = import_problems.df_from_csv(filename)

true_prob_idxs = [43, 47, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65] + [i for i in range(66, 94)] 

def gen_model_preds(true_prob_idxs): 
    """copied from analysis.py, organize later"""
    preds = {}
    for idx in true_prob_idxs: 
        print('Generating model predictions for problem ' + str(idx))
        preds[idx] = examples_full.Problem(all_problems, idx)
        preds[idx].runModel(nIter=40)
    return preds

preds = gen_model_preds(true_prob_idxs)

# %% Make  heatmaps


for i in true_prob_idxs: 
    preds[i].heatmaps()