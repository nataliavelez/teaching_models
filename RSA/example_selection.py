#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:44:49 2021

@author: aliciachen
"""

import numpy as np
import pandas as pd
from rectangle_model import *
from make_df import *

filename = 'teaching_stimuli - all_examples (7).csv'
all_problems = make_df_from_spreadsheet(filename)

#%%

df_500 = find_teacher_probs(500, 63, all_problems)
df_0 = find_teacher_probs(0, 63, all_problems)

#%%

def posterior_probs(df):
    frames = [df[1]['h'].rename(index=lambda a: (a,)),
              df[2]['h'],
              df[3]['h'],
              df[4]['h'],
              df[5]['h'],
              df[6]['h']]

    # Create a new df where rows are *all* possible examples, and column is p(h_1 | d)

    new_df = pd.concat(frames)
    new_df.drop(columns=['h_2', 'h_3', 'h_4'], inplace=True)
    new_df = new_df[new_df.h_1 != 0] # only positive examples
    return new_df

# separate dfs for pragmatic and literal learnerr
df_prag = posterior_probs(df_500)
df_lit = posterior_probs(df_0)
#%%

def next_steps(indices, all_indices):
    """Given a size n tuple of indices, find all tuples of size n+1 that contain that set of indices
    all_indices is generally df.index
    """
    nextsteps = []
    current_step = set(indices)
    for idx in all_indices:
        if current_step.issubset(set(idx)) and len(idx) == len(indices)+1:
            nextsteps.append(idx)

    return nextsteps

# Sanity check
print(next_steps((7,), df.index))
print(next_steps((7, 14), df.index))
print(next_steps((7, 14, 32), df.index))

# yay it works!!

#%% Cost and reward functions

# a is posterior probability h|d

def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))

def kl(a, b):
    return np.log(a) / np.log(b)


def R(a):
    "same reward function for each"
    return np.log(a)


def C1(idx, df, beta1):
    "imposes an extremely high cost when examples are redundant, basically forcing you to stop teaching"
    divs = []

    if len(idx) == 1:  # Case of only 1 example: zero cost
        return 0

    def find_subsets(idx, df):
        subsets = []
        for i in df.index:
            if set(i).issubset(set(idx)):
                subsets.append(i)

        #subsets.append(idx) # add original set of indices to subset
        return subsets

    subsets = find_subsets(idx, df)
    #return subsets

    for i in range(len(subsets)):
        for j in range(i+1, len(subsets)):
            if set(subsets[i]).issubset(set(subsets[j])) and len(subsets[j])-len(subsets[i]) == 1:
                #print(subsets[i])
                #print(subsets[j])
                a, b = float(df.loc[[subsets[j]], 'h_1']), float(df.loc[[subsets[i]], 'h_1'])
                #print(a)
                #print(b)
                div = kl(a, b)
                divs.append(div)

    return beta1*min(divs)**(-1)

def C2(a, beta2):
    "linear cost"
    return beta2*len(a)