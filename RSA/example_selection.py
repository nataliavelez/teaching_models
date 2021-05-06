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
    """same reward function for each"""
    return np.log(a)


def C1(idx, df, beta1):
    """imposes an extremely high cost when examples are redundant, basically forcing you to stop teaching
    maybe scale this later....
    """

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

    return beta1*min(divs)**(-1) if np.abs(beta1*min(divs)**(-1)) < 50 else 50

def C2(df, beta2):
    """linear cost"""
    return beta2*df.index.str.len()

#%% make dfs with utility

# Literal learner

df_lit['R'] = R(df_lit['h_1'])

for idx in df_lit.index:
    df_lit.loc[[idx], 'C1'] = C1(idx, df_lit, beta1=1)

df_lit['C2'] = C2(df_lit, beta2=1)

df_lit['U1'] = df_lit['R'] - df_lit['C1']
df_lit['U2'] = df_lit['R'] - df_lit['C2']

# Pragmatic learner

df_lit['R'] = R(df_prag['h_1'])

for idx in df_prag.index:
    df_prag.loc[[idx], 'C1'] = C1(idx, df_prag, beta1=1)

df_lit['C2'] = C2(df_prag, beta2=1)

df_prag['U1'] = df_prag['R'] - df_prag['C1']
df_prag['U2'] = df_prag['R'] - df_prag['C2']

#%% simulate!

# after selecting an example, make a sub dataframe with all the next possibilities. this will include utilities, then softmax
# want to store all before choices in a mtx

from numpy.random import rng
rng = default_rng()

alpha = 1

choices = []
all_indices = df_lit.index

# Add a staying option where cost is 0 and reward retains thte reward of the prev option

start = df_lit[df_lit.index.str.len() == 1]
start['prob'] = softmax(alpha * start['U'])

choice1 = rng.choice(start.index, 1, p = start['prob'])
choices.append(choice1)

choices2 = next_steps(choice1, all_indices)
sub_df_1 = df_lit.filter(items = choices2, axis=0)

sub_df_1['prob'] = softmax(alpha * sub_df_1['U'])