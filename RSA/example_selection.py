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
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

filename = 'teaching_stimuli - all_examples (7).csv'
all_problems = make_df_from_spreadsheet(filename)

#%%

df_500 = find_teacher_probs(500, 65, all_problems)
df_0 = find_teacher_probs(0, 65, all_problems)

#%%

plot_problem(find_problem(65, all_problems)[0])
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

# for idx in df_lit.index:
#     df_lit.loc[[idx], 'C1'] = C1(idx, df_lit, beta1=1)

df_lit['C2'] = C2(df_lit, beta2=1)

# df_lit['U1'] = df_lit['R'] - df_lit['C1']
df_lit['U2'] = df_lit['R'] - df_lit['C2']

# Pragmatic learner

df_prag['R'] = R(df_prag['h_1'])

# for idx in df_prag.index:
#     df_prag.loc[[idx], 'C1'] = C1(idx, df_prag, beta1=1)

df_prag['C2'] = C2(df_prag, beta2=1)

# df_prag['U1'] = df_prag['R'] - df_prag['C1']
df_prag['U2'] = df_prag['R'] - df_prag['C2']

#%% simulate!

# after selecting an example, make a sub dataframe with all the next possibilities. this will include utilities, then softmax
# want to store all before choices in a mtx

rng = np.random.default_rng()

def make_choices(df, alpha):
    alpha = alpha

    choices = []
    all_indices = df.index

    # Add a staying option where cost is 0 and reward retains thte reward of the prev option

    start = df[df.index.str.len() == 1]
    start['prob'] = softmax(alpha * start['U2'])

    choice1 = rng.choice(start.index.tolist(), 1, p = start['prob'])
    choice1 = tuple(choice1.flatten()) # probably not ideal... annoying cause rng outputs a numpy array
    choices.append(choice1)

    for i in range(6):

        choices2 = next_steps(choice1, all_indices)
        sub_df_1 = df.filter(items = choices2, axis=0)
        sub_df_1.loc['term', 'U2'] = df.loc[[choice1], 'R'].values # add staying option

        sub_df_1['prob'] = softmax(alpha * sub_df_1['U2'])

        choice2 = rng.choice(sub_df_1.index.tolist(), 1, p = sub_df_1['prob'])
        choice2 = choice2[0]

        if choice2 == 'term':
            break

        #choice2 = tuple(choice2.flatten())
        choices.append(choice2)
        choice1 = choice2

    return choices

#%%


print(make_choices(df_lit, alpha=1))
print(make_choices(df_prag, alpha=1))

#%% Plots

# Avg length of utterance until termination, for lit and prag learners

B = 1000

lit_lengths = []
prag_lengths = []

for i in range(B):

    lc = make_choices(df_lit, alpha=1)
    pc = make_choices(df_prag, alpha=1)

    lit_lengths.append(len(lc))
    prag_lengths.append(len(pc))





plt.figure()
plt.hist(lit_lengths)
plt.title('Literal learner, avg utterance length')

plt.figure()
plt.hist(prag_lengths)
plt.title('Pragmatic learner, avg utterance length')

# Termination probabilities for literal and pragmatic learner at each step

# Can we recover mean and variance parameters, as well as ground truth probabilities?

ll = np.array([lit_lengths])
pl = np.array([prag_lengths])

stop_probs_lit = []
stop_probs_prag = []

for i in range(1, 6):
    split = np.sum(ll == i) / np.sum(ll >= i)
    spprag = np.sum(pl == i) / np.sum(pl >= i)
    stop_probs_lit.append(split)
    stop_probs_prag.append(spprag)

plt.figure()
plt.plot(range(1, 6), stop_probs_lit, label='lit learner stop prob')
plt.xticks(range(1, 6))
plt.plot(range(1, 6), stop_probs_prag, label='prag learner stop prob')
plt.legend()