#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:22:34 2020

@author: aliciachen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from make_df import make_df_from_spreadsheet
filename = 'teaching_stimuli - all_examples.csv'
all_problems = make_df_from_spreadsheet(filename)

def find_problem(index, all_problems):
    '''Return flat and non flat representations of a specific problem'''
    h = np.array([i for i in all_problems.loc[index,:].to_numpy(dtype=object)])
    h_flat = all_problems.loc[index,:].to_numpy(dtype=object) # Instead of 3d matrix, create an array of arrays
    #print(h.shape)
    #print(h_flat.shape)
    return h, h_flat

def plot_problem(problem):
    fig, ax = plt.subplots(1,4, figsize = (16,4))
    opt_labels = ['$h_1$', '$h_2$', '$h_3$', '$h_4$']

    for idx,ax_i in enumerate(ax):
        hm = sns.heatmap(problem[idx,:,:], ax=ax_i,
                         cmap='gray_r', cbar=False,
                         linewidths=2, linecolor='#808080')
        ax_i.set(xticks=[], yticks=[], title=opt_labels[idx]) 
        
# k = 1

def get_pos_idx(concept):
    pos_coords = np.nonzero(concept)
    pos_idx = np.ravel_multi_index(pos_coords, (6,6))

    return list(pos_idx)

def find_pos_ex_indices(h_flat):
    '''
    Extract indices (flattened) of positive examples (starting from 0) for each h
    '''
    d_possible = {}
    #columns = ['h_1', 'h_2', 'h_3', 'h_4']
    #rows = ['d_%i' % (i+1) for i in range(h_flat[0].size)]

    for ex in range(len(h_flat)):
        d_possible[ex] = get_pos_idx(h_flat[ex])
    return d_possible

def make_df_iteration_zero(d_possible, h_flat):
    h_flat = h_flat
    df_0 = pd.DataFrame(index=[i for i in range(h_flat[0].size)], columns=[i for i in range(h_flat.size)])
 
    for k, v in d_possible.items(): 
        for i in v:
            df_0.loc[i, k] = 1
            
    # Drop rows without possible hypotheses, fill remaining NaNs with 0 
    # df_0 = df_0.dropna(how='all').fillna(0)

    # Columns should sum up to one (prior for each hypothesis is that you have an equal chance of teaching any of the data)
    df_0 = df_0.div(df_0.sum(axis=0), axis=1)
    df_0 = df_0.fillna(0)
    
    df_0.columns = ['h_1', 'h_2', 'h_3', 'h_4']
    return df_0

def plot_prob_heatmap(df, title):
    plt.figure(figsize=(4.8,7))
    sns.heatmap(df, annot=True, linewidths=0.25)
    plt.title(title)
    plt.show()
    
def find_teacher_probabilities_given_iter_0(n, df_0):
    '''
    given number of iterations n and P(d|h) matrix for iteration 0, find P(d|h) and P(h|d) matrix after iteration n 
    '''
    n_iter = n
    df_d = df_0

    for n in range(n_iter): 
        df_h = df_d.div(df_d.sum(axis=1), axis=0)  # P(h|d)
        df_d = df_h.div(df_h.sum(axis=0), axis=1)  # P(d|h)
    
    if n_iter == 0: 
        df_d = df_0 
        df_h = df_d.div(df_d.sum(axis=1), axis=0)
        
    return df_d, df_h

def find_teacher_probs_k1(n_iter, prob_idx, all_problems):
    '''given iteration number, index of problem, and df of all problems, return P(d|h) and P(h|d) after n interations for k=1'''
    _, h_flat = find_problem(prob_idx, all_problems)
    d_possible = find_pos_ex_indices(h_flat)
    df_0 = make_df_iteration_zero(d_possible, h_flat)
    df_d, df_h = find_teacher_probabilities_given_iter_0(n_iter, df_0)
    return df_d.fillna(0), df_h.fillna(0)

def drop_zero_rows(df):
    df = df.loc[(df!=0).any(axis=1)]
    return df

# Plot indices to make viz slightly easier
def plot_problem_with_indices(h_flat, problem):
    
    indices = np.arange(0, h_flat[0].size).reshape(6,6)
    fig, ax = plt.subplots(1,4, figsize = (16,4))
    opt_labels = '0123'

    for idx,ax_i in enumerate(ax):
        hm = sns.heatmap(problem[idx,:,:], ax=ax_i,
                         cmap='gray_r', cbar=False,
                         linewidths=2, linecolor='#808080', annot=indices)
        ax_i.set(xticks=[], yticks=[], title=opt_labels[idx]) 
        

def plot_prob_heatmap_k1(df, title):
    '''Plot probabilities heatmap given probabilities df'''
    problem = np.array([df[i].fillna(0).to_numpy().reshape(6,6) for i in df.columns])
    
    fig, ax = plt.subplots(1,4, figsize = (16,4))
    fig.suptitle(title, y=1.1)
    opt_labels = ['$h_1$', '$h_2$', '$h_3$', '$h_4$']

    for idx,ax_i in enumerate(ax):
        hm = sns.heatmap(problem[idx,:,:], ax=ax_i,
                         cmap='GnBu', cbar=False,
                         linewidths=2, linecolor='#808080', annot=True, fmt='.1g', annot_kws={"size":10})
        ax_i.set(xticks=[], yticks=[], title=opt_labels[idx]) 
    #return problem
    
def make_prob_heatmap_k1(df):
    probs = np.array([df[i].fillna(0).to_numpy().reshape(6,6) for i in df.columns])
    return probs

# Cases of k > 1

def make_empty_df_2ex():
    indices = []
    for i in range(36):
        for j in range(i+1, 36):
            indices.append((i,j))

    indices = pd.MultiIndex.from_tuples(indices, names=('i_1', 'i_2'))
    df_2ex = pd.DataFrame(columns=['h_1', 'h_2', 'h_3', 'h_4'], index=indices)
    
    return df_2ex

def make_empty_df_3ex():
    indices = []
    for i in range(36):
        for j in range(i+1, 36):
            for k in range(j+1, 36):
                indices.append((i,j,k))

    indices = pd.MultiIndex.from_tuples(indices, names=('i_1', 'i_2', 'i_3'))
    df_3ex = pd.DataFrame(columns=['h_1', 'h_2', 'h_3', 'h_4'], index=indices) # should have 36 choose 3 rows
    
    return df_3ex

def make_d_possible_with_column_labels(h_flat): 
    columns = ['h_1', 'h_2', 'h_3', 'h_4']
    d_possible = find_pos_ex_indices(h_flat)

    new_d_possible = {}
    for ex in range(len(columns)):
        new_d_possible[columns[ex]] = d_possible[ex]

    return new_d_possible

def normalize_probs_and_fill_nans(df): 
    df = df.div(df.sum(axis=0), axis=1)
    df = df.fillna(0)
    return df

def fill_df_2ex(h_flat):
    '''Fill df with initial probabilities'''
    df_2ex = make_empty_df_2ex()
    new_d_possible = make_d_possible_with_column_labels(h_flat)
    
    for column in df_2ex.columns:
        for i, j in df_2ex.index:
            if i in new_d_possible[column] and j in new_d_possible[column]:
                df_2ex.loc[(i,j), column] = 1
    
    df_2ex_short = df_2ex.dropna(how='all')
    
    df_2ex = normalize_probs_and_fill_nans(df_2ex)
    df_2ex_short = normalize_probs_and_fill_nans(df_2ex_short)
    
    return df_2ex, df_2ex_short

def find_teacher_probs_k2(n_iter, prob_idx, all_problems):
    '''given iteration number, index of problem, and df of all problems, return P(d|h) and P(h|d) after n interations for k=2'''
    _, h_flat = find_problem(prob_idx, all_problems)
    df_2ex_0, _ = fill_df_2ex(h_flat)
    df_d, df_h = find_teacher_probabilities_given_iter_0(n_iter, df_2ex_0)
    return df_d.fillna(0), df_h.fillna(0)

def make_prob_heatmap_k2(df):
    probs = np.array([np.zeros((6,6)) for column in df.columns])
    for h_idx in range(df.columns.size):
        for i_1, i_2 in df.index:
            probs[h_idx, np.unravel_index(i_1, (6,6))[0], np.unravel_index(i_1, (6,6))[1]] += df.loc[(i_1, i_2), df.columns[h_idx]]
            probs[h_idx, np.unravel_index(i_2, (6,6))[0], np.unravel_index(i_2, (6,6))[1]] += df.loc[(i_1, i_2), df.columns[h_idx]]
            
    # correct for double counting
    # probs = probs / 2
    return probs
#np.unravel_index(1621, (6,7,8,9))

def plot_problem_heatmap(df, title):
    '''Plot probabilities heatmap given probabilities df'''
    problem = df
    #problem = np.array([df[i].fillna(0).to_numpy().reshape(6,6) for i in df.columns])
    
    fig, ax = plt.subplots(1,4, figsize = (16,4))
    fig.suptitle(title, y=1.1)
    opt_labels = ['$h_1$', '$h_2$', '$h_3$', '$h_4$']

    for idx,ax_i in enumerate(ax):
        hm = sns.heatmap(problem[idx,:,:], ax=ax_i,
                         cmap='GnBu', cbar=False,
                         linewidths=2, linecolor='#808080', annot=True, fmt='.1g', annot_kws={"size":10})
        ax_i.set(xticks=[], yticks=[], title=opt_labels[idx]) 
    #return problem
    
def fill_df_3ex(h_flat):
    '''Fill df with initial probabilities'''
    df_3ex = make_empty_df_3ex()
    new_d_possible = make_d_possible_with_column_labels(h_flat)
    
    for column in df_3ex.columns:
        for i, j, k in df_3ex.index:
            if i in new_d_possible[column] and j in new_d_possible[column] and k in new_d_possible[column]:
                df_3ex.loc[(i,j,k), column] = 1
    
    df_3ex_short = df_3ex.dropna(how='all')
    
    df_3ex = normalize_probs_and_fill_nans(df_3ex)
    df_3ex_short = normalize_probs_and_fill_nans(df_3ex_short)
    
    return df_3ex, df_3ex_short

def find_teacher_probs_k3(n_iter, prob_idx, all_problems):
    '''given iteration number, index of problem, and df of all problems, return P(d|h) and P(h|d) after n interations for k=2'''
    _, h_flat = find_problem(prob_idx, all_problems)
    df_0, _ = fill_df_3ex(h_flat)
    df_d, df_h = find_teacher_probabilities_given_iter_0(n_iter, df_0)
    return df_d.fillna(0), df_h.fillna(0)

def make_prob_heatmap_k3(df):
    probs = np.array([np.zeros((6,6)) for column in df.columns])
    for h_idx in range(df.columns.size):
        for i_1, i_2, i_3 in df.index:
            probs[h_idx, np.unravel_index(i_1, (6,6))[0], np.unravel_index(i_1, (6,6))[1]] += df.loc[(i_1, i_2, i_3), df.columns[h_idx]]
            probs[h_idx, np.unravel_index(i_2, (6,6))[0], np.unravel_index(i_2, (6,6))[1]] += df.loc[(i_1, i_2, i_3), df.columns[h_idx]]
            probs[h_idx, np.unravel_index(i_3, (6,6))[0], np.unravel_index(i_3, (6,6))[1]] += df.loc[(i_1, i_2, i_3), df.columns[h_idx]]
    # correct for double counting
    # probs = probs / 6
    return probs
#np.unravel_index(1621, (6,7,8,9))

#%% Putting everything together

def find_teacher_probs(n_iter, prob_idx, all_problems):
    """Creates a dict of probability dataframes, indexed by k value and whether it is P(d|h) or P(h|d)"""
    my_dict = {}
    my_dict['n_iter'] = n_iter
    my_dict['problem_index'] = prob_idx
    
    for k in range(1,4):
        my_dict[k] = {}
    
    my_dict[1]['d'], my_dict[1]['h'] = find_teacher_probs_k1(n_iter, prob_idx, all_problems)
    my_dict[2]['d'], my_dict[2]['h'] = find_teacher_probs_k2(n_iter, prob_idx, all_problems)
    my_dict[3]['d'], my_dict[3]['h'] = find_teacher_probs_k3(n_iter, prob_idx, all_problems)
    
    return my_dict

# visualizing results 

def make_prob_heatmap(df):
    if len(df.index.names) == 1:
        heatmap = make_prob_heatmap_k1(df)
    elif len(df.index.names) == 2:
        heatmap = make_prob_heatmap_k2(df)
    else:
        heatmap = make_prob_heatmap_k3(df)
        
    return heatmap

def make_and_plot_prob_heatmap(df, title):
    heatmap = make_prob_heatmap(df)
    plot_problem_heatmap(heatmap, title)
    return heatmap

def sort_values_ascending_by_column(df, label):
    df = df.sort_values(by=[label], ascending=False)
    return df

def plot_some_examples(df, title):
    ''''''
    problem = df
    #problem = np.array([df[i].fillna(0).to_numpy().reshape(6,6) for i in df.columns])
    
    fig, ax = plt.subplots(1,4, figsize = (16,4))
    fig.suptitle(title, y=1.1)
    # opt_labels = ['$h_1$', '$h_2$', '$h_3$', '$h_4$']

    for idx,ax_i in enumerate(ax):
        hm = sns.heatmap(problem[idx,:,:], ax=ax_i,
                         cmap='GnBu', cbar=False,
                         linewidths=2, linecolor='#808080', annot=True, fmt='.1g', annot_kws={"size":10})
        ax_i.set(xticks=[], yticks=[]) 
    #return problem
        
def plot_high_prob_examples(prob, n_iter, types, idx, all_problems):
    '''given a probability df, plots 4 high probablility examples alongside the problem (designated by idx) for h1'''
    plot_problem(find_problem(idx, all_problems)[0]) 
    for k in range(1, 4):
        df = prob[n_iter][k][types]
        df = sort_values_ascending_by_column(df, 'h_1')
        ex = np.array([np.zeros((6,6)) for column in df.columns])
        #print(df)
        try: 
            _ = df.index.levels
        except AttributeError: 
            for i, row in df.head(4).reset_index().iterrows():
                ex[i, np.unravel_index(int(row[0]), (6,6))[0], np.unravel_index(int(row[0]), (6,6))[1]] = 1
        else:
            if len(df.index.levels) == 3: 
                for i, row in df.head(4).reset_index().iterrows():
                    ex[i, np.unravel_index(int(row['i_1']), (6,6))[0], np.unravel_index(int(row['i_1']), (6,6))[1]] = 1
                    ex[i, np.unravel_index(int(row['i_2']), (6,6))[0], np.unravel_index(int(row['i_2']), (6,6))[1]] = 1
                    ex[i, np.unravel_index(int(row['i_3']), (6,6))[0], np.unravel_index(int(row['i_3']), (6,6))[1]] = 1
            elif len(df.index.levels) == 2: 
                for i, row in df.head(4).reset_index().iterrows():
                    ex[i, np.unravel_index(int(row['i_1']), (6,6))[0], np.unravel_index(int(row['i_1']), (6,6))[1]] = 1
                    ex[i, np.unravel_index(int(row['i_2']), (6,6))[0], np.unravel_index(int(row['i_2']), (6,6))[1]] = 1

        plot_some_examples(ex, f'Examples for $h_1$, k={k}')
        
    
    #return ex