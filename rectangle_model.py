#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

def find_problem(index, all_problems):
    """Return flat and non flat representations of a specific problem"""
    # Make 3d matrix
    h = np.array([i for i in all_problems.loc[index, :]
                  .to_numpy(dtype=object)])

    # Make array of arrays
    h_flat = all_problems.loc[index, :].to_numpy(dtype=object)

    return h, h_flat


def plot_problem(h):
    """Plot problem given 3d matrix representation of h"""
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    opt_labels = ['$h_1$', '$h_2$', '$h_3$', '$h_4$']

    for idx, ax_i in enumerate(ax):
        hm = sns.heatmap(h[idx, :, :], ax=ax_i,
                         cmap='gray_r', cbar=False,
                         linewidths=2, linecolor='#808080')
        ax_i.set(xticks=[], yticks=[], title=opt_labels[idx])


def get_pos_idx(concept):
    pos_coords = np.nonzero(concept)
    pos_idx = np.ravel_multi_index(pos_coords, (6, 6))

    return list(pos_idx)


def find_pos_ex_indices(h_flat):
    """Extract indices (flattened) of positive examples for each h_i"""
    d_possible = {}

    for ex in range(len(h_flat)):
        d_possible[ex] = get_pos_idx(h_flat[ex])

    return d_possible


def make_df_iteration_zero(d_possible, h_flat):
    """Given pos example indices, return normalized iteration 0 df"""
    df_0 = pd.DataFrame(index=[i for i in range(h_flat[0].size)],
                        columns=[i for i in range(h_flat.size)])

    for k, v in d_possible.items():
        for i in v:
            df_0.loc[i, k] = 1

    # Columns should sum up to 1
    df_0 = df_0.div(df_0.sum(axis=1).replace(0, np.nan), axis=0)
    df_0 = df_0.fillna(0)

    df_0.columns = ['h_1', 'h_2', 'h_3', 'h_4']

    return df_0


def plot_prob_heatmap(df, title):
    """Plot heatmap similar to that in the Shafto paper (not used here)"""
    plt.figure(figsize=(4.8, 7))
    sns.heatmap(df, annot=True, linewidths=0.25)
    plt.title(title)
    plt.show()


def find_teacher_probabilities_given_iter_0(n, df_0):
    """
    Iterate over the model.

    Args:
        n (int): number of iterations
        df_0 (DataFrame): P(h|d) for iteration 0

    Returns:
        df_d (DataFrame): P(d|h) after iteration n
        df_h (DataFrame): P(h|d) after iteration n

    """
    n_iter = n
    df_h = df_0

    for n in range(n_iter):
        df_d = df_h.div(df_h.sum(axis=0), axis=1)  # P(d|h)
        df_h = df_d.div(df_d.sum(axis=1).replace(0, np.nan), axis=0)  # P(h|d)

    if n_iter == 0:
        df_h = df_0
        df_d = df_h.div(df_h.sum(axis=0), axis=1)

    return df_d, df_h


def find_teacher_probs_k1(n_iter, prob_idx, all_problems):
    """
    Return P(d|h) and P(h|d) after n interations for k=1

    Args:
        n_iter (int): number of iterations
        prob_idx (str): index of problem
        all_problems (DataFrame): df of all problems

    Returns:
        df_d (DataFrame): P(d|h) after n_iter
        df_h (DataFrame): P(h|d) after n_iter

    """
    _, h_flat = find_problem(prob_idx, all_problems)
    d_possible = find_pos_ex_indices(h_flat)
    df_0 = make_df_iteration_zero(d_possible, h_flat)
    df_d, df_h = find_teacher_probabilities_given_iter_0(n_iter, df_0)

    df_d = df_d.fillna(0)
    df_h = df_h.fillna(0)

    return df_d, df_h


def drop_zero_rows(df):
    df = df.loc[(df != 0).any(axis=1)]
    return df


def plot_problem_with_indices(h_flat, h):
    """Plot problem with indices to make viz slightly easier"""
    indices = np.arange(0, h_flat[0].size).reshape(6, 6)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    opt_labels = '0123'

    for idx, ax_i in enumerate(ax):
        hm = sns.heatmap(h[idx, :, :], ax=ax_i,
                         cmap='gray_r', cbar=False,
                         linewidths=2, linecolor='#808080', annot=indices)
        ax_i.set(xticks=[], yticks=[], title=opt_labels[idx])


def make_prob_heatmap_k1(df):
    probs = np.array([df[i].fillna(0).to_numpy().reshape(6,6)
                      for i in df.columns])
    return probs


def plot_prob_heatmap_k1(df, title):
    '''Plot probabilities heatmap for k=1 given either df_d or df_h'''
    problem = make_prob_heatmap_k1(df)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, y=1.1)
    opt_labels = ['$h_1$', '$h_2$', '$h_3$', '$h_4$']

    for idx, ax_i in enumerate(ax):
        hm = sns.heatmap(problem[idx, :, :], ax=ax_i,
                         cmap='GnBu', cbar=False,
                         linewidths=2, linecolor='#808080', annot=True,
                         fmt='.1g', annot_kws={"size": 10})
        ax_i.set(xticks=[], yticks=[], title=opt_labels[idx])


def make_empty_df_2ex():
    indices = []
    for i in range(36):
        for j in range(i+1, 36):
            indices.append((i, j))

    indices = pd.MultiIndex.from_tuples(indices, names=('i_1', 'i_2'))
    df_2ex = pd.DataFrame(columns=['h_1', 'h_2', 'h_3', 'h_4'], index=indices)

    return df_2ex


def make_empty_df_3ex():
    indices = []
    for i in range(36):
        for j in range(i+1, 36):
            for k in range(j+1, 36):
                indices.append((i, j, k))

    indices = pd.MultiIndex.from_tuples(indices, names=('i_1', 'i_2', 'i_3'))
    df_3ex = pd.DataFrame(columns=['h_1', 'h_2', 'h_3', 'h_4'], index=indices)
    return df_3ex


def make_d_possible_with_column_labels(h_flat):
    """Find possible indices, but keys are labels h_i instead"""
    columns = ['h_1', 'h_2', 'h_3', 'h_4']
    d_possible = find_pos_ex_indices(h_flat)

    new_d_possible = {}
    for ex in range(len(columns)):
        new_d_possible[columns[ex]] = d_possible[ex]

    return new_d_possible


def normalize_probs_and_fill_nans(df):
    """Columns (each of the h_i) should sum up to 1"""
    df = df.div(df.sum(axis=1).replace(0, np.nan), axis=0)
    df = df.fillna(0)
    return df


def fill_df_2ex(h_flat):
    '''Fill df with initial probabilities for k=2'''
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
    """
    Return P(d|h) and P(h|d) after n interations for k=2

    Args:
        n_iter (int): number of iterations
        prob_idx (str): index of problem
        all_problems (DataFrame): df of all problems

    Returns:
        df_d (DataFrame): P(d|h) after n_iter
        df_h (DataFrame): P(h|d) after n_iter
    """
    _, h_flat = find_problem(prob_idx, all_problems)
    df_2ex_0, _ = fill_df_2ex(h_flat)
    df_d, df_h = find_teacher_probabilities_given_iter_0(n_iter, df_2ex_0)

    df_d = df_d.fillna(0)
    df_h = df_h.fillna(0)

    return df_d, df_h

def make_prob_heatmap_k2(df):
    probs = np.array([np.zeros((6, 6)) for column in df.columns])

    for h_idx in range(df.columns.size):
        for i_1, i_2 in df.index:
            probs[h_idx, np.unravel_index(i_1, (6, 6))[0],
                  np.unravel_index(i_1, (6, 6))[1]] += df.loc[(i_1, i_2),
                                                              df.columns[h_idx]]
            probs[h_idx, np.unravel_index(i_2, (6, 6))[0],
                  np.unravel_index(i_2, (6, 6))[1]] += df.loc[(i_1, i_2),
                                                              df.columns[h_idx]]

    # correct for double counting
    # probs = probs / 2
    return probs


def plot_problem_heatmap(df, title):
    '''Plot probabilities heatmap given probabilities df'''
    problem = df
    # problem = np.array([df[i].fillna(0).to_numpy().reshape(6,6) for i in df.columns])

    fig, ax = plt.subplots(1, 4, figsize = (16, 4))
    fig.suptitle(title, y=1.1)
    opt_labels = ['$h_1$', '$h_2$', '$h_3$', '$h_4$']

    for idx, ax_i in enumerate(ax):
        hm = sns.heatmap(problem[idx, :, :], ax=ax_i,
                         cmap='GnBu', cbar=False,
                         linewidths=2, linecolor='#808080', annot=True,
                         fmt='.1g', annot_kws={"size": 10})
        ax_i.set(xticks=[], yticks=[], title=opt_labels[idx])


def fill_df_3ex(h_flat):
    '''Fill df with initial probabilities for k=3'''
    df_3ex = make_empty_df_3ex()
    new_d_possible = make_d_possible_with_column_labels(h_flat)

    for column in df_3ex.columns:
        for i, j, k in df_3ex.index:
            if i in new_d_possible[column] and j in new_d_possible[column] and k in new_d_possible[column]:
                df_3ex.loc[(i, j, k), column] = 1

    df_3ex_short = df_3ex.dropna(how='all')

    df_3ex = normalize_probs_and_fill_nans(df_3ex)
    df_3ex_short = normalize_probs_and_fill_nans(df_3ex_short)

    return df_3ex, df_3ex_short


def find_teacher_probs_k3(n_iter, prob_idx, all_problems):
    """
    Return P(d|h) and P(h|d) after n interations for k=3

    Args:
        n_iter (int): number of iterations
        prob_idx (str): index of problem
        all_problems (DataFrame): df of all problems

    Returns:
        df_d (DataFrame): P(d|h) after n_iter
        df_h (DataFrame): P(h|d) after n_iter
    """
    _, h_flat = find_problem(prob_idx, all_problems)
    df_0, _ = fill_df_3ex(h_flat)
    df_d, df_h = find_teacher_probabilities_given_iter_0(n_iter, df_0)

    df_d = df_d.fillna(0)
    df_h = df_h.fillna(0)

    return df_d, df_h


def make_prob_heatmap_k3(df):
    probs = np.array([np.zeros((6, 6)) for column in df.columns])

    for h_idx in range(df.columns.size):
        for i_1, i_2, i_3 in df.index:
            probs[h_idx, np.unravel_index(i_1, (6, 6))[0],
                  np.unravel_index(i_1, (6, 6))[1]] += df.loc[(i_1, i_2, i_3),
                                                             df.columns[h_idx]]
            probs[h_idx, np.unravel_index(i_2, (6, 6))[0],
                  np.unravel_index(i_2, (6, 6))[1]] += df.loc[(i_1, i_2, i_3),
                                                             df.columns[h_idx]]
            probs[h_idx, np.unravel_index(i_3, (6, 6))[0],
                  np.unravel_index(i_3, (6, 6))[1]] += df.loc[(i_1, i_2, i_3),
                                                             df.columns[h_idx]]
    # probs = probs / 6
    return probs


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
    '''Plot four examples'''
    problem = df

    fig, ax = plt.subplots(1, 4, figsize = (16, 4))
    fig.suptitle(title, y=1.1)

    for idx,ax_i in enumerate(ax):
        hm = sns.heatmap(problem[idx,:,:], ax=ax_i,
                         cmap='GnBu', cbar=False,
                         linewidths=2, linecolor='#808080', annot=True, fmt='.1g',
                         annot_kws={"size": 10})
        ax_i.set(xticks=[], yticks=[])


# %% External functions

def find_teacher_probs(n_iter, prob_idx, all_problems):
    """
    Creates a dict of probability dataframes for a specific problem

    Args:
        n_iter (int): n iterations
        prob_idx (int): index of problem
        all_problems (df): df of all problems

    Returns:
        my_dict (dict): dict indexed by k and ('h' or 'd')

    """
    my_dict = {}
    my_dict['n_iter'] = n_iter
    my_dict['problem_index'] = prob_idx

    for k in range(1, 4):
        my_dict[k] = {}

    my_dict[1]['d'], my_dict[1]['h'] = find_teacher_probs_k1(n_iter, prob_idx, all_problems)
    my_dict[2]['d'], my_dict[2]['h'] = find_teacher_probs_k2(n_iter, prob_idx, all_problems)
    my_dict[3]['d'], my_dict[3]['h'] = find_teacher_probs_k3(n_iter, prob_idx, all_problems)

    return my_dict


def plot_high_prob_examples(prob, n_iter, types, idx, all_problems):
    """
    Plot 4 high probablility examples for h_1, alongside the problem

    Args:
        prob (dict): dict of prob dfs for a specific problem (output of find_teacher_probs)
        n_iter (int): # iterations
        types (str): either 'd' (for P(d|h)) or 'h' (for P(h|d))
        idx (int): index of problem (for title)
        all_problems (df): df of all problems

    Returns:
        None.

    """
    plot_problem(find_problem(idx, all_problems)[0])

    for k in range(1, 4):
        df = prob[n_iter][k][types]
        df = sort_values_ascending_by_column(df, 'h_1')
        ex = np.array([np.zeros((6, 6)) for column in df.columns])

        try:
            _ = df.index.levels
        except AttributeError:  # No index levels, meaning that k=1
            for i, row in df.head(4).reset_index().iterrows():
                ex[i, np.unravel_index(int(row[0]), (6, 6))[0], np.unravel_index(int(row[0]), (6, 6))[1]] = 1
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


def plot_pragmatic_literal_successive_probs_with_entropy(problem, exs, problem_index):

    # Probabilities over h_1
    p_h1_0 = [problem[0][1]['h'].loc[exs[0], 'h_1'],
            problem[0][2]['h'].loc[exs[1], 'h_1'],
            problem[0][3]['h'].loc[exs[2], 'h_1']]

    p_h1_500 = [problem[500][1]['h'].loc[exs[0], 'h_1'],
            problem[500][2]['h'].loc[exs[1], 'h_1'],
            problem[500][3]['h'].loc[exs[2], 'h_1']]

    n_ex = range(1,4)

    # Probabilities over all hypotheses
    p_h_0 = [problem[0][1]['h'].loc[exs[0]],
        problem[0][2]['h'].loc[exs[1]],
        problem[0][3]['h'].loc[exs[2]]]

    p_h_500 = [problem[500][1]['h'].loc[exs[0]],
            problem[500][2]['h'].loc[exs[1]],
            problem[500][3]['h'].loc[exs[2]]]

    # Calculate entropy
    s_0 = [entropy(p_h_0[i].to_numpy()) for i in range(len(p_h_0))]
    s_500 = [entropy(p_h_500[i].to_numpy()) for i in range(len(p_h_500))]

    plt.figure(figsize=(8, 6))

    plt.plot(n_ex, p_h1_0, 'b--', label='Literal')
    plt.plot(n_ex, p_h1_500, 'b', label='Pragmatic')

    plt.title(f'Problem {problem_index}, learner\'s belief in $h_1$')
    plt.xlabel('Examples')
    plt.ylabel('$P(h_1|d)$')
    plt.xticks(n_ex, exs)
    plt.ylim((-0.05, 1.05))
    plt.tick_params(axis='y', labelcolor='b')
    plt.legend(loc='lower left', title='$P(h_1|d)$')

    plt.twinx()

    plt.plot(n_ex, s_0, 'g--', label='Literal')
    plt.plot(n_ex, s_500, 'g', label='Pragmatic')
    plt.ylabel('Entropy')
    plt.tick_params(axis='y', labelcolor='g')
    plt.legend(loc='upper left', title='Entropy')

    #plt.tight_layout()
    plt.show()


def plot_pragmatic_literal_successive_probs(problem, exs, problem_index):

    plt.figure()

    p_h_0 = [problem[0][1]['h'].loc[exs[0], 'h_1'],
            problem[0][2]['h'].loc[exs[1], 'h_1'],
            problem[0][3]['h'].loc[exs[2], 'h_1']]

    p_h_500 = [problem[500][1]['h'].loc[exs[0], 'h_1'],
            problem[500][2]['h'].loc[exs[1], 'h_1'],
            problem[500][3]['h'].loc[exs[2], 'h_1']]

    n_ex = range(1, 4)

    plt.plot(n_ex, p_h_0, label='Literal')
    plt.plot(n_ex, p_h_500, label='Pragmatic')

    plt.title(f'Problem {problem_index}, learner\'s belief in $h_1$')
    plt.xlabel('Examples')
    plt.ylabel('$P(h_1|d)$')
    plt.xticks(n_ex, exs)
    plt.ylim((-0.05, 1.05))
    plt.legend()

    plt.show()


def make_many_plots(all_problems, problem_index):

    # Plot a few high probability examples
    problem = {}
    problem[0] = find_teacher_probs(0, problem_index, all_problems)  # 0 iterations
    problem[500] = find_teacher_probs(500, problem_index, all_problems)

    plot_high_prob_examples(problem, 500, 'd', problem_index, all_problems)

    # Plot heatmap
    iters = [0, 500]
    k_values = [1, 2, 3]
    types = {'d': '$P(d|h)$', 'h': '$P(h|d)$'}

    for k in k_values:
        for i in iters:
            for t, v in types.items():
                _ = make_and_plot_prob_heatmap(problem[i][k][t], f'{i} iterations, k={k}, {v}')

    # Find a few high quality examples and plot

    ex_seqs = []
    exs = sort_values_ascending_by_column(problem[500][3]['h'], 'h_1').head(5).index  # randomize later?

    for ex in exs:
        ex_seqs.append([(ex[0]), (ex[0], ex[1]), (ex[0], ex[1], ex[2])])
        ex_seqs.append([(ex[0]), (ex[0], ex[2]), (ex[0], ex[1], ex[2])])
        ex_seqs.append([(ex[1]), (ex[0], ex[1]), (ex[0], ex[1], ex[2])])
        ex_seqs.append([(ex[1]), (ex[1], ex[2]), (ex[0], ex[1], ex[2])])
        ex_seqs.append([(ex[2]), (ex[1], ex[2]), (ex[0], ex[1], ex[2])])
        ex_seqs.append([(ex[2]), (ex[0], ex[2]), (ex[0], ex[1], ex[2])])

    # Plot pragmatic vs. literal listener plots with entropy
    for i in range(len(ex_seqs)):
        plot_pragmatic_literal_successive_probs_with_entropy(problem, ex_seqs[i], problem_index)