#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from rectangle_model import *

from make_df import *

filename = 'teaching_stimuli - all_examples (7).csv'
all_problems = make_df_from_spreadsheet(filename)

#%% example (testing with only three examples for now)

prob_65 = find_teacher_probs(500, 63, all_problems)
# first two parameters are levels of recursion and problem index

frames = [prob_65[1]['h'].rename(index=lambda a: (a,)), prob_65[2]['h'], prob_65[3]['h']]

# Create a new df where rows are *all* possible examples, and column is p(h_1 | d)

df_65 = pd.concat(frames)
df_65.drop(columns=['h_2', 'h_3', 'h_4'], inplace=True)
df_65 = df_65[df_65.h_1 != 0] # only positive examples
#%%

def utility(alpha, beta, df_65):
#alpha = 1
#beta = 1

    def softmax(a):
        result = np.exp(a) / np.sum(np.exp(a))
        return result

    # add cost
    df_65['c'] = df_65.index.str.len()

    # Add utility
    df_65['u'] = np.log(df_65['h_1']) - beta*df_65['c']
    df_65[df_65 == -1*np.inf] = 0

    df_65['prob_d'] = softmax(alpha * df_65['u'])
    return df_65

df_65.sort_values(by=['prob_d'])

#%% Plot a bunch of these examples
import seaborn as sns
#sns.set()

#%%

_ = sns.scatterplot(data=df_65, x='h_1', y='prob_d', hue='c')
# Expected utility is higher for more examples, because it tells you more about the true h
# In this case, all the other examples include thte true hypothesis

#%%

plot_problem(find_problem(65, all_problems)[0])
plot_problem(find_problem(63, all_problems)[0])
plot_problem(find_problem(61, all_problems)[0])
#%% Include 4 examples?

_ = sns.scatterplot(data=df_65, x='c', y='prob_d', hue='c')

#%%

# Need to include 4 examples to have a good viewpoint of utility here
# Should include 4 examples and also 5 examples tto see how the utility changes

_, df4 = find_teacher_probs_k4(500, 65, all_problems)

#%% Add 4 examples

frames = [prob_65[1]['h'].rename(index=lambda a: (a,)), prob_65[2]['h'], prob_65[3]['h'], df4]

# Create a new df where rows are *all* possible examples, and column is p(h_1 | d)

df_65 = pd.concat(frames)
df_65.drop(columns=['h_2', 'h_3', 'h_4'], inplace=True)
df_65 = df_65[df_65.h_1 != 0] # only positive examples

#%%

new_df_65 = utility(.5, 2, df_65)

plt.figure()
plt.scatter(df_65['c'], df_65['prob_d'])
#_ = sns.scatterplot(data=df_65, x='c', y='prob_d', hue='c')
plt.xticks([1, 2, 3, 4])
#plt.yticks([0.000390, 0.00041, 0.000430])
plt.xlabel('Number of examples given')
plt.ylabel('Probability')
plt.title('Problem 63, $p(d|h_1)$, probability of individual sets of examples')

#%%
# sum up all the probs of providing 2 examples, 3 examples, 4 examples

def plot_utterance_probs(alpha, beta, df_65):

    new_df_65 = utility(alpha, beta, df_65)

    sums = [
            new_df_65.loc[[len(i) == 1 for i in new_df_65.index]]['prob_d'].sum(),
            new_df_65.loc[[len(i) == 2 for i in new_df_65.index]]['prob_d'].sum(),
            new_df_65.loc[[len(i) == 3 for i in new_df_65.index]]['prob_d'].sum(),
            new_df_65.loc[[len(i) == 4 for i in new_df_65.index]]['prob_d'].sum()
            ]


    plt.figure()
    plt.scatter([1, 2, 3, 4], sums)
    plt.xticks([1, 2, 3, 4])
    plt.xlabel('Number of examples given')
    plt.ylabel('Probability')
    plt.ylim([0, 1])
    plt.title(rf'Problem 63, probability of utterance lengths, $\alpha={alpha}$, $\beta={beta}$')

#%%

plot_utterance_probs(1, 1, df_65)
plot_utterance_probs(.5, 1, df_65)
plot_utterance_probs(.8, 2, df_65)
plot_utterance_probs(.5, 2, df_65)
plot_utterance_probs(.8, 3, df_65)
plot_utterance_probs(.5, 3, df_65)
plot_utterance_probs(.8, 2.5, df_65)

plot_utterance_probs(10, .5, df_65)
plot_utterance_probs(10, .8, df_65)

#%%