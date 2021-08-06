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
        preds[idx].runModel(nIter=30)
    return preds

preds = gen_model_preds(true_prob_idxs)

# %% test heatmap code

import matplotlib.pyplot as plt
import seaborn as sns

def make_prob_heatmap_k2(df):
    probs = np.array([np.zeros((6, 6)) for column in df.columns])

    for h_idx in range(df.columns.size):
        for coords in df.index:
            probs[h_idx, coords[0][0], coords[0][1]] += df.xs(coords)[df.columns[h_idx]]
            probs[h_idx, coords[1][0], coords[1][1]] += df.xs(coords)[df.columns[h_idx]]

    # correct for double counting
    # probs = probs / 2
    return probs

def plot_problem_heatmap(probs, title):
    '''Plot probabilities heatmap given probabilities'''
    problem = probs
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
    
    return None

df = preds[43].hGd_prag[1]
probs = make_prob_heatmap_k2(df)
_ = plot_problem_heatmap(probs, f'k=2 heatmap, problem {preds[43].prob_idx}')

# %% Make  heatmaps


for i in true_prob_idxs: 
    df = preds[i].dGh_prag[1]
    probs = make_prob_heatmap_k2(df)
    _ = plot_problem_heatmap(probs, f'k=2 heatmap, problem {preds[i].prob_idx}')
# %%
for i in true_prob_idxs: 
    df = preds[i].dGh_lit[1]
    probs = make_prob_heatmap_k2(df)
    _ = plot_problem_heatmap(probs, f'k=2 heatmap, problem {preds[i].prob_idx}')
# %%
