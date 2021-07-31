"""
0. Compare participants' confidence ratings against model p(h|d) for literal and pragmatic
1. Calculate log likelihood for each participant's total + individual choices, for literal and pragmatic learner
"""
import numpy as np
import pandas as pd
# from seaborn.miscplot import palplot
# from seaborn.palettes import color_palette
# import examples
import examples_full
import import_problems

# %% Load and prepare data

filename = '/Users/aliciachen/Dropbox/teaching_models/teaching_stimuli - all_examples (9).csv'
all_problems = import_problems.df_from_csv(filename)

def load(expt1_filename, expt2_filename, true_prob_idxs): 
    """
    Load and prepare data
    true_prob_idxs is list of indices of problems in original csv 
    """

    df_exp1 = pd.read_csv(expt1_filename)
    df_exp2 = pd.read_csv(expt2_filename)
    
    dfs = [df_exp1, df_exp2]

    for df in dfs: 
        
        # Change to zero indexing
        cols_to_change = df.columns[1:5]
        for c in cols_to_change: 
            df[c] = df[c] - 1

        # Add column with coords 
        df['coords'] = list(zip(df['row'], df['col']))

        # Get rid of row and col 
        df.drop(labels=['row', 'col'], axis=1, inplace=True)

        # Add true problem index 
        df['true_prob_idx'] = df['problem'].apply(lambda x : true_prob_idxs[x])

        # Group df by worker and problem tuple
        df = df.set_index(['worker', 'true_prob_idx'], inplace=True) #['coords_idx'].apply(tuple).reset_index()

    return df_exp1, df_exp2


true_prob_idxs = [43, 47, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65] + [i for i in range(66, 94)]  # Later: maybe add a dict of idx mappings
expt1_filename = "/Users/aliciachen/Dropbox/teaching_models/data/exp1_varex_data.csv"
expt2_filename = "/Users/aliciachen/Dropbox/teaching_models/data/exp2_fixedex_data.csv"

df_expt1, df_expt2 = load(expt1_filename, expt2_filename, true_prob_idxs)

# %% Generate model predictions 

def gen_example_sequence(df): 
    """
    Given df, generate sequence of coords for each unique worker problem pair
    Output df is sorted by ascending problem index
    """
    grouped = df.groupby(level=[0, 1])['coords'].apply(tuple).reset_index().set_index(['worker', 'true_prob_idx'])
    return grouped 

def gen_model_preds(true_prob_idxs): 
    preds = {}
    for idx in true_prob_idxs: 
        print('Generating model predictions for problem ' + str(idx))
        preds[idx] = examples_full.Problem(all_problems, idx)
        preds[idx].runModel(nIter=50)
    return preds

def predict(preds): 
    """
    Loop over problem worker pairs 
    Make a list of model predictions for each problem worker pair for both lit and prag 
    Add model predictions + metrics to dataframe
    """
    df_expt1_grouped = gen_example_sequence(df_expt1)
    df_expt2_grouped = gen_example_sequence(df_expt2)
    dfs_grouped = [df_expt1_grouped, df_expt2_grouped]

    preds = preds

    for df in dfs_grouped: 
        
        # Add empty columns to df
        newcolnames = ['h1Gd_lit', 'H_lit', 'KL_lit', 'dGh1_lit', 
                        'h1Gd_prag', 'H_prag', 'KL_prag', 'dGh1_prag']
        
        for i in newcolnames: 
            df[i] = pd.Series("", dtype=object)
            # df = df_expt2_grouped
        
        for row in df.index: 
            try: 
                idx = row[1]  # True problem index 
                coords = df.loc[row, 'coords']
                
                # Select examples and generate model outputs
                preds[idx].selected_examples(coords)
                _ = preds[idx].outputs() 

                print(row)

                df.at[row, 'h1Gd_lit'] = preds[idx].h1Gd_lit
                df.at[row, 'dGh1_lit'] = preds[idx].dGh1_lit
                df.at[row, 'h1Gd_prag'] = preds[idx].h1Gd_prag
                df.at[row, 'dGh1_prag'] = preds[idx].dGh1_prag
                df.at[row, 'H_lit'] = preds[idx].H_lit
                df.at[row, 'H_prag'] = preds[idx].H_prag
                df.at[row, 'KL_lit'] = preds[idx].KL_lit
                df.at[row, 'KL_prag'] = preds[idx].KL_prag

            # Return semi populated df if we stop early
            except KeyboardInterrupt: 
                return dfs_grouped

    return dfs_grouped

preds = gen_model_preds(true_prob_idxs)
df_expt1_final, df_expt2_final = predict(preds)

# %% df management

# add participant estimated posterior to df_expt2_final
df_expt2_estimates = df_expt2.groupby(level=[0, 1])['posterior'].apply(list).reset_index().set_index(['worker', 'true_prob_idx'])
df_expt2_final_with_estimates = pd.concat([df_expt2_final, df_expt2_estimates], axis=1)

df_1 = df_expt1_final
df_2 = df_expt2_final_with_estimates


# Explode dataframe
df_1_long = df_1.explode(column=['coords', 'h1Gd_lit', 'H_lit', 'KL_lit', 'dGh1_lit', 
                        'h1Gd_prag', 'H_prag', 'KL_prag', 'dGh1_prag'])

df_2_long = df_2.explode(column=['coords', 'h1Gd_lit', 'H_lit', 'KL_lit', 'dGh1_lit', 
        'h1Gd_prag', 'H_prag', 'KL_prag', 'dGh1_prag', 'posterior'])

df_1 = df_1_long.reset_index()
df_2 = df_2_long.reset_index()


# Change object dtypes to numerics 
numeric_cols1 = ['h1Gd_lit', 'H_lit', 'KL_lit', 'dGh1_lit', 'h1Gd_prag', 
'H_prag', 'KL_prag', 'dGh1_prag']
numeric_cols2 = ['h1Gd_lit', 'H_lit', 'KL_lit', 'dGh1_lit', 'h1Gd_prag', 
'H_prag', 'KL_prag', 'dGh1_prag', 'posterior']

df_1[numeric_cols1] = df_1[numeric_cols1].apply(pd.to_numeric)
df_2[numeric_cols2] = df_2[numeric_cols2].apply(pd.to_numeric)


# Add log like
df_1['log dGh1_lit'] = df_1['dGh1_lit'].apply(np.log)
df_1['log dGh1_prag'] = df_1['dGh1_prag'].apply(np.log)

df_2['log dGh1_lit'] = df_2['dGh1_lit'].apply(np.log)
df_2['log dGh1_prag'] = df_2['dGh1_prag'].apply(np.log)

#%% Save

df_1.to_pickle("./df_expt1.pkl")
df_2.to_pickle("./df_expt2.pkl")

df_1.to_csv("df_expt1_update_250iter.csv")
df_2.to_csv("df_expt2_update_250iter.csv")
