"""
0. Compare participants' confidence ratings against model p(h|d) for literal and pragmatic
1. Calculate log likelihood for each participant's total + individual choices, for literal and pragmatic learner
"""
import pandas as pd
import examples
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

# %% 

def gen_example_sequence(df): 
    """
    Given df, generate sequence of coords for each unique worker problem pair
    Output df is sorted by ascending problem index
    """
    grouped = df.groupby(level=[0, 1])['coords'].apply(tuple).reset_index().set_index(['worker', 'true_prob_idx'])
    return grouped 

def predict(true_prob_idxs): 
    """
    Loop over problem worker pairs 
    Make a list of model predictions for each problem worker pair for both lit and prag 
    Add model predictions to 
    """
    df_expt1_grouped = gen_example_sequence(df_expt1)
    df_expt2_grouped = gen_example_sequence(df_expt2)
    dfs_grouped = [df_expt1_grouped, df_expt2_grouped]

    for df in dfs_grouped: 
        
        # Add empty columns to df
        newcolnames = ['h1Gd_lit', 'dGh1_lit', 'h1Gd_prag', 'dGh1_prag']
        for i in newcolnames: 
            df[i] = pd.Series("", dtype=object)
            # df = df_expt2_grouped
        
        for row in df.index: 
            try: 
                idx = row[1]  # True problem index 
                coords = df.loc[row, 'coords']
                prob = examples.Problem(all_problems, idx)
                prob.selected_examples(coords)
                
                _, _ = prob.literal()
                _, _ = prob.pragmatic(250)
                a, b, c, d = prob.h1_probs() 

                # Fix later: 
                # hGd_lit, dGh_lit = prob.literal()
                # hGd_prag, dGh_prag = prob.pragmatic()
                
                # [df[i] for i in newcolnames] = [pd.Series("", dtype=object) for i in range(4)]

                # print(df.dtypes)
                print(row)
                print(prob.h1Gd_lit)
                df.at[row, 'h1Gd_lit'] = prob.h1Gd_lit
                df.at[row, 'dGh1_lit'] = prob.dGh1_lit
                df.at[row, 'h1Gd_prag'] = prob.h1Gd_prag
                df.at[row, 'dGh1_prag'] = prob.dGh1_prag
                # OMG IT WORKED

            except KeyboardInterrupt: 
                return df

    return dfs_grouped

test = predict(true_prob_idxs)

# TODO: for each worker problem pair, make a list of tuples of coords (selected examples) that we put in example 

# TODO: THen filter these examples from p(h|d) and p(d|h), add this to the data frame

# testprob = Problem(all_problems, 3)
# testprob.view()
# testprob.selected_examples([(1, 1), (2, 1), (3, 1), (4, 1)])
# _, _ = testprob.literal()
# _, _ = testprob.pragmatic(250)

# %% 

def extract_confidence(data):
    """Extract list of lists of each participant's confidence ratings""" 
    pass

def extract_examples(data): 
    """Extract list of tuples of examples selected for each participant"""
    pass 

class Analyze(Problem): 

    def heatmap(self): 
        # TODO 
        pass
    
    def plots(self): 
        # TODO: conditioned h\d plots 
        pass

    def metrics(self): 

        # TODO: h|d entropy at each step 
        # TODO: belief update of h|D at each step (KL divergence)
        pass

    def loglike(self): 

        # TODO: from selected_examples, calculate log likelihood of specific set of examples for participant
        # maybe add this back to selected_examples
        pass

    def hGd_choices(self): 

        # TODO: extract p(h|d) for all the choices the participant selects in a list, for literal and pragmatic 
        # combine this with previous stuff 
        # maybe add this to analysis script...? 

        # TODO: make tuple where first value is output of hGd_choices and second value is confidence .. jk do this later

        self.hGd_lit 
        self.hGd_prag

        # Variables to store
        self.hGd_lit_vs_conf = None
        self.hGd_prag_vs_conf = None
        pass

    def confidence_plots(self, exs, conf): 
        # TODO: literal plot of confidence with output of hGd_choiecs (aggregated for all participants)
        # TODO: pragmatic plot 

        pass

# TODO: function that loops through all the participants, makes plots 


#%% 


