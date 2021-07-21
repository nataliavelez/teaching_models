"""
0. Compare participants' confidence ratings against model p(h|d) for literal and pragmatic
1. Calculate log likelihood for each participant's total + individual choices, for literal and pragmatic learner
"""
import pandas as pd
from examples import Problem

# %% Load and prepare data

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

    return df_exp1, df_exp2


true_prob_idxs = [43, 47, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65] + [i for i in range(66, 94)]  # Later: maybe add a dict of idx mappings
expt1_filename = "/Users/aliciachen/Dropbox/teaching_models/data/exp1_varex_data.csv"
expt2_filename = "/Users/aliciachen/Dropbox/teaching_models/data/exp2_fixedex_data.csv"

df_expt1, df_expt2 = load(expt1_filename, expt2_filename, true_prob_idxs)

# %% 

def predict(true_prob_idxs): 
    """Make a dict of model predictions for each problem"""


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


