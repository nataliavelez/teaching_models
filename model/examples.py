"""Find possible examples at each step"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import import_problems

filename = '/Users/aliciachen/Dropbox/teaching_models/teaching_stimuli - all_examples (9).csv'
all_problems = import_problems.df_from_csv(filename)

class Problem: 

    """Load a single problem"""

    def __init__(self, all_problems, prob_idx): 
        
        self.prob_idx = prob_idx
        self.problem_df = all_problems.loc[prob_idx, :]  # Dataframe representation of problem

        # 3d representation of problem, dimensions (4, 6, 6)
        self.problem_3d = np.array([i for i in self.problem_df.to_numpy(dtype=object)])

        # Store shape of problem: (nHypotheses, nCols, nRows)
        self.shape_ = self.problem_3d.shape

        # Store each hypothesis separately as list of lists; h1 is true hypothesis
        self.h1, self.h2, self.h3, self.h4 = [self.problem_df[i].tolist() for i in range(self.shape_[0])]
        self.hs = [self.h1, self.h2, self.h3, self.h4]

        self.k = 4  # max number of examples

    def view(self): 
        """View problem"""
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        opt_labels = ['$h_1$', '$h_2$', '$h_3$', '$h_4$']

        for idx, ax_i in enumerate(ax):
            hm = sns.heatmap(self.problem_3d[idx, :, :], ax=ax_i,
                            cmap='gray_r', cbar=False,
                            linewidths=2, linecolor='#808080')
            ax_i.set(xticks=[], yticks=[], title=opt_labels[idx])


    ### Model parts (combine later)

    def selected_examples(self, exs): 
        """
        Store size k tuple of coords of participant's selected examples
        Find and store possible examples at each step

        Output: list (each step) of list of tuples of possible coordinates
        """
        self.exs = exs 
        possible_exs = []

        # First step
        step1_exs = []
        for col in range(self.shape_[1]): 
            for row in range(self.shape_[2]): 
                if self.h1[col][row] == 1: 
                    step1_exs.append(((col, row),))
        
        possible_exs.append(step1_exs)

        # Second step 
        step2_exs = []
        for col in range(self.shape_[1]): 
            for row in range(self.shape_[2]): 
                if self.h1[col][row] == 1 and (col, row) != self.exs[0]: 
                    step2_exs.append((self.exs[0], (col, row)))

        possible_exs.append(step2_exs)

        # Third step 
        step3_exs = []
        for col in range(self.shape_[1]): 
            for row in range(self.shape_[2]): 
                if (self.h1[col][row] == 1) and ((col, row) != self.exs[0]) and ((col, row) != self.exs[1]): 
                    step3_exs.append((self.exs[0], self.exs[1], (col, row)))
        
        possible_exs.append(step3_exs)

        # Fourth step
        step4_exs = []
        for col in range(self.shape_[1]): 
            for row in range(self.shape_[2]): 
                if (self.h1[col][row] == 1) and ((col, row) != self.exs[0]) and ((col, row) != self.exs[1]) and ((col, row) != self.exs[2]): 
                    step4_exs.append((self.exs[0], self.exs[1], self.exs[2], (col, row)))
        
        possible_exs.append(step4_exs)

        self.possible_exs_by_step = possible_exs


    def literal(self): 
        """make initial df for each step (literal h|d)"""

        def isConsistent(h_i, ex):
            """Check if ex (list of tuples) is consistent with selected hypothesis h"""
            flag = True
            for e in ex: 
                if h_i[e[0]][e[1]] != 1: 
                    flag = False  
            
            return flag
        
        columns = ['h1', 'h2', 'h3', 'h4']
        
        # Initialize empty dataframes
        df_0_step1 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[0]), columns=columns)
        df_0_step2 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[1]), columns=columns)
        df_0_step3 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[2]), columns=columns)
        df_0_step4 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[3]), columns=columns)

        self.init_dfs = [df_0_step1, df_0_step2, df_0_step3, df_0_step4]

        # Fill dataframes with 1 if selected examples are consistent
        for n in range(self.k): 
            for ex in self.possible_exs_by_step[n]: 
                for idx, col_name in enumerate(columns): 
                    if isConsistent(self.hs[idx], ex): 
                        self.init_dfs[n].loc[ex, col_name] = 1
                    else: 
                        self.init_dfs[n].loc[ex, col_name] = 0

        def normalize_cols(df): 
            df = df.div(df.sum(axis=1).replace(0, np.nan), axis=0)
            df = df.fillna(0)
            return df 

        def normalize_rows(df): 
            df = df.div(df.sum(axis=0).replace(0, np.nan), axis=1)
            df = df.fillna(0)
            return df 

        # TODO: take this out of self? 
        self.hGd_0_step1 = normalize_cols(df_0_step1)
        self.hGd_0_step2 = normalize_cols(df_0_step2)
        self.hGd_0_step3 = normalize_cols(df_0_step3)
        self.hGd_0_step4 = normalize_cols(df_0_step4)

        # Generate P(h|d) for literal learner
        self.hGd_lit = [normalize_cols(df_0_step1), normalize_cols(df_0_step2), normalize_cols(df_0_step3), normalize_cols(df_0_step4)]
        self.dGh_lit = [normalize_rows(df_0_step1), normalize_rows(df_0_step2), normalize_rows(df_0_step3), normalize_rows(df_0_step4)]

        return self.hGd_lit, self.dGh_lit
    
    def pragmatic(self, nIter): 
        """return pragmatic h|d and d|h"""
        hGd_prag = []
        dGh_prag = []

        for h in self.hGd_lit: 
            df_h = h 

            for _ in range(nIter):
                df_d = df_h.div(df_h.sum(axis=0).replace(0, np.nan), axis=1)  # P(d|h)
                df_h = df_d.div(df_d.sum(axis=1).replace(0, np.nan), axis=0)  # P(h|d)

            hGd_prag.append(df_h.fillna(0))
            dGh_prag.append(df_d.fillna(0))

        self.hGd_prag = hGd_prag
        self.dGh_prag = dGh_prag

        return self.hGd_prag, self.dGh_prag

    
# Testing
testprob = Problem(all_problems, 3)
testprob.view()
testprob.selected_examples([(1, 1), (2, 1), (3, 1), (4, 1)])
_, _ = testprob.literal()
_, _ = testprob.pragmatic(250)

# TODO: loop through all pilot problems and add model predictions to each 
# TODO: add log likelihoods to analysis dataframe
# TODO: add exception for when selected examples aren't positive examples? 