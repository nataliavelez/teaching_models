"""Model for full example space"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import import_problems

filename = '/Users/aliciachen/Dropbox/teaching_models/teaching_stimuli - all_examples (9).csv'
all_problems = import_problems.df_from_csv(filename)

class ProblemFull: 

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

    
    def selected_examples(self, exs): 

        """
        Generate example space for all k up to # exs selected
        """

        self.exs = exs
        self.exs_length = len(exs)
        possible_exs = []

        if self.exs_length >= 1: 

            step1_exs = []
            for col in range(self.shape_[1]): 
                for row in range(self.shape_[2]): 
                    if self.h1[col][row] == 1: 
                        step1_exs.append(((col, row),))
                
            possible_exs.append(step1_exs)

        
        if self.exs_length >= 2: 

            step2_exs = []
            for ex in step1_exs: 
                for col in range(self.shape_[1]): 
                    for row in range(self.shape_[2]): 
                        if self.h1[col][row] == 1 and (col, row) != ex[0]: # and ((col, row), ex[0]) not in step2_exs: 
                            newex = (ex[0], (col, row))
                            if set(newex) not in [set(i) for i in step2_exs]:
                                step2_exs.append(newex)

                            # Maybe later: add test that # examples is size of trueH choose 2

            possible_exs.append(step2_exs)


        if self.exs_length >= 3: 
            
            step3_exs = []
            for ex in step2_exs: 
                for col in range(self.shape_[1]): 
                    for row in range(self.shape_[2]): 
                        if self.h1[col][row] == 1 and (col, row) != ex[0] and (col, row) != ex[1]:
                            newex = (ex[0], ex[1], (col, row))
                            if set(newex) not in [set(i) for i in step3_exs]:
                                step3_exs.append(newex)

            possible_exs.append(step3_exs)


        if self.exs_length >= 3: 
            
            step4_exs = []
            for ex in step3_exs: 
                for col in range(self.shape_[1]): 
                    for row in range(self.shape_[2]): 
                        if self.h1[col][row] == 1 and (col, row) != ex[0] and (col, row) != ex[1] and (col, row) != ex[2]:
                            newex = (ex[0], ex[1], ex[2], (col, row))
                            if set(newex) not in [set(i) for i in step4_exs]:
                                step4_exs.append(newex)

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

        self.init_dfs = []
        if self.exs_length >= 1: 
            df_0_step1 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[0]), columns=columns)
            self.init_dfs.append(df_0_step1)
        if self.exs_length >= 2: 
            df_0_step2 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[1]), columns=columns)
            self.init_dfs.append(df_0_step2)
        if self.exs_length >= 3: 
            df_0_step3 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[2]), columns=columns)
            self.init_dfs.append(df_0_step3)
        if self.exs_length >= 4: 
            df_0_step4 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[3]), columns=columns)
            self.init_dfs.append(df_0_step4)
        
        # self.init_dfs = [df_0_step1, df_0_step2, df_0_step3, df_0_step4]

        # Fill dataframes with 1 if selected examples are consistent
        for n in range(self.exs_length): 
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

        self.hGd_lit = [normalize_cols(df) for df in self.init_dfs]
        self.dGh_lit = [normalize_rows(df) for df in self.init_dfs]

        # TODO: take this out of self? 
        # self.hGd_0_step1 = normalize_cols(df_0_step1)
        # self.hGd_0_step2 = normalize_cols(df_0_step2)
        # self.hGd_0_step3 = normalize_cols(df_0_step3)
        # self.hGd_0_step4 = normalize_cols(df_0_step4)


        # Generate P(h|d) for literal learner
        # self.hGd_lit = [normalize_cols(df_0_step1), normalize_cols(df_0_step2), normalize_cols(df_0_step3), normalize_cols(df_0_step4)]
        # self.dGh_lit = [normalize_rows(df_0_step1), normalize_rows(df_0_step2), normalize_rows(df_0_step3), normalize_rows(df_0_step4)]

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

    def outputs(self): 
            """
            Generate model outputs (for literal and pragmatic model) to append to dataframe: 
            - P(h|d) entropy
            - P(h|d) KL divergence between sequential example selection
            - Beliefs in true hypothesis h1 P(h1|d)
            - Likelihoods P(d|h1)
            """
            def extract_probs_from_exs(dfs): 
                probs = []  # Belief in h1
                all_probs = []  # Full belief distribution
                #print(dfs)
                for i, ex in enumerate(self.exs): 
                    idx = sorted(self.exs[:i+1])  # Check indexing later
                    # print(dfs[i])
                    # print(idx)
                    # self.a = dfs[i]
                    probs.append(dfs[i].xs(idx)['h1'])  # What is this issue with tuple indexing? 
                    all_probs.append(dfs[i].xs(idx))  # Append full belief distrubtion 
                return probs, all_probs  # test this later

            def calc_entropy(df): 
                """Given list of lists of full belief dist, create entropy list of size self.length_exs"""
                
                _, all_probs = extract_probs_from_exs(df)
                entropy_list = []
                for prob in all_probs: 
                    entropy_list.append(entropy(prob))
                
                return entropy_list
            
            def calc_KL(df): 
                """Given list of lists full belief dist, create D_KL list of size self.length_exs (keep in mind uniform priors)"""
                _, all_probs = extract_probs_from_exs(df)

                prior = [[.25, .25, .25, .25]]
                all_probs_w_prior = prior + all_probs
                # print(all_probs_w_prior)
                KL_list = []
                for i in range(len(all_probs_w_prior)-1): 
                    KL_list.append(entropy(all_probs_w_prior[i+1], all_probs_w_prior[i]))
                
                return KL_list

            # Store P(h1|d) and P(d|h1)
            model_outputs = [self.hGd_lit, self.dGh_lit, self.hGd_prag, self.dGh_prag]
            probs_list = []      
            for output in model_outputs: 
                probs, all_probs = extract_probs_from_exs(output)
                # print(probs)
                probs_list.append(probs)
            
            [self.h1Gd_lit, self.dGh1_lit, self.h1Gd_prag, self.dGh1_prag] = probs_list 

            # store entropy and KL divergence
            self.H_lit = calc_entropy(self.hGd_lit)
            self.H_prag = calc_entropy(self.hGd_prag)
            self.KL_lit = calc_KL(self.hGd_lit)
            self.KL_prag = calc_KL(self.hGd_prag)
            
            return None
# Testing

testprob = ProblemFull(all_problems, 3)
testprob.view()
testprob.selected_examples(((1, 1), (2, 1), (3, 1), (4, 1)))
# print(len(testprob.possible_exs_by_step[3]))

_, _ = testprob.literal()
_, _ = testprob.pragmatic(200)


_ = testprob.outputs()