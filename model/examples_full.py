"""Model for full example space"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import import_problems

filename = '/Users/aliciachen/Dropbox/teaching_models/teaching_stimuli - all_examples (9).csv'
all_problems = import_problems.df_from_csv(filename)

class Problem: 

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
        """Store selected examples for outputs method"""
        self.exs = exs
        self.exs_length = len(exs)

    def example_space(self): 

        """
        Generate example space for all k up to # exs selected
        """
        # # Generate all possible combos of coords
        # full_exs = []

        # if self.k >= 1: 
        #     exsk1 = []
        #     for col in range(self.shape_[1]): 
        #         for row in range(self.shape_[2]): 
        #             exsk1.append(((col, row), ))

        #     full_exs.append(exsk1)


        # # self.exs = exs

        def flat_idx_to_tuple(idx): 
            """Convert 0 to 35 index to tuple of coords"""
            row = (idx) // 6
            col = (idx) % 6
            return (row, col)

        if self.k >= 1: 
            step1_exs = [(flat_idx_to_tuple(i), ) for i in range(36)]

        if self.k >= 2: 
            step2_exs = []
            for i in range(36): 
                for j in range(i+1, 36): 
                    step2_exs.append((flat_idx_to_tuple(i), flat_idx_to_tuple(j)))

        if self.k >= 3: 
            step3_exs = []
            for i in range(36): 
                for j in range(i+1, 36): 
                    for k in range(j+1, 36): 
                        step3_exs.append((flat_idx_to_tuple(i), flat_idx_to_tuple(j), flat_idx_to_tuple(k)))

        if self.k >= 4: 
            step4_exs = []
            for i in range(36):
                for j in range(i+1, 36):
                    for k in range(j+1, 36):
                        for l in range(k+1, 36):
                            step4_exs.append((flat_idx_to_tuple(i), flat_idx_to_tuple(j), flat_idx_to_tuple(k), flat_idx_to_tuple(l)))

        possible_exs = [step1_exs, step2_exs, step3_exs, step4_exs]

        # if self.k >= 1: 

        #     step1_exs = []
        #     for col in range(self.shape_[1]): 
        #         for row in range(self.shape_[2]): 
        #             step1_exs.append(((col, row),))
                
        #     possible_exs.append(step1_exs)

        
        # if self.k >= 2: 

        #     step2_exs = []
        #     for ex in step1_exs: 
        #         for col in range(self.shape_[1]): 
        #             for row in range(self.shape_[2]): 
        #                 if (col, row) != ex[0]: 
        #                     newex = (ex[0], (col, row))
        #                     if set(newex) not in [set(i) for i in step2_exs]:
        #                         step2_exs.append(newex)

        #                     # Maybe later: add test that makes sure # examples is size of h1 choose 2, etc. 

        #     possible_exs.append(step2_exs)


        # if self.k >= 3: 
            
        #     step3_exs = []
        #     for ex in step2_exs: 
        #         for col in range(self.shape_[1]): 
        #             for row in range(self.shape_[2]): 
        #                 if (col, row) != ex[0] and (col, row) != ex[1]:
        #                     newex = (ex[0], ex[1], (col, row))
        #                     if set(newex) not in [set(i) for i in step3_exs]:
        #                         step3_exs.append(newex)

        #     possible_exs.append(step3_exs)


        # if self.k >= 4: 
            
        #     step4_exs = []
        #     for ex in step3_exs: 
        #         for col in range(self.shape_[1]): 
        #             for row in range(self.shape_[2]): 
        #                 if (col, row) != ex[0] and (col, row) != ex[1] and (col, row) != ex[2]:
        #                     newex = (ex[0], ex[1], ex[2], (col, row))
        #                     if set(newex) not in [set(i) for i in step4_exs]:
        #                         step4_exs.append(newex)

        #     possible_exs.append(step4_exs)

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
        if self.k >= 1: 
            df_0_step1 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[0]), columns=columns)
            self.init_dfs.append(df_0_step1)
        if self.k >= 2: 
            df_0_step2 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[1]), columns=columns)
            self.init_dfs.append(df_0_step2)
        if self.k >= 3: 
            df_0_step3 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[2]), columns=columns)
            self.init_dfs.append(df_0_step3)
        if self.k >= 4: 
            df_0_step4 = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.possible_exs_by_step[3]), columns=columns)
            self.init_dfs.append(df_0_step4)
        

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


        #Edited
        self.hGd_lit = [normalize_cols(df) for df in self.init_dfs]
        self.dGh_lit = [normalize_rows(df) for df in self.init_dfs]

        # TODO: conditioning for dGh to filter out only possible selected examples, and then normalize
        # DO this same thing for pragmatic 

        return self.hGd_lit, self.dGh_lit
        
    def pragmatic(self, nIter): 
        """return pragmatic h|d and d|h after nIter iterations"""
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

    def runModel(self, nIter): 
        self.example_space()
        self.literal()
        self.pragmatic(nIter)
        return None

    def heatmaps(self): 

        """Make heatmaps for k=2 examples
        Later: k=4 examples?"""

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

        df = self.hGd_prag
        probs = make_prob_heatmap_k2(df)
        _ = plot_problem_heatmap(probs, f'k=2 heatmap, problem {self.prob_idx}')
        pass

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