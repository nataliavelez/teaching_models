"""Validate new implementation of model; compare with fall 2020 implementation"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import examples_full

df_1 = pd.read_pickle("./df_expt1.pkl")
df_2 = pd.read_pickle("./df_expt2.pkl")

# %% 

# Create dict of test probs with (flattened) example coords
test_exs = {
    68: [(13, ), (13, 19), (13, 19, 26)], 
    70: [(27, ), (10, 27), (8, 10, 27)], 
    71: [(3, ), (3, 17), (3, 17, 25)], 
    73: [(0, ), (0, 13), (0, 10, 13)]
}

# Create a new dict w/ each flat idx as a tuple of coords
def flat_idx_to_tuple(idx): 
    """Convert 0 to 35 index to tuple of coords"""
    row = idx // 6
    col = idx % 6
    return (row, col)

test_exs_new = {}

for k, v in test_exs.items(): 
    for i, tup in enumerate(v): 
        for coord in tup: 
            coord_new = flat_idx_to_tuple(coord)
            test_exs[k][i] = coord_new

print(test_exs)
# %%
