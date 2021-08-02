"""Validate new implementation of model; compare with fall 2020 implementation"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import examples_full
import dill as pickle

df_1 = pd.read_pickle("./df_expt1.pkl")
df_2 = pd.read_pickle("./df_expt2.pkl")

# %% Make test examples

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
    
    new_coords = []
    
    for i, tup in enumerate(v): 
        new_tup = ()
        for coord in tup: 
            coord_new = flat_idx_to_tuple(coord)
            new_tup += (coord_new, )
        new_coords.append(new_tup)
    
    test_exs[k] = new_coords

print(test_exs)

# # %% Load model preds FIX LATER

# def pickle_loader(filename):
#     """Deserialize a file of pickled objects. 
#     https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence/4529901"""
#     with open(filename, "rb") as f:
#         while True:
#             try:
#                 yield pickle.load(f)
#             except EOFError:
#                 break

# preds = []
# for pred in pickle_loader(filename="./model_preds.pkl"): 
#     preds.append(pred)

# with open("./model_preds.pkl", 'rb') as inp: 
#     preds = pickle.load(inp)


# f = open("./model_preds.pkl", 'rb')
# preds = pickle.load(f)



# %%
