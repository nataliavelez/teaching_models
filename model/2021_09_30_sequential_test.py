# Implementation of 2x2 example in 
# sequential cooperate bayesian inference (Wang et al. 2020)

import numpy as np

M = np.array([
    [0.3, 0.3], 
    [0.1, 0.3]
])
n = 2
m = 2

def normalize_cols(M, r): 
    return (M / M.sum(axis=0))*r

def normalize_rows(M, c): 
    return (M / (M.sum(axis=1)[:, np.newaxis]))*c


# Round 1

r = np.array([[1, 1]])
c = np.array([[1, 1]])
Mprimeprime = M.copy()

for i in range(500): 
    Mprime = normalize_cols(Mprimeprime, r)
    Mprimeprime = normalize_rows(Mprime, c)

print(Mprimeprime)

# Round 2

c2 = n * Mprimeprime[0:1, :]
r2 = np.array([[1, 1]])

for i in range(500): 
    Mprime = normalize_cols(Mprimeprime, c2)
    Mprimeprime = normalize_rows(Mprime, r2)

