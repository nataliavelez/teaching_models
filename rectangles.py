#!/usr/bin/env python
# coding: utf-8

import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cityblock


def make_grid(n,*args):
    '''
    Define n x n grid space with rectangles
    
    Inputs:
    n: grid dimensions
    *args: list of rectangle dimensions in indices (x0,x,y0,y)
    
    Output:
    m: n x n array containing rectangles
    '''
    
    m = np.zeros((n,n)) # Universe  
    for rect in args:
        x0,x,y0,y = rect
        m[y0:y, x0:x] = 1
    return m

def plot_grid(m):
    '''
    Plot grid
    
    light = does not include concept
    dark = includes concept
    gray = not yet revealed (learner's view)
    '''
    
    m_bool = m.astype(bool)
    if np.ma.is_masked(m):
        mask = m.mask
    else:
        mask = None
    
    ax = sns.heatmap(m_bool, linewidths=2, linecolor = 'white',
                     square=True, cbar=False, mask=mask,
                     cmap = sns.light_palette('#82bdba', as_cmap=True),
                     vmin = 0, vmax = 1)
    ax.set_facecolor('#777777')
    ax.tick_params(axis='both', which='both', length=0)
    ax.xaxis.tick_top()
    return ax

def generate_evidence(m, *args):
    '''
    Generates a masked hypothesis space with revealed examples.
    
    Input:
    m - true hypothesis (h x w grid w/ rectangles)
    *args - (r,c) tuples containing coordinates of revealed examples
    
    Output:
    m_masked - masked version of m, where points in the grid that haven't been revealed by example are masked
    '''
    
    is_revealed = np.zeros(m.shape)
    for x,y in args:
        is_revealed[x,y] = 1
    revealed = np.ma.masked_less(is_revealed, 1)
    m_masked = np.ma.masked_array(m, revealed.mask)
    return m_masked

def evidence_pairs(m, *args):
    '''
    Generates all possible pairs of examples
    Input:
    m - hypothesis (h x w grid w/ rectangles)
    
    Output:
    evidence: (?,h,w) array of paired examples
    '''
    
    evidence = []
    
    # How many points are there in the grid?
    n = m.shape[0]
    n_elem = n*n
    
    for d1 in range(n_elem):
        for d2 in range(d1+1, n_elem):
            coords = [np.unravel_index(d, m.shape) for d in [d1, d2]]
            e = generate_evidence(m, *coords)
            evidence.append(e)
            
    return np.ma.stack(evidence, axis=0)

def evidence_sequence(m, *args):
    '''
    Generates all possible next examples in a sequence.
    Input:
    m - hypothesis (h x w grid w/ rectangles)
    *args - (r,c) tuples containing coordinates of previous examples
    
    Output:
    evidence: (?,h,w) array of possible examples
    '''
    
    # How many points are there in the grid?
    n = m.shape[0]
    n_elem = n*n
    
    # Which points have already been revealed in previous examples?
    revealed_coords = list(args)
    revealed_idx = [np.ravel_multi_index(c, m.shape) for c in args]
    
    evidence = []
    for idx in range(n_elem):
        if idx not in revealed_idx:
            new_coords = revealed_coords + [np.unravel_index(idx, m.shape)]
            e = generate_evidence(m, *new_coords)
            evidence.append(e)
            
    return np.ma.stack(evidence, axis=0)

def k1_hypothesis_space(n):
    '''
    Returns all possible rectangles within an n x n grid
    Constraints:
    k = 1 rectangle
    min rectangle dimensions: (2,2)
    max rectangle dimensions: (n-1, n-1)
    '''
    
    hypotheses = []
    
    for x0 in range(n):
        for y0 in range(n):
            for x in range(x0+2, n+1):
                for y in range(y0+2, n+1):
                    m = make_grid(n, (x0,x,y0,y))
                    hypotheses.append(m)

    return np.array(hypotheses)

def index_mtx(m):
    '''
    Creates a matrix of dimensions h.shape, where each element i,j is the flat index of coordinates (i,j)
    Used to make masks
    
    e.g., 
    input: a 3x3 grid
    output:
    [0 1 2
     3 4 5
     6 7 8]
    '''
    
    idx_m = np.zeros(m.shape)
    for coords, _ in np.ndenumerate(m):
        idx_m[coords] = np.ravel_multi_index(coords, idx_m.shape)
        
    return idx_m

def adjacency_mask(m):
    '''
    Filter out points in the search space in the rectangle's perimeter
    (diagonals OK)
    '''
    
    rect_coords = np.argwhere(m)

    distance = np.empty(m.shape)
    for coords, _ in np.ndenumerate(distance):
        distance[coords] = min([cityblock(coords, rect) for rect in rect_coords])

    non_adjacent = distance > 1
    
    return non_adjacent

def visited_mask(m):
    '''
    Filter out points in the search space that have been visited before
    '''
    
    rect_idx = min([np.ravel_multi_index(c, m.shape) for c in np.argwhere(m)])
    not_visited = index_mtx(m) >= rect_idx
    
    return not_visited


def conjunction_mask(m): return np.logical_and(adjacency_mask(m), visited_mask(m))


def k2_hypothesis_space(n):
    '''
    Returns all possible pairs of rectangles within an n x n grid
    Constraints:
    k = 2 rectangles
    min rectangle dimensions: (2,2)
    max rectangle dimensions: n on either side
    Rectangles must be non-adjacent (diagonals OK)
    '''
    
    # "seed" rectangles
    k1 = k1_hypothesis_space(n)
    hypotheses = []
    
    # iterate over all k = 1 hypotheses
    for i in range(k1.shape[0]):
        rect = k1[i,:,:]
        search_space = conjunction_mask(rect)
    
        # possible second rectangles
        for (y0,x0), in_search in np.ndenumerate(search_space):
             if in_search:
                for x in range(x0+2,n+1):
                    for y in range(y0+2,n+1):
                        
                        # propose second rect
                        h_i = make_grid(n, (x0,x,y0,y))
                        
                        # accept proposal if all points lie in search space
                        if np.array_equal(h_i, h_i * search_space):
                            hypotheses.append(rect + h_i)

    hypotheses = np.array(hypotheses)
    
    return hypotheses

