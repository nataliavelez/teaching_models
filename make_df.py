#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 21:57:29 2020

@author: aliciachen
"""
import pandas as pd 

def make_df_from_spreadsheet(filename):
    '''
    given csv file from teaching_stimuli, turning each grid into array and assembling dataframe
    '''
    df = pd.read_csv(filename, header=1)
    
    # fixing row and column names, getting rid of the k values 
    df.fillna(method='ffill', inplace=True)
    df.columns = df.columns.to_series().mask(lambda x: x.str.startswith('Unnamed')).ffill()
    df = df.drop(columns=['Designer','k_A','k_B','k_C','k_D'])
    
    # setting index labels
    df = df.set_index(['#'])
    
    # making new df, which has the images in matrix form 
    rows = map(int, df.index.unique().tolist())
    columns = df.columns.unique().tolist() # should be ['A', 'B', 'C', 'D']
    new_df = pd.DataFrame(columns = columns).astype(object)
    
    # filling new df
    for row in rows: 
        for column in columns: 
            new_df.loc[row, column] = df.loc[row, column].to_numpy()
    
    return new_df