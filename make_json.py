#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:59:24 2021

@author: aliciachen
"""

# my_dict = [{
#         "h1": [(insert list of rows here)],
#         "h2": [],
#         "h3": [],
#         "h4": []
#         }, {
#         "h1": [],
#         "h2": [],
#         "h3": [],
#         "h4": []
#         }
#     ... etc
#     ]

import pandas as pd

from make_df import *
#%%

df = make_df_from_spreadsheet('teaching_stimuli - all_examples (7).csv')