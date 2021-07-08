#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 08:27:47 2021

@author: aliciachen
"""

### Make list of lists of randomized problem indices for each subject

import random
import json

random.seed(8)


nSubjects = 30
nRuns = 8
nProbsPerRun = 5

nProbs = nRuns * nProbsPerRun

probs = [prob for prob in range(nProbs)]

subj_list = [[[] for j in range(nRuns)] for i in range(nSubjects)]

for sub in range(nSubjects):

    # Randomize problem list for each subject
    random.shuffle(probs)

    # Divide into nRuns sublists
    probs_div = [probs[i:i + nProbsPerRun] for i in range(0, len(probs), nProbsPerRun)]

    for run in range(nRuns):
        subj_list[sub][run] = probs_div[run]

