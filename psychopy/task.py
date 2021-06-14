#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 08:46:01 2021

@author: aliciachen
"""
from psychopy import gui, core, visual, event, data
import numpy as np

class Canvas:

    def __init__(self):

        # Make empty canvas, where value (0 or 1) is whether square has been pressed
        squares = {}
        for i in range(6):
            squares[i] = {}
            for j in range(6):
                squares[i][j] = 0
        self.squares = squares

        # TODO: Import json file and turn into dict of dicts for pos/neg exs

        # Starting cursor location
        self.cursor_loc = [0, 0]

    def select(self):
        self.squares[self.cursor_loc[0]][self.cursor_loc[1]] = 1
        # TODO: add something where you can't select the negative examples

class Move(Canvas):

    def left(self):
        x = self.cursor_loc[0] - 1
        if x < 0:
            print("Invalid move")
        else:
            self.cursor_loc[0] = x

    def right(self):
        x = self.cursor_loc[0] + 1
        if x < 0:
            print("Invalid move")
        else:
            self.cursor_loc[0] = x

    def up(self):
        y = self.cursor_loc[1] + 1
        if y < 0:
            print("Invalid move")
        else:
            self.cursor_loc[1] = y

    def down(self):
        y = self.cursor_loc[1] - 1
        if y < 0:
            print("Invalid move")
        else:
            self.cursor_loc[1] = y
#%%

# TODO: add psychopy input w key presses, etc.

# TODO: highlight cursor_loc

# TODO: save key presses

thisResp = None
while thisResp == None:
    allKeys=event.waitKeys()
    for thisKey in allKeys:
        if thisKey=='left':
            if targetSide==-1: thisResp = 1  # correct
            else: thisResp = -1              # incorrect
        elif thisKey=='right':
            if targetSide== 1: thisResp = 1  # correct
            else: thisResp = -1              # incorrect
        elif thisKey in ['q', 'escape']:
            core.quit()  # abort experiment
    event.clearEvents()  # clear other (eg mouse) events - they clog the buffe