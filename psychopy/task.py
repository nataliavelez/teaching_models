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

win = visual.Window([800, 600], monitor="testMonitor", units="norm")
trial = Move()

# TODO: make 36 blank rectangles with their positions

# make a d1ictionary like the previous one, but have each of them be an instance of visual.Rect


xlocs = [-.5, -.3, -.1, .1, .3, .5]
ylocs = xlocs.copy()

rects = {}
for i in range(6):
    rects[i] = {}
    for j in range(6):
        rects[i][j] =  visual.Rect(win=win,
                       fillColor='black',
                       pos=(xlocs[i], ylocs[j]),
                       units='norm',
                       size=.1)

# testrect = visual.Rect(win=win,
#                        fillColor='black', pos=(-.6, -.6),
#     #pos=(1/(trial.cursor_loc[0]+1), 1/(trial.cursor_loc[1]+1)), # avoid division by zero
#     units='norm',
#     size=.1
#     )

while True:
    for i in range(6):
        for j in range(6):
            rects[i][j].draw()

    win.flip()

    if len(event.getKeys()) > 0:
         break
    event.clearEvents()

#win.flip()

win.close()
core.quit()

## TODO: find way to somehow tie 0-5 coordinates to the way coordinates are plotted in psychopy

#######

# TODO: highlight pressed stuff on canvas, adn dont refresh this every time you move

# TODO: highlight cursor_loc, but have this refresh everytime you move

# TODO: save all key presses (check how??)


#%% for each trial:


while True:

    allKeys=event.waitKeys()
    allowed_keys = ['left', 'right', 'up', 'down', 'enter', 'q']
    for thisKey in allKeys:
        if thisKey == 'left':
            trial.left()
        elif thisKey == 'right':
            trial.right()
        elif thisKey == 'up':
            trial.up()
        elif thisKey == 'down':
            trial.down()
        elif thisKey == 'enter':
            trial.select()
        elif thisKey == 'q':
            win.close()
            core.quit()

    event.clearEvents()