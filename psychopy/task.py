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

#######

# TODO: import four problems and draw / highlight true hypothesis

# TODO: save all key presses (check how??)


#%% for each trial:

xlocs = [-.5, -.3, -.1, .1, .3, .5]
ylocs = xlocs.copy()
win = visual.Window([800, 600], monitor="testMonitor", units="norm")
trial = Move()

rects = {}
for i in range(6):
    rects[i] = {}
    for j in range(6):
        rects[i][j] = visual.Rect(win=win,
                       fillColor='black',
                       pos=(xlocs[i], ylocs[j]),
                       units='norm',
                       size=.17)

while True:

    allKeys=event.waitKeys()
    allowed_keys = ['left', 'right', 'up', 'down', 'enter', 'q']

    for i in range(6):
        for j in range(6):
            rects[i][j].draw()

    win.flip()

    old_cursor_loc = trial.cursor_loc.copy()


    for thisKey in allKeys:
        if thisKey == 'left':
            trial.left()
        elif thisKey == 'right':
            trial.right()
        elif thisKey == 'up':
            trial.up()
        elif thisKey == 'down':
            trial.down()
        elif thisKey == 'space':
            trial.select()

            # Update canvas to make past selections blue
            rects[trial.cursor_loc[0]][trial.cursor_loc[1]] = visual.Rect(win=win,
                   fillColor='blue',
                   pos=(xlocs[trial.cursor_loc[0]], ylocs[trial.cursor_loc[1]]),
                   units='norm',
                   size=.17)
            break
        elif thisKey == 'q':
            win.close()
            core.quit()

    print(old_cursor_loc)
    print(trial.cursor_loc)

    # Change current cursor location color
    if thisKey != 'space':
        try:
            rects[old_cursor_loc[0]][old_cursor_loc[1]] = visual.Rect(win=win,
                            fillColor='black',
                            pos=(xlocs[old_cursor_loc[0]], ylocs[old_cursor_loc[1]]),
                            units='norm',
                            size=.17)


            rects[trial.cursor_loc[0]][trial.cursor_loc[1]] = visual.Rect(win=win,
                            fillColor='green',
                            pos=(xlocs[trial.cursor_loc[0]], ylocs[trial.cursor_loc[1]]),
                            units='norm',
                            size=.17)
        except IndexError:
            print('index out of range')



    event.clearEvents()

# TODO: save key presses