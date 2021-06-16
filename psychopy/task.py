#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 08:46:01 2021

@author: aliciachen
"""
from psychopy import gui, core, visual, event, data
import numpy as np
import json
import random

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

    def unselect(self):
        self.squares[self.cursor_loc[0]][self.cursor_loc[1]] = 0

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

## TODO: import four problems and draw / highlight true hypothesis

# TODO: import json file, loop tthru items and change to dict of dict of dicst

f = open('/Users/aliciachen/Dropbox/teaching_models/problems.json')
problems_raw = json.load(f)
f.close()

#%%


# Import problems s.t. each value can be accessed by coordinates

probs = {}

for prob_idx, prob in enumerate(problems_raw):
    probs[prob_idx] = {}
    for h, val in prob.items():
        probs[prob_idx][h] = {}
        for row_idx, row in enumerate(val):
            probs[prob_idx][h][row_idx] = {}
            for col_idx, value in enumerate(row):
                probs[prob_idx][h][row_idx][col_idx] = value


#%% TODO: map to four images

test_prob = probs[0]
prob_idxs = list(test_prob.keys())

random.shuffle(prob_idxs)
test_prob_shuffled = {i: test_prob[i] for i in prob_idxs}

true_h_idx = prob_idxs.index('A')

h_1 = visual.ElementArrayStim(win=win,
                              fieldSize=.17,
                              fieldPos=(-.75, .5),
                              fieldShape='sqr',
                              nElements=36,
                              elementMask=None,
                              elementTex=None)


# TODO: hilight true hypothesis. find idx that corresponds to A and make a square in that location

#%% Place for testing stuff

# array of coordinates for each element

# Flatten problem to figure out colors; make color lists
colordict = {}
for h, val in test_prob_shuffled.items():
    colordict[h] = []
    for k, v in val.items():
        for a, b in v.items():
            colordict[h].append(b)

# Convert from dict into np array to assign colors, thten back into listt of lists
colorlist = []
for k, v in colordict.items():
        colorlist.append(v)

newcolorlist = [[] for i in range(4)]
for idx, h in enumerate(colorlist):
    for val in h:
        if val == 0:
            newcolorlist[idx].append([30, 76, 124])
        else:
            newcolorlist[idx].append([72, 160, 248])


locs = [[-.75, .5], [-.25, .5], [.25, .5], [.75, .5]]

# populate xys
sidelength = 6
sq_size = .05


hs = []
win = visual.Window([800, 600], monitor="test", units="norm")

# colors = np.random.random((36, 3))
# colors = np.array([["#004D7F" for i in range(36)]], dtype='str').T
#colors = [[30, 76, 124] for i in range(36)]

#colors[7] = [72, 160, 248]


for i, loc in enumerate(locs):

    xys = []
    x_low, x_high = loc[0] - 2.5*sq_size, loc[0] + 2.5*sq_size
    y_low, y_high = loc[1] - 2.5*sq_size, loc[1] + 2.5*sq_size

    xs = np.linspace(x_low, x_high, 6)
    ys = np.linspace(y_low, y_high, 6)

    for y in ys:
        for x in xs:
            xys.append((x, y))


    hs.append(visual.ElementArrayStim(win=win,
                                   xys=xys,
                                   colors=newcolorlist[i],
                                   colorSpace='rgb255',
                                  fieldShape='sqr',
                                  nElements=36,
                                  elementMask=None,
                                  elementTex=None,
                                  sizes=(.03, .03)))



for h in hs:
    h.draw()

win.flip()
core.wait(5.0)
win.close()
core.quit()


# TODO: can call setcolors to set colors

#%% for each trial:

xlocs = [-.5, -.3, -.1, .1, .3, .5]
ylocs = xlocs.copy()
win = visual.Window([800, 600], monitor="test", units="norm")
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

    allKeys = event.waitKeys()
    allowed_keys = ['left', 'right', 'up', 'down', 'enter', 'q']


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
            if trial.squares[trial.cursor_loc[0]][trial.cursor_loc[1]] == 0:
                trial.select()

                rects[trial.cursor_loc[0]][trial.cursor_loc[1]] = visual.Rect(win=win,
                   fillColor='blue',
                   pos=(xlocs[trial.cursor_loc[0]], ylocs[trial.cursor_loc[1]]),
                   units='norm',
                   size=.17)
            else:
                trial.unselect()

            # Update canvas to make past selections blue


            # TODO: make it impossible to select negative examples (like display a message?)
            break
        elif thisKey == 'q':
            win.close()
            core.quit()

    print(allKeys)
    #print(old_cursor_loc)
    #print(trial.cursor_loc)

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

    # Pressed squares
    for i in range(6):
        for j in range(6):
            if trial.squares[i][j] == 1:
                rects[i][j] = visual.Rect(win=win,
                            fillColor='blue',
                            pos=(xlocs[i], ylocs[j]),
                            units='norm',
                            size=.17)

    # TODO: should we make it possible to unprerss the squares if u accidentally pressed thte squares?

    for i in range(6):
            for j in range(6):
                rects[i][j].draw()

    win.flip()


    event.clearEvents()

# TODO: save all key presses