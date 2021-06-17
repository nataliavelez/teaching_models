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

        # Make empty canvas
        # Value in squares (0 or 1) is whether square has been pressed
        squares = {}
        for i in range(6):
            squares[i] = {}
            for j in range(6):
                squares[i][j] = 0
        self.squares = squares

        # Starting cursor location
        self.cursor_loc = [0, 0]

    def select(self):
        self.squares[self.cursor_loc[0]][self.cursor_loc[1]] = 1

    def unselect(self):
        """Not needed"""
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


# %% Import and format teaching problems

f = open('/Users/aliciachen/Dropbox/teaching_models/problems.json')
problems_raw = json.load(f)
f.close()

probs = {}

for prob_idx, prob in enumerate(problems_raw):
    probs[prob_idx] = {}
    for h, val in prob.items():
        probs[prob_idx][h] = {}
        for row_idx, row in enumerate(val):
            probs[prob_idx][h][row_idx] = {}
            for col_idx, value in enumerate(row):
                probs[prob_idx][h][row_idx][col_idx] = value

# %% Test first teaching problem

test_prob = probs[0]
prob_idxs = list(test_prob.keys())

# Shuffle order of four images
random.shuffle(prob_idxs)
test_prob_shuffled = {i: test_prob[i] for i in prob_idxs}
true_h_idx = prob_idxs.index('A')  # Extract order of index

# Flatten problem to figure out colors; make color lists
colordict = {}
for h, val in test_prob_shuffled.items():
    colordict[h] = []
    for k, v in val.items():
        for a, b in v.items():
            colordict[h].append(b)

# Convert from dict into np array to assign colors
colorlist = []
for k, v in colordict.items():
    colorlist.append(v)

# Assign colors
newcolorlist = [[] for i in range(4)]
for idx, h in enumerate(colorlist):
    for val in h:
        if val == 0:
            newcolorlist[idx].append([30, 76, 124])
        else:
            newcolorlist[idx].append([72, 160, 248])

# Set locations of four hypotheses
locs = [[-.75, .5], [-.25, .5], [.25, .5], [.75, .5]]

# Draw
win = visual.Window([800, 600], monitor="test", units="norm", color='black')

# Make canvas
canv_sq_size = .05
canvas_loc = [0, -.5]

x_low, x_high = canvas_loc[0] - 2.5*canv_sq_size, canvas_loc[0] + 2.5*canv_sq_size
y_low, y_high = canvas_loc[1] - 2.5*canv_sq_size, canvas_loc[1] + 2.5*canv_sq_size

xlocs = np.linspace(x_low, x_high, 6)
ylocs = np.linspace(y_low, y_high, 6)

# Make hypotheses
sq_size = .05
hs = []

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


# Highlight true hypothesis
true_h_border = visual.Rect(win,
                            size=5*sq_size + .05,
                            lineWidth=7,
                            lineColor=(238, 188, 64),
                            colorSpace='rgb255',
                            pos=(locs[true_h_idx][0], locs[true_h_idx][1]))

# Add letters on top
letters = ['A', 'B', 'C', 'D']
letter_locs = [(-.75, .8), (-.25, .8), (.25, .8), (.75, .8)]

lets = []
for i, l in enumerate(letter_locs):
    lets.append(visual.TextStim(win=win,
                                text=letters[i],
                                pos=letter_locs[i],
                                color='white'
                                ))

# Start instance of trial
trial = Move()

# Make blank canvas, indicating true hypothesis
rects = {}
for i in range(6):
    rects[i] = {}
    for j in range(6):
        if test_prob['A'][i][j] == 0:
            rects[i][j] = visual.Rect(win=win,
                                      fillColor=(94, 93, 95),  # dark gray
                                      colorSpace='rgb255',
                                      pos=(xlocs[i], ylocs[j]),
                                      units='norm',
                                      size=canv_sq_size-.01)
        else:
            rects[i][j] = visual.Rect(win=win,
                                      fillColor=(214, 214, 214),  # light gray
                                      colorSpace='rgb255',
                                      pos=(xlocs[i], ylocs[j]),
                                      units='norm',
                                      size=canv_sq_size-.01)


# Show the example selected to the learner
msg = visual.TextStim(win, text='Sending example to [NAME]...')
learner_rects = {}
for i in range(6):
    learner_rects[i] = {}
    for j in range(6):
        learner_rects[i][j] = visual.Rect(win=win,
                                          fillColor=(94, 93, 95), # dark gray
                                          colorSpace='rgb255',
                                          pos=(xlocs[i], ylocs[j]),
                                          units='norm',
                                          size=canv_sq_size-.01)


# Add move on vs. stay on the same example keys
move_on = visual.TextBox(window=win,
                         text='Yes',
                         pos=(-.5, 0),
                         units='norm',
                         size=(.4,.2))

stay = visual.TextBox(window=win,
                         text='No',
                         pos=(0.5, 0),
                         units='norm',
                         size=(.4,.2))


# Interactivity
while True:

    # Canvas part
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
            if test_prob['A'][trial.cursor_loc[0]][trial.cursor_loc[1]] == 0:
                print('error :( cant pick negative examples')
            else:
                if trial.squares[trial.cursor_loc[0]][trial.cursor_loc[1]] == 0:
                    trial.select()
                    rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor = 'blue'

                    win.flip(clearBuffer=True)

                    msg.draw()

                    win.flip()

                    core.wait(2.0)

                    win.flip(clearBuffer=True)

                    for h in hs:
                        h.draw()

                    true_h_border.draw()

                    for lett in lets:
                        lett.draw()

                    for i in range(6):
                        for j in range(6):
                            learner_rects[i][j].draw()

                    win.flip()

                    core.wait(1.0)

                    for h in hs:
                        h.draw()

                    true_h_border.draw()

                    for lett in lets:
                        lett.draw()

                    learner_rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor=(72, 160, 248)

                    for i in range(6):
                        for j in range(6):
                            learner_rects[i][j].draw()

                    win.flip(clearBuffer=True)
                    core.wait(3.0)

                    move_on.draw()
                    stay.draw()
                    win.flip()

                    core.wait(2.0)

                else:
                    trial.unselect()
                    # TODO: get rid of unselect

            # TODO: Display a message for selecting negative examples?
            break
        elif thisKey == 'q':
            win.close()
            core.quit()

    print(allKeys)

    if thisKey != 'space':

        try:
            rects[old_cursor_loc[0]][old_cursor_loc[1]].lineColor = 'none'

            if test_prob['A'][old_cursor_loc[0]][old_cursor_loc[1]] == 0:
                rects[old_cursor_loc[0]][old_cursor_loc[1]].fillColor = (94, 93, 95)  # dark gray
            else:
                rects[old_cursor_loc[0]][old_cursor_loc[1]].fillColor = (214, 214, 214)  # light gray
            rects[trial.cursor_loc[0]][trial.cursor_loc[1]].lineColor = (72, 160, 248)  # blue
            rects[trial.cursor_loc[0]][trial.cursor_loc[1]].lineWidth = 6
        except (IndexError, KeyError):
            print('index out of range')

    # Highlight already pressed squares and neg example squares
    for i in range(6):
        for j in range(6):
            try:
                if test_prob['A'][i][j] == 0 and trial.cursor_loc == [i, j]:
                    rects[i][j].lineColor = (218, 60, 37)  # red
                    rects[i][j].lineWidth = 6
                if trial.squares[i][j] == 1:
                    rects[i][j].fillColor = (72, 160, 248)
                    if trial.cursor_loc == [i, j]:
                        rects[i][j].lineColor = (218, 60, 37)  # red
                        rects[i][j].lineWidth = 6
            except (IndexError, KeyError):
                print('index out of range')

    # Draw canvas
    for i in range(6):
        for j in range(6):
            rects[i][j].draw()

    # Draw top stuff
    for h in hs:
        h.draw()

    true_h_border.draw()

    # Draw ABCD
    for lett in lets:
        lett.draw()

    win.flip()

    event.clearEvents()