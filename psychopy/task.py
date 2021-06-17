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

# h_1 = visual.ElementArrayStim(win=win,
#                               fieldSize=.17,
#                               fieldPos=(-.75, .5),
#                               fieldShape='sqr',
#                               nElements=36,
#                               elementMask=None,
#                               elementTex=None)


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


# Highlight true hypothesis

true_h_border = visual.Rect(win,
                            size=5*sq_size + .05,
                            lineWidth=5,
                            lineColor=(238, 188, 64),
                            colorSpace='rgb255',
                            pos=(locs[true_h_idx][0], locs[true_h_idx][1]))


for h in hs:
    h.draw()

true_h_border.draw()

win.flip()
core.wait(5.0)
win.close()
core.quit()

#%% for each trial:
canv_sq_size = .05
canvas_loc = [0, -.5]
# xlocs = [-.5, -.3, -.1, .1, .3, .5]
# ylocs = xlocs.copy()

x_low, x_high = canvas_loc[0] - 2.5*canv_sq_size, canvas_loc[0] + 2.5*canv_sq_size
y_low, y_high = canvas_loc[1] - 2.5*canv_sq_size, canvas_loc[1] + 2.5*canv_sq_size

xlocs = np.linspace(x_low, x_high, 6)
ylocs = np.linspace(y_low, y_high, 6)

win = visual.Window([800, 600], monitor="test", units="norm", color='black')

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

trial = Move()

rects = {}
for i in range(6):
    rects[i] = {}
    for j in range(6):
        if test_prob['A'][i][j] == 0:
            rects[i][j] = visual.Rect(win=win,
                       fillColor=(94, 93, 95), # dark gray
                       colorSpace='rgb255',
                       pos=(xlocs[i], ylocs[j]),
                       units='norm',
                       size=canv_sq_size-.01)
        else:
            rects[i][j] = visual.Rect(win=win,
                       fillColor=(214, 214, 214),
                       colorSpace='rgb255',
                       pos=(xlocs[i], ylocs[j]),
                       units='norm',
                       size=canv_sq_size-.01)


# Showing the example selected to the learner

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
                    rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor='blue'
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

                    #while True:

                    move_on.draw()
                    stay.draw()
                    win.flip()

                    core.wait(2.0)


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
    ## theres some sort of edge case error here..... fix later
    if thisKey != 'space':

        try:
            rects[old_cursor_loc[0]][old_cursor_loc[1]].lineColor = 'none'

            if test_prob['A'][old_cursor_loc[0]][old_cursor_loc[1]] == 0:
                rects[old_cursor_loc[0]][old_cursor_loc[1]].fillColor = (94, 93, 95) # dark gray
            else:
                rects[old_cursor_loc[0]][old_cursor_loc[1]].fillColor = (214, 214, 214) # light gray
            rects[trial.cursor_loc[0]][trial.cursor_loc[1]].lineColor = (72, 160, 248) # blue
            rects[trial.cursor_loc[0]][trial.cursor_loc[1]].lineWidth = 6
        except (IndexError, KeyError):
            print('index out of range')

    # Highlight already pressed squares
    # Highlight squares tthat are neg examples
    for i in range(6):
        for j in range(6):
            try:
                if test_prob['A'][i][j] == 0 and trial.cursor_loc == [i, j]:
                    rects[i][j].lineColor = (218, 60, 37) # red
                    rects[i][j].lineWidth = 6
                if trial.squares[i][j] == 1:
                    rects[i][j].fillColor = (72, 160, 248)
                    if trial.cursor_loc == [i, j]:
                        rects[i][j].lineColor = (218, 60, 37) # red
                        rects[i][j].lineWidth = 6
            except (IndexError, KeyError):
                print('index out of range')

    for i in range(6):
            for j in range(6):
                rects[i][j].draw()

        # Draw top stuff
    for h in hs:
        h.draw()

    true_h_border.draw()

    for lett in lets:
        lett.draw()

    win.flip()


    event.clearEvents()

# TODO: after every enter, a screen pops up asking hem if they would like to stop or continue