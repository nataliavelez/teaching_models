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
        if x > 5:
            print("Invalid move")
        else:
            self.cursor_loc[0] = x

    def up(self):
        y = self.cursor_loc[1] + 1
        if y > 5:
            print("Invalid move")
        else:
            self.cursor_loc[1] = y

    def down(self):
        y = self.cursor_loc[1] - 1
        if y < 0:
            print("Invalid move")
        else:
            self.cursor_loc[1] = y


class Feedback:

    def __init__(self):
        self.cursor_loc = 0  # 0 for left, 1 for right
        self.exs_left = 4  # 4 possible examples to select per teaching prob
        self.selected = False

    def left(self):
        self.cursor_loc = 0

    def right(self):
        self.cursor_loc = 1

    def select(self):
        self.selected = True

        if self.cursor_loc == 0:
            self.exs_left -= 1  # Move on to selecting next block


        # elif self.cursor_loc == 1:
        #     # TODO: Move on to next teaching problem
        #     win.close()
        #     core.quit()
        #     break

        # if self.exs_left == 0:
        #     # TODO: move on to next teaching problem
        #     win.close()
        #     core.quit()


# %% Import and format teaching problems

f = open('/Users/aliciachen/Dropbox/teaching_models/problems.json')
problems_raw = json.load(f)
f.close()

probs = {}

# Fix issue with rotation AHHHHHHHH OMG it starts counting from the top

# OMG ROWS AND COLUMNS ARE SWITCHED AHHAAHHAHAsdklfjlkj how to switch this.... .

# honestly we can just do a numpy transpose kind of situation

# transpose matrices to make indexing easier
probstest = []
for prob in problems_raw:
    thisprob = {}
    for h, val in prob.items():  # h is ABCD index
        thismtx = np.array(val)
        thismtxT = thismtx.T
        thisprob[h] = thismtxT.tolist()
    probstest.append(thisprob)


# Testing problem 27 because idk why it has 7 columns...
# prob_27 = problems_raw[27]
# thisprob = {}
# for h, val in prob_27.items():  # h is ABCD index
#     thismtx = np.array(val)
#     thismtxT = thismtx.T
#     thisprob[h] = thismtxT.tolist()


###
problems_raw = probstest

for prob_idx, prob in enumerate(problems_raw):
    probs[prob_idx] = {}
    for h, val in prob.items():
        probs[prob_idx][h] = {}
        for row_idx, row in enumerate(val):
            probs[prob_idx][h][row_idx] = {}
            for col_idx, value in enumerate(reversed(row)):
                probs[prob_idx][h][row_idx][col_idx] = value


# Load problems per run from subj lists of lists for prepared problem file

f = open('subj_list.json')
subj_list = json.load(f)
f.close()

prepared_probs = {}

for subj_id, runs in enumerate(subj_list):
    prepared_probs[subj_id] = {}
    for run_idx, run in enumerate(runs):
        prepared_probs[subj_id][run_idx] = {i: probs[i] for i in run}

# %%

# Access problem based on subject and run:
subj_id = 1
run_idx = 1
probs = prepared_probs[subj_id][run_idx]


#%% actual stuff to run

data = {}

win = visual.Window([800, 600], monitor="test", units="norm", color='black')

for prob_idx, v in probs.items():

    data[prob_idx] = {}
    data[prob_idx]['prob'] = v

    problemFinished = False
    test_prob = v
    prob_idxs = list(test_prob.keys())

    # Shuffle order of four images
    random.shuffle(prob_idxs)
    test_prob_shuffled = {i: test_prob[i] for i in prob_idxs}
    true_h_idx = prob_idxs.index('h1')  # Extract order of index
    data[prob_idx]['true_h_idx'] = true_h_idx

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
                newcolorlist[idx].append([30, 76, 124])  # Dark blue
            else:
                newcolorlist[idx].append([72, 160, 248])

    # Set locations of four hypotheses
    locs = [[-.75, .5], [-.25, .5], [.25, .5], [.75, .5]]

    # Draw
    #win = visual.Window([800, 600], monitor="test", units="norm", color='black')

    # Make hypotheses
    sq_size = .05
    hs = []

    for i, loc in enumerate(locs):

        xys = []
        x_low, x_high = loc[0] - 3*sq_size, loc[0] + 3*sq_size
        y_low, y_high = loc[1] - 4*sq_size, loc[1] + 4*sq_size

        xs = np.linspace(x_low, x_high, 6)
        ys = np.linspace(y_low, y_high, 6)

        for x in xs:
            for y in ys:
                xys.append((x, y))

        hs.append(visual.ElementArrayStim(win=win,
                                          xys=xys,
                                          colors=newcolorlist[i],
                                          colorSpace='rgb255',
                                          fieldShape='sqr',
                                          nElements=36,
                                          elementMask=None,
                                          elementTex=None,
                                          sizes=(.05, .0666)))


    # Highlight true hypothesis
    true_h_border = visual.Rect(win,
                                size=((6*sq_size + .085), (4/3)*(6*sq_size + .085)),
                                lineWidth=30,
                                lineColor=(238, 188, 64),
                                colorSpace='rgb255',
                                pos=(locs[true_h_idx][0], locs[true_h_idx][1]))

    # Add letters on top
    letters = ['A', 'B', 'C', 'D']
    letter_locs = [(-.75, .83), (-.25, .83), (.25, .83), (.75, .83)]

    lets = []
    for i, l in enumerate(letter_locs):
        lets.append(visual.TextStim(win=win,
                                    text=letters[i],
                                    pos=letter_locs[i],
                                    color='white'
                                    ))

    # Start instance of trial
    trial = Move()

    # Make study text
    studytext = visual.TextStim(win, text='Study problem', pos=(0, -.72))

    # Make canvas
    canv_sq_size = .08
    canvas_loc = [0, -.22]

    x_low, x_high = canvas_loc[0] - 3*canv_sq_size, canvas_loc[0] + 3*canv_sq_size
    y_low, y_high = canvas_loc[1] - 4*canv_sq_size, canvas_loc[1] + 4*canv_sq_size

    xlocs = np.linspace(x_low, x_high, 6)
    ylocs = np.linspace(y_low, y_high, 6)


    # Make blank canvas, indicating true hypothesis
    rects = {}
    for i in range(6):
        rects[i] = {}
        for j in range(6):
            if test_prob['h1'][i][j] == 0:
                rects[i][j] = visual.Rect(win=win,
                                          fillColor=(94, 93, 95),  # dark gray
                                          colorSpace='rgb255',
                                          pos=(xlocs[i], ylocs[j]),
                                          units='norm',
                                          size=(canv_sq_size, (4/3)*canv_sq_size))
            else:
                rects[i][j] = visual.Rect(win=win,
                                          fillColor=(214, 214, 214),  # light gray
                                          colorSpace='rgb255',
                                          pos=(xlocs[i], ylocs[j]),
                                          units='norm',
                                          size=(canv_sq_size, (4/3)*canv_sq_size))


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
                                              size=(canv_sq_size, (4/3)*canv_sq_size))



    # Move on and stay text boxes
    stay = visual.TextBox(window=win,
                             text='Yes',
                             font_size=30,
                             #colorSpace='rgb255',
                             opacity=1,
                             font_color=[1, 1, 1],
                             #background_color=[0, 0, 0],
                             border_color=[1, 1, 1],
                             border_stroke_width=4,
                             textgrid_shape=[3,1],
                             pos=(-.5, 0)
                             )

    stay.setBackgroundColor([.122, .297, .486])  # Default starting key point

    move_on = visual.TextBox(window=win,
                             text='No',
                             font_size=30,
                             opacity=1,
                             #colorSpace='rgb255',
                             font_color=[1, 1, 1],
                             #background_color=[0, 0, 0],
                             border_color=[1, 1, 1],
                             border_stroke_width=4,
                             textgrid_shape=[3,1],
                             pos=(.5, 0)
                             )

    buttons = [move_on, stay]

    # Text on top and underneath move on and stay text boxes
    cont = visual.TextStim(win=win,
                           text="Continue?",
                           pos=(0, .3),
                           height=.09)



    # New continue screen text
    leftarrow = visual.TextStim(win=win, text='\u2190', pos=(-.5, 0),
                                font="Menlo", height=.3)
    rightarrow = visual.TextStim(win=win, text='\u2192', pos=(.5, 0),
                                 font="Menlo", height=.3)

    yestext = visual.TextStim(win=win,
                          text="Yes",
                          pos=(-.5, -.26),
                          height=.1)

    notext = visual.TextStim(win=win,
                             text="No",
                             pos=(.5, -.26),
                             height=.1)



    nKeys = 0
    maxExamples = 4
    examplesLeft = 4

    # fixation cross
    iti = visual.TextStim(win=win, text="+", height=.2)

    # Wait time at beginning of each problem

    studytext.draw()

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
    core.wait(15.0) # Study time

    # Countdown

    onesec = visual.TextStim(win=win, text='3', pos=(0, -.88))

    onesec.draw()
    studytext.draw()

    for i in range(6):
        for j in range(6):
            rects[i][j].draw()

    for h in hs:
        h.draw()

    true_h_border.draw()

    for lett in lets:
        lett.draw()

    win.flip()

    core.wait(1.0)

    twosec = visual.TextStim(win=win, text='2', pos=(0, -.88))

    twosec.draw()
    studytext.draw()

    for i in range(6):
        for j in range(6):
            rects[i][j].draw()

    for h in hs:
        h.draw()

    true_h_border.draw()

    for lett in lets:
        lett.draw()

    win.flip()

    core.wait(1.0)


    threesec = visual.TextStim(win=win, text='1', pos=(0, -.88))
    threesec.draw()
    studytext.draw()

    for i in range(6):
        for j in range(6):
            rects[i][j].draw()

    for h in hs:
        h.draw()

    true_h_border.draw()

    for lett in lets:
        lett.draw()

    win.flip()

    core.wait(1.0)

    # starting cursor color
    if test_prob['h1'][0][0] == 0:
        rects[0][0].lineColor = (218, 60, 37)  # red
        rects[0][0].lineWidth = 15
    else:
        rects[0][0].lineColor = (72, 160, 248)  # blue
        rects[0][0].lineWidth = 15

    timer = core.CountdownTimer(15)
    while examplesLeft > 0:

        # Start by drawing the stuff we start with



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

         # Time runs out, but in this state only works if you are clicking things
        if timer.getTime() < 0:

            if examplesLeft == 1:
                break # TODO: move on to next teaching problem

            len_allKeys1 = 0
            timer1 = core.CountdownTimer(7)

            # The issue with the timer is that the items arent in
            while True: # Move on and stay buttons



                leftarrow.draw()
                rightarrow.draw()
                cont.draw()
                yestext.draw()
                notext.draw()



                win.flip()

                allKeys1 = event.getKeys()

                if timer1.getTime() < 0:
                    event.clearEvents()

                    examplesLeft -= 1  # decrease by one example
                    timer.reset()
                    timer1.reset()
                    break


                if len(allKeys1) > len_allKeys1:

                    thisKey = allKeys1[-1]
                    if thisKey == "left":

                        event.clearEvents()

                        examplesLeft -= 1  # decrease by one example
                        timer.reset()
                        timer1.reset()
                        break

                    elif thisKey == "right":

                        examplesLeft == 0
                        problemFinished = True
                        break



                len_allKeys1 = len(allKeys1)
                event.clearEvents()

            core.wait(0.5)

        if problemFinished:
                break
        #nKeys = 0
        allKeys = event.getKeys()
        allowed_keys = ['left', 'right', 'up', 'down', 'enter', 'q']
        # Canvas part

        old_cursor_loc = trial.cursor_loc.copy()

        if len(allKeys) > nKeys:
            thisKey = allKeys[-1]

            if thisKey == 'left':
                trial.left()
            elif thisKey == 'right':
                trial.right()
            elif thisKey == 'up':
                trial.up()
            elif thisKey == 'down':
                trial.down()
            elif thisKey == 'space':
                if test_prob['h1'][trial.cursor_loc[0]][trial.cursor_loc[1]] == 0:
                    print('error :( cant pick negative examples')
                else:
                    if trial.squares[trial.cursor_loc[0]][trial.cursor_loc[1]] == 0:
                        trial.select()
                        rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor = 'blue'

                        win.flip(clearBuffer=True)
                        core.wait(random.uniform(1, 3)) # ISI: black screen
                        # End of time block
                        msg.draw()

                        win.flip()

                        core.wait(3)

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

                        core.wait(1.7) # blank canvas before example shown to learner

                        # make selected example appear

                        for h in hs:
                            h.draw()

                        true_h_border.draw()

                        for lett in lets:
                            lett.draw()

                        learner_rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor=(161, 103, 201) # fillColor=(72, 160, 248)

                        for i in range(6):
                            for j in range(6):
                                learner_rects[i][j].draw()

                        win.flip(clearBuffer=True)
                        core.wait(4)
                        win.flip(clearBuffer=True)
                        core.wait(random.uniform(1, 3))

                        if examplesLeft == 1:
                            break  # TODO: move on to next teaching problem

                        # Feedback screen
                        event.clearEvents()

                        len_allKeys1 = 0
                        timer1 = core.CountdownTimer(7)

                        # previous learner rects are the same color
                        learner_rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor=(72, 160, 248)

                        while True: # Move on and stay buttons


                            leftarrow.draw()
                            rightarrow.draw()
                            cont.draw()
                            yestext.draw()
                            notext.draw()



                            win.flip()

                            allKeys1 = event.getKeys()

                            if timer1.getTime() < 0:
                                event.clearEvents()

                                examplesLeft -= 1  # decrease by one example
                                timer.reset()
                                timer1.reset()
                                break


                            if len(allKeys1) > len_allKeys1:

                                thisKey = allKeys1[-1]
                                if thisKey == "left":

                                    event.clearEvents()

                                    examplesLeft -= 1  # decrease by one example
                                    timer.reset()
                                    timer1.reset()
                                    break

                                elif thisKey == "right":

                                    examplesLeft == 0
                                    problemFinished = True
                                    break



                            len_allKeys1 = len(allKeys1)
                            event.clearEvents()

                        win.flip(clearBuffer=True)

                        iti.draw()
                        win.flip()
                        core.wait(random.uniform(5, 7))

                        # ITI




                    else:
                        trial.unselect()
                        # TODO: get rid of unselect

                # TODO: Display a message for selecting negative examples?
                #break
            elif thisKey == 'q':
                win.close()
                core.quit()

            print(thisKey)


            if problemFinished:
                break

            if thisKey != 'space':

                try:
                    rects[old_cursor_loc[0]][old_cursor_loc[1]].lineColor = 'none'

                    if test_prob['h1'][old_cursor_loc[0]][old_cursor_loc[1]] == 0:
                        rects[old_cursor_loc[0]][old_cursor_loc[1]].fillColor = (94, 93, 95)  # dark gray
                    else:
                        rects[old_cursor_loc[0]][old_cursor_loc[1]].fillColor = (214, 214, 214)  # light gray
                    rects[trial.cursor_loc[0]][trial.cursor_loc[1]].lineColor = (72, 160, 248)  # blue
                    rects[trial.cursor_loc[0]][trial.cursor_loc[1]].lineWidth = 15
                except (IndexError, KeyError):
                    print('index out of range')  # TODO: change to a visual stim



            # Highlight already pressed squares and neg example squares
            for i in range(6):
                for j in range(6):
                    try:
                        if test_prob['h1'][i][j] == 0 and trial.cursor_loc == [i, j]:
                            rects[i][j].lineColor = (218, 60, 37)  # red
                            rects[i][j].lineWidth = 15
                        if trial.squares[i][j] == 1:
                            rects[i][j].fillColor = (72, 160, 248)
                            if trial.cursor_loc == [i, j]:
                                rects[i][j].lineColor = (218, 60, 37)  # red
                                rects[i][j].lineWidth = 15
                    except (IndexError, KeyError):
                        print('index out of range')  # TODO: change to a visual stim



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

        data[prob_idx]['canvas'] = trial.squares
        #event.clearEvents()

        nKeys = len(allKeys)

    win.flip(clearBuffer=True)

    iti.draw()
    win.flip()
    core.wait(random.uniform(5, 7))


endofrun = visual.TextStim(win, text="End of run", height=0.25)
endofrun.draw()
win.flip()

core.wait(3.3)

win.close()
core.quit()