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

# %% feedback
win = visual.Window([800, 600], monitor="test", units="norm", color='black')


# stay = visual.TextBox(window=win,
#                          text='Yes',
#                          font_size=30,
#                          #colorSpace='rgb255',
#                          opacity=1,
#                          font_color=[1, 1, 1],
#                          #background_color=[0, 0, 0],
#                          border_color=[1, 1, 1],
#                          border_stroke_width=4,
#                          textgrid_shape=[3,1], # 20 cols (20 chars wide)
#                                                  # by 4 rows (4 lines of text)
#                          pos=(-.5, 0)
#                          )

# # visual.TextBox(window=win,
# #                          text='Yes',
# #                          pos=(-.5, 0),
# #                          color_space='rgb',
# #                          units='norm',
# #                          size=(.4,.2))

# move_on = visual.TextBox(window=win,
#                          text='No',
#                          font_size=30,
#                          opacity=1,
#                          #colorSpace='rgb255',
#                          font_color=[1, 1, 1],
#                          #background_color=[0, 0, 0],
#                          border_color=[1, 1, 1],
#                          border_stroke_width=4,
#                          textgrid_shape=[3,1], # 20 cols (20 chars wide)
#                                                  # by 4 rows (4 lines of text)
#                          pos=(.5, 0)
#                          )

# buttons = [move_on, stay]
#%%

# testtrial = Feedback()


# allKeys = event.waitKeys()

# for thisKey in allKeys:
#     if thisKey == 'left':
#         testtrial.left()
#         move_on.setBackgroundColor('green')
#     if thisKey == 'right':
#         testtrial.right()
#         stay.setBackgroundColor('green')
#     if thisKey == 'enter':
#         buttons[testtrial.cursor_loc].setBackgroundColor('blue')
#         testtrial.select()





## Make feedback screen


# cursor_loc and selected tells you where you are re: the examples
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
prepared_probs = {}

for subj_id, runs in enumerate(subj_list):
    prepared_probs[subj_id] = {}
    for run_idx, run in enumerate(runs):
        prepared_probs[subj_id][run_idx] = {i: probs[i] for i in run}

# %%

# Access problem based on subject and run:
subj_id = 0
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
    true_h_idx = prob_idxs.index('A')  # Extract order of index
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

    # Make canvas
    canv_sq_size = .08
    canvas_loc = [0, -.45]

    x_low, x_high = canvas_loc[0] - 3*canv_sq_size, canvas_loc[0] + 3*canv_sq_size
    y_low, y_high = canvas_loc[1] - 4*canv_sq_size, canvas_loc[1] + 4*canv_sq_size

    xlocs = np.linspace(x_low, x_high, 6)
    ylocs = np.linspace(y_low, y_high, 6)


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

    yestext = visual.TextStim(win=win,
                              text="Provide another example",
                              pos=(-.5, -.2),
                              height=.07)

    notext = visual.TextStim(win=win,
                             text="Next problem",
                             pos=(.5, -.2),
                             height=.07)


    timer = core.CountdownTimer(15)
    nKeys = 0
    maxExamples = 4
    examplesLeft = 4

    # starting cursor color
    if test_prob['A'][0][0] == 0:
        rects[0][0].lineColor = (218, 60, 37)  # red
        rects[0][0].lineWidth = 15
    else:
        rects[0][0].lineColor = (72, 160, 248)  # blue
        rects[0][0].lineWidth = 15

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

            testtrial = Feedback()

            len_allKeys1 = 0
            timer1 = core.CountdownTimer(7)

            # The issue with the timer is that the items arent in
            while True: # Move on and stay buttons



                move_on.draw()
                stay.draw()
                cont.draw()
                yestext.draw()
                notext.draw()



                win.flip()

                allKeys1 = event.getKeys()

                if timer1.getTime() < 0:
                    testtrial.select()
                        # timer1.reset()
                    if testtrial.cursor_loc == 1:
                        examplesLeft == 0
                        problemFinished = True
                        break

                    if testtrial.cursor_loc == 0:
                        event.clearEvents()

                        examplesLeft -= 1  # decrease by one example
                        timer.reset()
                        timer1.reset()
                        break

                if len(allKeys1) > len_allKeys1:
                    thisKey = allKeys1[-1]

                    if thisKey == 'left':
                        testtrial.left()
                        stay.setBackgroundColor([.122, .297, .486]) # blue
                        move_on.setBackgroundColor(None) # black
                    if thisKey == 'right':
                        testtrial.right()
                        move_on.setBackgroundColor([.122, .297, .486]) # blue
                        stay.setBackgroundColor(None) # black
                    if (thisKey == 'space') or (timer1.getTime() < 0):
                        testtrial.select()
                        # timer1.reset()
                        if testtrial.cursor_loc == 1:
                            examplesLeft == 0
                            problemFinished = True
                            break



                    #win.flip()

                    # Continue providing examples
                    if ((thisKey == 'space') or (timer1.getTime() < 0)) and (testtrial.cursor_loc == 0):
                        event.clearEvents()

                        examplesLeft -= 1  # decrease by one example
                        timer.reset()
                        timer1.reset()
                        break


                len_allKeys1 = len(allKeys1)
                event.clearEvents()

            core.wait(1.0)

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
                if test_prob['A'][trial.cursor_loc[0]][trial.cursor_loc[1]] == 0:
                    print('error :( cant pick negative examples')
                else:
                    if trial.squares[trial.cursor_loc[0]][trial.cursor_loc[1]] == 0:
                        trial.select()
                        rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor = 'blue'

                        win.flip(clearBuffer=True)

                        # End of time block
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


                        if examplesLeft == 1:
                            break  # TODO: move on to next teaching problem

                        # Feedback screen
                        event.clearEvents()
                        testtrial = Feedback()

                        len_allKeys1 = 0
                        timer1 = core.CountdownTimer(7)


                        while True: # Move on and stay buttons



                            move_on.draw()
                            stay.draw()
                            cont.draw()
                            yestext.draw()
                            notext.draw()



                            win.flip()

                            if timer1.getTime() < 0:

                                testtrial.select()
                                    # timer1.reset()
                                if testtrial.cursor_loc == 1:
                                    examplesLeft == 0
                                    problemFinished = True
                                    break

                                if testtrial.cursor_loc == 0:
                                    event.clearEvents()

                                    examplesLeft -= 1  # decrease by one example
                                    timer.reset()
                                    timer1.reset()
                                    break

                            len_allkeys1 = len(allKeys1)
                            allKeys1 = event.getKeys()
                            if len(allKeys1) > len_allKeys1:
                                thisKey = allKeys1[-1]

                            # For reference:
                            # stay = visual.TextBox(window=win,
                            #  text='No',
                            #  font_size=30,
                            #  font_color=[1,-1,-1],
                            #  background_color=[-1,-1,-1,1],
                            #  border_color=[-1,-1,1,1],
                            #  border_stroke_width=4,
                            #  textgrid_shape=[3,1], # 20 cols (20 chars wide)
                            #                          # by 4 rows (4 lines of text)
                            #  pos=(.5, 0)
                            #  )

                            #for thisKey in allKeys:
                                if thisKey == 'left':
                                    testtrial.left()
                                    stay.setBackgroundColor([.122, .297, .486]) # blue
                                    move_on.setBackgroundColor(None) # black
                                if thisKey == 'right':
                                    testtrial.right()
                                    move_on.setBackgroundColor([.122, .297, .486]) # blue
                                    stay.setBackgroundColor(None) # black
                                if thisKey == 'space' or timer1.getTime() < 0:
                                    testtrial.select()

                                    if testtrial.cursor_loc == 1:
                                        examplesLeft == 0
                                        problemFinished = True
                                        break
                                    # timer1.reset()
                                    # if testtrial.cursor_loc == 0:
                                    #     break



                                #win.flip()

                                # Continue providing examples
                                if (thisKey == 'space' or timer1.getTime() < 0) and testtrial.cursor_loc == 0:
                                    event.clearEvents()

                                    examplesLeft -= 1  # decrease by one example
                                    timer.reset()
                                    timer1.reset()
                                    break



                        core.wait(1.0)
                        # testtrial = Feedback()

                        # len_allKeys1 = 0
                        # timer2 = core.CountdownTimer(10)

                        # while True: # Move on and stay buttons



                        #     move_on.draw()
                        #     stay.draw()
                        #     cont.draw()
                        #     yestext.draw()
                        #     notext.draw()



                        #     win.flip()

                        #     allKeys1 = event.getKeys()
                        #     if len(allKeys1) > len_allKeys1:
                        #         thisKey = allKeys1[-1]
                        #     len_allKeys1 = len(allKeys1)
                        #     # For reference:
                        #     # stay = visual.TextBox(window=win,
                        #     #  text='No',
                        #     #  font_size=30,
                        #     #  font_color=[1,-1,-1],
                        #     #  background_color=[-1,-1,-1,1],
                        #     #  border_color=[-1,-1,1,1],
                        #     #  border_stroke_width=4,
                        #     #  textgrid_shape=[3,1], # 20 cols (20 chars wide)
                        #     #                          # by 4 rows (4 lines of text)
                        #     #  pos=(.5, 0)
                        #     #  )

                        #     #for thisKey in allKeys:
                        #     if thisKey == 'left':
                        #         testtrial.left()
                        #         stay.setBackgroundColor([.122, .297, .486]) # blue
                        #         move_on.setBackgroundColor(None) # black
                        #     if thisKey == 'right':
                        #         testtrial.right()
                        #         move_on.setBackgroundColor([.122, .297, .486]) # blue
                        #         stay.setBackgroundColor(None) # black
                        #     if thisKey == 'space' or timer2.getTime() < 0:
                        #         testtrial.select()
                        #         # timer1.reset()
                        #         # if testtrial.cursor_loc == 0:
                        #         #     break



                        #     #win.flip()

                        #     # Continue providing examples
                        #     if (thisKey == 'space' or timer2.getTime() < 0) and testtrial.cursor_loc == 0:
                        #         event.clearEvents()

                        #         examplesLeft -= 1  # decrease by one example
                        #         timer.reset()
                        #         timer2.reset()
                        #         break



                        # core.wait(2.0)


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

                    if test_prob['A'][old_cursor_loc[0]][old_cursor_loc[1]] == 0:
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
                        if test_prob['A'][i][j] == 0 and trial.cursor_loc == [i, j]:
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


endofrun = visual.TextStim(win, text="End of run", height=0.25)
endofrun.draw()
win.flip()

core.wait(3.3)

win.close()
core.quit()