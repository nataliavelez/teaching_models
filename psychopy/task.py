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


# %% Import and format teaching problems

f = open('/Users/aliciachen/Dropbox/teaching_models/problems.json')
problems_raw = json.load(f)
f.close()

probs = {}

# transpose matrices to make indexing easier (index from bottom left)
probstest = []
for prob in problems_raw:
    thisprob = {}
    for h, val in prob.items():  # h is ABCD index
        thismtx = np.array(val)
        thismtxT = thismtx.T
        thisprob[h] = thismtxT.tolist()
    probstest.append(thisprob)


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
# f = open('subj_list.json')
f = open('../subj_list.json')
subj_list = json.load(f)
f.close()

prepared_probs = {}

for subj_id, runs in enumerate(subj_list):
    prepared_probs[subj_id] = {}
    for run_idx, run in enumerate(runs):
        prepared_probs[subj_id][run_idx] = {i: probs[i] for i in run}

# %% Access subject and run

subj_id = 1
run_idx = 2
probs = prepared_probs[subj_id][run_idx]

# %% Task
data = [] # dete later
expt_data = []

key_mapping = {
    'left': '1',
    'right': '4',
    'up': '2',
    'down': '3',
    'space': '0'
}

##
win = visual.Window([800, 600], monitor="test", units="norm", color='black')
#win = visual.Window(size=[1280, 700], fullscr=True, screen=1, winType='pyglet', color='black')

iti = visual.TextStim(win=win, text="+", height=.2)

waitTriggerText = visual.TextStim(win, text="Waiting for scanner trigger...")
waitTriggerText.draw()

win.flip()

while True:
    trigger = event.waitKeys()
    print(trigger)
    if trigger[0] == 'equal':
        break

exptTimer = core.Clock()

iti.draw()
win.flip()


core.wait(10)

trial_no = -1
event.clearEvents()


for prob_idx, v in probs.items():

    prob_dict = v

    # data[prob_idx] = {}
    # data[prob_idx]['prob'] = v

    problemFinished = False
    test_prob = v
    prob_idxs = list(test_prob.keys())

    # Shuffle order of four images
    random.shuffle(prob_idxs)
    test_prob_shuffled = {i: test_prob[i] for i in prob_idxs}
    true_h_idx = prob_idxs.index('h1')  # Extract order of index
    # data[prob_idx]['true_h_idx'] = true_h_idx

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
    examplesLeft = maxExamples

    # fixation cross
    iti = visual.TextStim(win=win, text="+", height=.2)

    ######## 1. Study

    trial_type = 'study'
    trial_no += 1
    problem = [prob_idx, prob_dict]
    onset = exptTimer.getTime()

    ## Draw initial study screen

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
    core.wait(27) # Study time, change to 15 later

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

    # get dur, save data from study block
    dur = exptTimer.getTime() - onset

    trial_data = {
    'trial_type': trial_type,
    'trial_no': trial_no,
    'problem': problem,
    'onset': onset,
    'dur': dur
    }

    expt_data.append(trial_data)




    # Starting cursor color for start of interactivity
    if test_prob['h1'][0][0] == 0:
        rects[0][0].lineColor = (218, 60, 37)  # red
        rects[0][0].lineWidth = 15
    else:
        rects[0][0].lineColor = (72, 160, 248)  # blue
        rects[0][0].lineWidth = 15

    timer = core.CountdownTimer(8)

    changeFlag = True

    # Start
    while examplesLeft > 0:

        if changeFlag == True:
            # TODO: start Choose block
            print('change block')
            start_loc = trial.cursor_loc.copy()
            moves = []
            response = None
            state = None
            rt = None

            trial_type = 'choose'
            trial_no += 1
            problem = [prob_idx, prob_dict]
            onset = exptTimer.getTime()

        changeFlag = False

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

            ## Save data from CHOOSE

            dur = exptTimer.getTime() - onset

            trial_data = {
            'trial_type': trial_type,
            'trial_no': trial_no,
            'problem': problem,
            'onset': onset,
            'dur': dur,
            'start_loc': start_loc,
            'moves': moves,
            'response': None,
            'state': trial.squares, # might need to change, include current or no
            'rt': None
            }

            expt_data.append(trial_data)


            # Start of CONT block
            trial_type = 'cont'
            trial_no += 1
            problem = [prob_idx, prob_dict]
            onset = exptTimer.getTime()


            if examplesLeft == 1:
                break

            len_allKeys1 = 0
            timer1 = core.CountdownTimer(2)


            while True: # Move on and stay buttons



                leftarrow.draw()
                rightarrow.draw()
                cont.draw()
                yestext.draw()
                notext.draw()



                win.flip()

                allKeys1 = event.getKeys()

                if timer1.getTime() < 0:



                    examplesLeft -= 1

                    # Data

                    dur = exptTimer.getTime() - onset

                    trial_data = {
                    'trial_type': trial_type,
                    'trial_no': trial_no,
                    'problem': problem,
                    'onset': onset,
                    'dur': dur,
                    'response': True,
                    'rt': None,
                    'remaining': examplesLeft
                    }

                    expt_data.append(trial_data)

                    # Other stuff
                    event.clearEvents()

                      # decrease by one example
                    timer.reset()
                    timer1.reset()
                    changeFlag = True
                    break


                if len(allKeys1) > len_allKeys1:

                    thisKey = allKeys1[-1]
                    if thisKey == key_mapping['space']:

                        event.clearEvents()

                        examplesLeft -= 1  # decrease by one example

                        # Data
                        dur = exptTimer.getTime() - onset

                        trial_data = {
                        'trial_type': trial_type,
                        'trial_no': trial_no,
                        'problem': problem,
                        'onset': onset,
                        'dur': dur,
                        'response': True,
                        'rt': dur,
                        'remaining': examplesLeft
                        }

                        expt_data.append(trial_data)


                        timer.reset()
                        timer1.reset()
                        changeFlag = True
                        break

                    elif thisKey == key_mapping['right']:

                        #examplesLeft = 0

                        # Data
                        dur = exptTimer.getTime() - onset

                        trial_data = {
                        'trial_type': trial_type,
                        'trial_no': trial_no,
                        'problem': problem,
                        'onset': onset,
                        'dur': dur,
                        'response': False,
                        'rt': dur,
                        'remaining': examplesLeft
                        }

                        expt_data.append(trial_data)


                        problemFinished = True
                        changeFlag = True
                        break



                len_allKeys1 = len(allKeys1)
                event.clearEvents()

            core.wait(0.5)

        if problemFinished:
                break # This is a probably better way of doing it than incrementing examplesLeft and waiting for the loop to quit

        #nKeys = 0
        allKeys = event.getKeys()
        #allowed_keys = ['left', 'right', 'up', 'down', 'enter', 'q']


        # Canvas part

        old_cursor_loc = trial.cursor_loc.copy()

        if len(allKeys) > nKeys:
            thisKey = allKeys[-1]

            if thisKey == key_mapping['left']:
                trial.left()
                clickable = False if (trial.squares[trial.cursor_loc[0]][trial.cursor_loc[1]] == 1) or (test_prob['h1'][trial.cursor_loc[0]][trial.cursor_loc[1]] == 0) else True
                moves.append([thisKey, exptTimer.getTime(), clickable])
            elif thisKey == key_mapping['right']:
                trial.right()
                clickable = False if (trial.squares[trial.cursor_loc[0]][trial.cursor_loc[1]] == 1) or (test_prob['h1'][trial.cursor_loc[0]][trial.cursor_loc[1]] == 0) else True
                moves.append([thisKey, exptTimer.getTime(), clickable])
            elif thisKey == key_mapping['up']:
                trial.up()
                clickable = False if (trial.squares[trial.cursor_loc[0]][trial.cursor_loc[1]] == 1) or (test_prob['h1'][trial.cursor_loc[0]][trial.cursor_loc[1]] == 0) else True
                moves.append([thisKey, exptTimer.getTime(), clickable])
            elif thisKey == key_mapping['down']:
                trial.down()
                clickable = False if (trial.squares[trial.cursor_loc[0]][trial.cursor_loc[1]] == 1) or (test_prob['h1'][trial.cursor_loc[0]][trial.cursor_loc[1]] == 0) else True
                moves.append([thisKey, exptTimer.getTime(), clickable])
            elif thisKey == key_mapping['space']:
                if test_prob['h1'][trial.cursor_loc[0]][trial.cursor_loc[1]] == 0:
                    print('error :( cant pick negative examples')
                else:
                    if trial.squares[trial.cursor_loc[0]][trial.cursor_loc[1]] == 0:

                        old_state = trial.squares  # For saving

                        trial.select()

                        new_state = trial.squares  # For saving


                        selected_coords = trial.cursor_loc  # For saving

                        ## Save data from CHOOSE

                        dur = exptTimer.getTime() - onset

                        trial_data = {
                        'trial_type': trial_type,
                        'trial_no': trial_no,
                        'problem': problem,
                        'onset': onset,
                        'dur': dur,
                        'start_loc': start_loc,
                        'moves': moves,
                        'response': selected_coords,
                        'state': new_state, # might need to change, include current or no
                        'rt': moves[-1][1] - onset
                        }

                        expt_data.append(trial_data)

                        ######

                        rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor = 'blue'

                        # End of time block
                        msg.draw()

                        win.flip(clearBuffer=True)

                        core.wait(random.uniform(1,3))

                        win.flip(clearBuffer=True)
                        #core.wait(random.uniform(1, 3)) # ISI: black screen




                        #win.flip(clearBuffer=True)

                        for h in hs:
                            h.draw()

                        true_h_border.draw()

                        for lett in lets:
                            lett.draw()

                        for i in range(6):
                            for j in range(6):
                                learner_rects[i][j].draw()

                        win.flip() # Onset of canvas shown to learner

                        trial_type = 'present'
                        trial_no += 1
                        problem = [prob_idx, prob_dict]
                        onset = exptTimer.getTime()

                        core.wait(0.5) # blank canvas before example shown to learner

                        # make selected example appear

                        for h in hs:
                            h.draw()

                        true_h_border.draw()

                        for lett in lets:
                            lett.draw()

                        # Show example to learner
                        learner_rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor=	(254, 223, 0) # fillColor=(72, 160, 248)

                        for i in range(6):
                            for j in range(6):
                                learner_rects[i][j].draw()

                        win.flip(clearBuffer=True)

                        exOnset = exptTimer.getTime()

                        core.wait(4)  # Amt of time new square is presesnted to learner

                        ## Save data

                        dur = exptTimer.getTime() - onset

                        trial_data = {
                        'trial_type': trial_type,
                        'trial_no': trial_no,
                        'problem': problem,
                        'onset': onset,
                        'dur': dur,
                        'old_state': old_state,
                        'new_state': new_state,
                        'selected_coords': selected_coords,
                        'ex_onset': exOnset
                        }

                        expt_data.append(trial_data)

                        ##
                        win.flip(clearBuffer=True)

                        core.wait(random.uniform(1, 3)) # ISI

                        if examplesLeft == 1:
                            break

                        # Feedback screen
                        event.clearEvents()

                        len_allKeys1 = 0
                        timer1 = core.CountdownTimer(2)

                        # previous learner rects are the same color
                        learner_rects[trial.cursor_loc[0]][trial.cursor_loc[1]].fillColor=(72, 160, 248)


                        # Another start of CONT
                        trial_type = 'cont'
                        trial_no += 1
                        problem = [prob_idx, prob_dict]
                        onset = exptTimer.getTime()

                        while True:



                            leftarrow.draw()
                            rightarrow.draw()
                            cont.draw()
                            yestext.draw()
                            notext.draw()



                            win.flip()

                            allKeys1 = event.getKeys()

                            if timer1.getTime() < 0:



                                examplesLeft -= 1

                                # Data

                                dur = exptTimer.getTime() - onset

                                trial_data = {
                                'trial_type': trial_type,
                                'trial_no': trial_no,
                                'problem': problem,
                                'onset': onset,
                                'dur': dur,
                                'response': True,
                                'rt': None,
                                'remaining': examplesLeft
                                }

                                expt_data.append(trial_data)

                                # Other stuff
                                event.clearEvents()

                                  # decrease by one example
                                timer.reset()
                                timer1.reset()
                                changeFlag = True
                                break


                            if len(allKeys1) > len_allKeys1:

                                thisKey = allKeys1[-1]
                                if thisKey == key_mapping['space']:

                                    event.clearEvents()

                                    examplesLeft -= 1  # decrease by one example

                                    # Data
                                    dur = exptTimer.getTime() - onset

                                    trial_data = {
                                    'trial_type': trial_type,
                                    'trial_no': trial_no,
                                    'problem': problem,
                                    'onset': onset,
                                    'dur': dur,
                                    'response': True,
                                    'rt': dur,
                                    'remaining': examplesLeft
                                    }

                                    expt_data.append(trial_data)


                                    timer.reset()
                                    timer1.reset()
                                    changeFlag = True
                                    break

                                elif thisKey == key_mapping['right']:

                                    #examplesLeft = 0

                                    # Data
                                    dur = exptTimer.getTime() - onset

                                    trial_data = {
                                    'trial_type': trial_type,
                                    'trial_no': trial_no,
                                    'problem': problem,
                                    'onset': onset,
                                    'dur': dur,
                                    'response': False,
                                    'rt': dur,
                                    'remaining': examplesLeft
                                    }

                                    expt_data.append(trial_data)


                                    problemFinished = True
                                    changeFlag = True
                                    break



                            len_allKeys1 = len(allKeys1)
                            event.clearEvents()

                        win.flip(clearBuffer=True)

                        # ITI and fixation cross
                        iti.draw()
                        win.flip()
                        core.wait(random.uniform(5, 7))


                    else:
                        trial.unselect()
                        # TODO: get rid of unselect

                # TODO: Display a message for selecting negative examples?

            elif thisKey == 'q':
                win.close()
                core.quit()

            print(thisKey)


            if problemFinished:
                break

            if thisKey != key_mapping['space']:

                try:
                    rects[old_cursor_loc[0]][old_cursor_loc[1]].lineColor = 'none'

                    if test_prob['h1'][old_cursor_loc[0]][old_cursor_loc[1]] == 0:
                        rects[old_cursor_loc[0]][old_cursor_loc[1]].fillColor = (94, 93, 95)  # dark gray
                    else:
                        rects[old_cursor_loc[0]][old_cursor_loc[1]].fillColor = (214, 214, 214)  # light gray
                    rects[trial.cursor_loc[0]][trial.cursor_loc[1]].lineColor = (72, 160, 248)  # blue
                    rects[trial.cursor_loc[0]][trial.cursor_loc[1]].lineWidth = 15
                except (IndexError, KeyError):
                    print('index out of range')  # TODO: change to a visual stim?



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
                        print('index out of range')  # TODO: change to a visual stim?


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


        #event.clearEvents()

        nKeys = len(allKeys)

    # Between teaching problems
    win.flip(clearBuffer=True)

    iti.draw()
    win.flip()
    core.wait(random.uniform(5, 7))


# endofrun = visual.TextStim(win, text="End of run", height=0.25)
iti.draw()
win.flip()

core.wait(10)

win.close()
core.quit()


#%%

# Save to json file
import datetime
out_tstamp = datetime.datetime.now().strftime('%y%m%d_%H%M')
out_fname = "/Users/aliciachen/Dropbox/teaching_models/psychopy/test_data/%s_run%i_%s.json" % (subj_id, run_idx, out_tstamp)

# actual_out_fname = "sub-%s_task_teaching_run%i_beh.json"
subj_list_json = json.dumps(subj_list)
jsonFile = open(out_fname, "w")
jsonFile.write(subj_list_json)
jsonFile.close()

# TODO: change file naming later