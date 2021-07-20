#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:09:04 2021

@author: aliciachen
"""

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


                            allKeys1 = event.getKeys()
                            len_allkeys1 = len(allKeys1)
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