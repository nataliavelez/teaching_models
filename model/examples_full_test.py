
execfile('examples_full.py')

testprob = Problem(all_problems, 3)
testprob.view()
testprob.example_space()
testprob.selected_examples(((1, 1), (2, 1), (3, 1), (4, 1)))
# print(len(testprob.possible_exs_by_step[3]))

_, _ = testprob.literal()
_, _ = testprob.pragmatic(200)


_ = testprob.outputs()