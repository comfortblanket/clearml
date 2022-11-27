
# IMPORTANT DISTINCTION:
# 
# A "state" without any other qualifiers is refering to a Markov state of the 
# environment. For a given environment, there are states 0, 1, ..., N, for 
# some N specific to the environment.
# 
# A "SimState" is an object which holds values relevant to the simulation 
# (either during training or simply running it), such as the current episode 
# number, the learning rate, the current state (as in, the environment's 
# current state), the current action which has been chosen to perform from the 
# current state, the reward for performing that action, etc. Not all values 
# are available at all times.
