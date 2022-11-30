import numpy as np


class DiscreteQLearner:
    # Only handles discrete environments.

    def __init__(self, num_states, num_actions, dtype=float):
        self.num_states = num_states
        self.num_actions = num_actions

        # The Q table maps actions taken from each state to the expected value 
        # of taking that action from that state.
        self.Q = np.zeros(
            (self.num_states, self.num_actions), 
            dtype = dtype, 
        )
    
    def set_Q_values(self, newQ):
        self.Q[:,:] = newQ
    

    def update(self, alpha, gamma, state_a, action_a, reward_a, state_b):
        dQ = alpha * (
            reward_a + gamma * self.Q[state_b].max() - self.Q[state_a, action_a]
        )

        self.Q[state_a, action_a] += dQ
