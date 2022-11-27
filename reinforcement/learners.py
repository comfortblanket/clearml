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
    
    def update(self, sim_state):
        alpha = sim_state.get_alpha()
        gamma = sim_state.get_gamma()
        reward = sim_state.get_reward_a()
        new_state = sim_state.get_state_b()
        prev_state = sim_state.get_state_a()
        prev_action = sim_state.get_action_a()

        dQ = alpha * (
            reward + gamma * self.Q[new_state].max() - self.Q[prev_state, prev_action]
        )

        self.Q[prev_state, prev_action] += dQ
