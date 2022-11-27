import numpy as np

class TestQLearningActor:
    def __init__(self, qlearner, epsilon=0.05):
        self.qlearner = qlearner
        self.epsilon = epsilon

    def choose(self, sim_state):
        # action = choose(sim_state) : Called during learning with a 
        #   SimState object, and returns an action to be taken -- ie. a 
        #   number in [0, num_actions-1).

        if np.random.random() < self.epsilon:
            return np.random.choice(sim_state.get_num_actions())
        else:
            return np.argmax(self.qlearner.Q[sim_state.get_state_a()])
    
    def generate_policy(self, num_states, num_actions):
        return np.argmax(self.qlearner.Q, axis=1)
