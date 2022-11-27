
class EpisodeStepStopper:
    def __init__(self, num_steps):
        self.num_steps = num_steps
    
    def should_stop(self, sim_state):
        return sim_state.get_episode_step() >= self.num_steps
