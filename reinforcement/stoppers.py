
class EpisodeStepStopper:
    def __init__(self, num_steps):
        self.num_steps = num_steps
    
    def should_stop(self, episode_step):
        return episode_step >= self.num_steps
