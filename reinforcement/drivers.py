from dataclasses import dataclass, field


@dataclass
class SimState():
    env           : object
    episode       : int
    episode_step  : int   = -1
    episode_reward: float = 0.0
    alpha         : float = None
    gamma         : float = None
    state_a       : int   = None
    action_a      : int   = None
    reward_a      : float = None
    state_b       : int   = None
    done          : bool  = False
    num_states    : int   = field(init=False)
    num_actions   : int   = field(init=False)

    def __post_init__(self):
        self.num_states  = self.env.get_num_states()
        self.num_actions = self.env.get_num_actions()

    def reset_for_next_step(self):
        self.alpha = None
        self.gamma = None
        self.action_a = None
        self.reward_a = None
        self.state_b = None
        self.done = False


class EnvironmentDriver:

    def __init__(self, env, actor, alpha_setter, gamma_setter, learner, training_stopper=None, run_stopper=None):
        self.env = env
        self.actor = actor
        self.alpha_setter = alpha_setter
        self.gamma_setter = gamma_setter
        self.learner = learner
        self.training_stopper = training_stopper
        self.run_stopper = run_stopper
    

    def run_episode(self, episode_number, training=False, render=False, new_only=True):
        stopper = self.training_stopper if training else self.run_stopper

        sim = SimState(self.env, episode_number)
        sim.state_a = self.env.reset()

        while not sim.done:
            # sim["episode_step"] += 1
            sim.episode_step += 1

            # Choose action to take
            sim.action_a = self.actor.choose(sim)

            if training:
                sim.alpha = self.alpha_setter.get_value(sim)
                sim.gamma = self.gamma_setter.get_value(sim)

            # Take the action
            sim.state_b, sim.reward_a, sim.done, info = self.env.step( sim.action_a )
            sim.episode_reward += sim.reward_a
            
            # Give feedback to learner so it can learn
            if training:
                self.learner.update(sim)

            # Check stop conditions
            if stopper is not None:
                if stopper.should_stop(sim):
                    break
            elif sim.done:
                break
            
            # State A should now point to our new state
            sim.state_a = sim.state_b

            if render:
                print("Episode {}, step {}:".format(sim.episode, sim.episode_step))
                self.env.render(new_only)
                print()

            # Reset values from sim state dict which are no longer valid
            sim.reset_for_next_step()
        
        return sim if training else sim.episode_reward
    

    def run_training_episodes(self, num_episodes):
        for episode in range(num_episodes):
            self.run_episode(episode, training=True)
