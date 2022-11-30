from utils.smart_api import cwrp


# class SimState():
        
#     env: None = None
#     episode: int = None
#     episode_step = None
#     alpha = None
#     gamma = None
#     state_a = None
#     action_a = None
#     reward_a = None
#     state_b = None
#     done = False

#     def reset_for_next_step(self):
#         self.alpha = None
#         self.gamma = None
#         self.action_a = None
#         self.reward_a = None
#         self.state_b = None
#         self.done = False

    # def __init__(self, env):
    #     self._env = env

    #     self._episode = None
    #     self._episode_step = None

    #     # Learning rate, or step size. Determines to what extent newly 
    #     # acquired information should override old information. A factor of 0 
    #     # should makes the agent learn nothing (exclusively exploiting prior 
    #     # knowledge), while a factor of 1 should make the agent consider only 
    #     # the most recent information (ignoring prior knowledge to explore 
    #     # possibilities). In a fully deterministic environment, a learning 
    #     # rate of 1 is optimal. When the problem is stochastic, the Q-Learning 
    #     # algorithm converges under some technical conditions on the learning 
    #     # rate that require it to decrease to zero. In practice, often a 
    #     # constant learning rate is used with Q-Learning, such as 0.1.
    #     self._alpha = None

    #     # Discount factor. Determines the importance which should be given to 
    #     # future rewards. A factor of 0 should make the agent "myopic" (or 
    #     # short-sighted) by only considering current rewards, while a factor 
    #     # approaching 1 should make it strive for a long-term high reward. If 
    #     # the discount factor meets or exceeds 1, the action values may 
    #     # diverge. For a value of 1, without a terminal state, or if the agent 
    #     # never reaches one, all environment histories become infinitely long, 
    #     # and utilities with additive, undiscounted rewards generally become 
    #     # infinite. Even with a discount factor only slightly lower than 1, 
    #     # Q-function learning leads to propagation of errors and instabilities 
    #     # when the value function is approximated with an artificial neural 
    #     # network. In that case, starting with a lower discount factor and 
    #     # increasing it towards its final value accelerates learning.
    #     self._gamma = None

    #     self._state_a = None
    #     self._action_a = None
    #     self._reward_a = None
    #     self._state_b = None
    #     self._done = False
    
    # def get_env(self):
    #     return self._env
    
    # def get_episode(self):
    #     return self._episode
    # def set_episode(self, e=None):
    #     self._episode = e
    # def inc_episode(self, d=1):
    #     self._episode += d
    
    # def get_episode_step(self):
    #     return self._episode_step
    # def set_episode_step(self, s=None):
    #     self._episode_step = s
    # def inc_episode_step(self, d=1):
    #     self._episode_step += d
    
    # def get_alpha(self):
    #     return self._alpha
    # def set_alpha(self, a=None):
    #     self._alpha = a
        
    # def get_gamma(self):
    #     return self._gamma
    # def set_gamma(self, g=None):
    #     self._gamma = g
        
    # def get_state_a(self):
    #     return self._state_a
    # def set_state_a(self, s=None):
    #     self._state_a = s
        
    # def get_action_a(self):
    #     return self._action_a
    # def set_action_a(self, a=None):
    #     self._action_a = a
    
    # def get_reward_a(self):
    #     return self._reward_a
    # def set_reward_a(self, r=None):
    #     self._reward_a = r
    
    # def get_state_b(self):
    #     return self._state_b
    # def set_state_b(self, s=None):
    #     self._state_b = s
    
    # def get_done(self):
    #     return self._done
    # def set_done(self, d=False):
    #     self._done = d
    
    # def get_num_states(self):
    #     return self._env.get_num_states()
    
    # def get_num_actions(self):
    #     return self._env.get_num_actions()


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

        sim = {
            "env": self.env, 
            "num_states": self.env.get_num_states(), 
            "num_actions": self.env.get_num_actions(), 
            "episode": episode_number, 
            "episode_step": -1, 
            "episode_reward": 0, 
            "done": False, 
        }

        sim["state_a"] = self.env.reset()

        while not sim["done"]:
            sim["episode_step"] += 1

            # Choose action to take
            sim["action_a"] = cwrp(sim, self.actor.choose)

            if training:
                sim["alpha"] = cwrp(sim, self.alpha_setter.get_value)
                sim["gamma"] = cwrp(sim, self.gamma_setter.get_value)

            # Take the action
            sim["state_b"], sim["reward_a"], sim["done"], info = self.env.step( sim["action_a"] )
            sim["episode_reward"] += sim["reward_a"]
            
            # Give feedback to learner so it can learn
            if training:
                cwrp(sim, self.learner.update)

            # Check stop conditions
            if stopper is not None:
                if cwrp(sim, stopper.should_stop):
                    break
            elif sim["done"]:
                break
            
            # State A should now point to our new state
            sim["state_a"] = sim["state_b"]

            if render:
                print("Episode {}, step {}:".format(sim["episode"], sim["episode_step"]))
                self.env.render(new_only)
                print()

            # Remove/reset values from sim state dict which are no longer valid
            if training:
                del sim["alpha"], sim["gamma"]
            del sim["action_a"], sim["reward_a"], sim["state_b"]
            sim["done"] = False
        
        return sim if training else sim["episode_reward"]
    

    def run_training_episodes(self, num_episodes):
        for episode in range(num_episodes):
            self.run_episode(episode, training=True)
