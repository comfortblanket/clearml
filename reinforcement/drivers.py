
class SimState:
    def __init__(self, env):
        self._env = env

        self._episode = None
        self._episode_step = None

        # Learning rate, or step size. Determines to what extent newly 
        # acquired information should override old information. A factor of 0 
        # should makes the agent learn nothing (exclusively exploiting prior 
        # knowledge), while a factor of 1 should make the agent consider only 
        # the most recent information (ignoring prior knowledge to explore 
        # possibilities). In a fully deterministic environment, a learning 
        # rate of 1 is optimal. When the problem is stochastic, the Q-Learning 
        # algorithm converges under some technical conditions on the learning 
        # rate that require it to decrease to zero. In practice, often a 
        # constant learning rate is used with Q-Learning, such as 0.1.
        self._alpha = None

        # Discount factor. Determines the importance which should be given to 
        # future rewards. A factor of 0 should make the agent "myopic" (or 
        # short-sighted) by only considering current rewards, while a factor 
        # approaching 1 should make it strive for a long-term high reward. If 
        # the discount factor meets or exceeds 1, the action values may 
        # diverge. For a value of 1, without a terminal state, or if the agent 
        # never reaches one, all environment histories become infinitely long, 
        # and utilities with additive, undiscounted rewards generally become 
        # infinite. Even with a discount factor only slightly lower than 1, 
        # Q-function learning leads to propagation of errors and instabilities 
        # when the value function is approximated with an artificial neural 
        # network. In that case, starting with a lower discount factor and 
        # increasing it towards its final value accelerates learning.
        self._gamma = None

        self._state_a = None
        self._action_a = None
        self._reward_a = None
        self._state_b = None
        self._done = False
    
    def get_env(self):
        return self._env
    
    def get_episode(self):
        return self._episode
    def set_episode(self, e=None):
        self._episode = e
    def inc_episode(self, d=1):
        self._episode += d
    
    def get_episode_step(self):
        return self._episode_step
    def set_episode_step(self, s=None):
        self._episode_step = s
    def inc_episode_step(self, d=1):
        self._episode_step += d
    
    def get_alpha(self):
        return self._alpha
    def set_alpha(self, a=None):
        self._alpha = a
        
    def get_gamma(self):
        return self._gamma
    def set_gamma(self, g=None):
        self._gamma = g
        
    def get_state_a(self):
        return self._state_a
    def set_state_a(self, s=None):
        self._state_a = s
        
    def get_action_a(self):
        return self._action_a
    def set_action_a(self, a=None):
        self._action_a = a
    
    def get_reward_a(self):
        return self._reward_a
    def set_reward_a(self, r=None):
        self._reward_a = r
    
    def get_state_b(self):
        return self._state_b
    def set_state_b(self, s=None):
        self._state_b = s
    
    def get_done(self):
        return self._done
    def set_done(self, d=False):
        self._done = d
    
    def get_num_states(self):
        return self._env.get_num_states()
    
    def get_num_actions(self):
        return self._env.get_num_actions()


class EnvironmentDriver:

    def __init__(self, env, actor, alpha_setter, gamma_setter, learner, training_stopper=None, run_stopper=None):
        self.env = env
        self.actor = actor
        self.alpha_setter = alpha_setter
        self.gamma_setter = gamma_setter
        self.learner = learner
        self.training_stopper = training_stopper
        self.run_stopper = run_stopper
    

    def run_training_episode(self, episode_number):
        sim_state = SimState(self.env)
        sim_state.set_episode( episode_number )
        sim_state.set_episode_step( -1 )

        sim_state.set_state_a( self.env.reset() )

        while not sim_state.get_done():
            sim_state.inc_episode_step()

            sim_state.set_action_a( self.actor.choose(sim_state) )
            sim_state.set_alpha( self.alpha_setter.get_value(sim_state) )
            sim_state.set_gamma( self.gamma_setter.get_value(sim_state) )

            state_b, reward_a, done, info = self.env.step( sim_state.get_action_a() )

            sim_state.set_reward_a( reward_a )
            sim_state.set_state_b( state_b )
            sim_state.set_done( done )

            self.learner.update( sim_state )

            if self.training_stopper is not None:
                if self.training_stopper.should_stop( sim_state ):
                    break
            elif sim_state.get_done():
                break

            sim_state.set_state_a( sim_state.get_state_b() )

            # print(sim_state.get_episode(), sim_state.get_episode_step())
            # self.env.render(True)
            # print()
            
            # Reset
            sim_state.set_alpha()
            sim_state.set_gamma()
            sim_state.set_action_a()
            sim_state.set_reward_a()
            sim_state.set_state_b()
            sim_state.set_done()
        
        return sim_state
    

    def run_training_episodes(self, num_episodes):
        for episode in range(num_episodes):
            self.run_training_episode(episode)
    

    def run_episode(self, episode_number):
        sim_state = SimState(self.env)
        sim_state.set_episode( episode_number )
        sim_state.set_episode_step( -1 )

        sim_state.set_state_a( self.env.reset() )

        total_reward = 0

        while not sim_state.get_done():
            sim_state.inc_episode_step()

            sim_state.set_action_a( self.actor.choose(sim_state) )

            state_b, reward_a, done, info = self.env.step( sim_state.get_action_a() )
            total_reward += reward_a

            sim_state.set_reward_a( reward_a )
            sim_state.set_state_b( state_b )
            sim_state.set_done( done )

            if self.run_stopper is not None:
                if self.run_stopper.should_stop( sim_state ):
                    break
            elif sim_state.get_done():
                break

            sim_state.set_state_a( sim_state.get_state_b() )

            print(sim_state.get_episode(), sim_state.get_episode_step())
            self.env.render(True)
            print()
            
            # Reset
            sim_state.set_action_a()
            sim_state.set_reward_a()
            sim_state.set_state_b()
            sim_state.set_done()
        
        return total_reward
