import numpy as np

from utils.math import prob_choose
from utils.strings import str_find_all
from utils.terminal_formatting import TerminalFormatting as tf

class MazeEnvironment:
    def __init__(
                self, 
                map = [
                    "   G", 
                    " W X", 
                    "S   ", 
                ], 
                rewards = {
                    " " : -0.1,   # Empty
                    "S" : -0.1,   # Start (same as empty)
                    "G" : 1,      # Goal
                    "X" : -1,     # Pit, bad
                }, 
                move_prob = 0.7, 
                starts = "S", 
                goals = "G", 
                walls = "W", 
            ):
        
        for i,row in enumerate(map[1:]):
            assert len(map[0]) == len(row), \
                f"Row {i} is {len(row)} wide, but must match the first row's width of {len(map[0])}"

        self._map = map.copy()
        self._flat_map = "".join(self._map)
        self._num_states = len(self._flat_map)

        assert set(rewards.keys()).isdisjoint(walls), "Walls should not have a reward value assigned"
        assert set(rewards.keys()).issuperset(starts), "Start states must have a reward value assigned"
        assert set(rewards.keys()).issuperset(goals), "Goal states must have a reward value assigned"
        assert set(rewards.keys()).union(walls).issuperset(set([c for row in map for c in row])), \
            "All characters in the map must be walls or have a reward value defined"
        assert set(self._flat_map).intersection(starts), "Map must contain at least one start location"
        
        assert goals is None or set(self._flat_map).intersection(goals), \
            "Map must contain at least one goal location, or else goals must be set to None -- but in " \
            "that case note that step() will never return done=True, so you must have some other way " \
            "to terminate episodes/runs, for example using a stopper passed to the EnvironmentDriver"

        if goals is None:
            goals = ""
        
        # For each character in <starts/goals/walls>, get a list together of 
        # all of their locations in the flat map (which is equivalent to their 
        # state). Create a set() of those to get only unique values, then 
        # create a sorted list of those unique values.
        self._start_states = sorted(set([c for cc in starts for c in str_find_all(self._flat_map, cc)]))
        self._goal_states  = sorted(set([c for cc in goals  for c in str_find_all(self._flat_map, cc)]))
        self._wall_states  = sorted(set([c for cc in walls  for c in str_find_all(self._flat_map, cc)]))
        
        self._rewards = rewards.copy()
        
        # Probability of moving 90 left (and the probability of moving 90 
        # degrees right) of where the action intended to go
        side_prob = (1 - move_prob) / 2

        # Actions map to tuples containing (p, (x, y)) where p is the 
        # probability of the action resulting in offset x,y from the current 
        # location. Probabilities for a given action must sum to 1.
        self._actions = {
            0 : [(move_prob, (-1,  0)), (        0, ( 1,  0)), (side_prob, ( 0, -1)), (side_prob, ( 0,  1))], 
            1 : [(        0, (-1,  0)), (move_prob, ( 1,  0)), (side_prob, ( 0, -1)), (side_prob, ( 0,  1))], 
            2 : [(side_prob, (-1,  0)), (side_prob, ( 1,  0)), (move_prob, ( 0, -1)), (        0, ( 0,  1))], 
            3 : [(side_prob, (-1,  0)), (side_prob, ( 1,  0)), (        0, ( 0, -1)), (move_prob, ( 0,  1))], 
        }
        self._num_actions = len(self._actions)

        self._cur_state = np.random.choice(self._start_states)

        # The string representation of the environment last rendered by a call 
        # to render()
        self._last_render_string = ""


    def render(self, only_new=False, do_print=True, colorize_return=False):
        # Returns a string representation of the environment. do_print 
        # controls whether it also gets printed to stdout. If do_print is 
        # true, only_new controls whether only representations which differ 
        # from the previously rendered representation are displayed, or 
        # whether they all are.

        render_map = [list(s) for s in self._map]
        cur_loc = self._state_to_pos(self._cur_state)
        render_map[cur_loc[1]][cur_loc[0]] = "o"
        render_string = "\n".join("".join(line) for line in render_map)
        if not only_new or render_string != self._last_render_string:
            if do_print:
                print(render_string)
            self._last_render_string = render_string
            return render_string
        return ""


    def render_policy(self, policy, do_print=True, format_dict=None):
        # Returns a string representation of the given policy in the 
        # environment. do_print controls whether it also gets printed to 
        # stdout.

        if format_dict is None:
            format_dict = {}

        render_map = [list(s) for s in self._map]
        action_map = "<>^v"
        for state in range(self._num_states):
            loc = self._state_to_pos(state)
            
            if state not in self._wall_states and state not in self._goal_states:
                render_map[loc[1]][loc[0]] = action_map[policy[state]]
            
            fmt_a, fmt_b = "", ""
            state_type = self._flat_map[state]
            if state_type in format_dict:
                fmt_ab = format_dict[state_type]
                if isinstance(fmt_ab, tuple):
                    fmt_a, fmt_b = fmt_ab
                else:
                    fmt_a = fmt_ab
                    fmt_b = tf.DEFAULT
            elif state in self._start_states:
                fmt_a = tf.BG_LIGHT_BLUE
                fmt_b = tf.DEFAULT
            elif state in self._goal_states:
                fmt_a = tf.BG_GREEN
                fmt_b = tf.DEFAULT
            elif state in self._wall_states:
                fmt_a = tf.BG_GRAY
                fmt_b = tf.DEFAULT
            render_map[loc[1]][loc[0]] = fmt_a + render_map[loc[1]][loc[0]] + fmt_b

        render_string = "\n".join("".join(line) for line in render_map)
        if do_print:
            print(render_string)
        return render_string


    def _state_to_pos(self, state):
        return (state % len(self._map[0]), state // len(self._map[0]))
    
    def _pos_to_state(self, pos):
        return pos[0] + pos[1] * len(self._map[0])


    def get_num_actions(self):
        # num_actions = get_num_actions() : Returns the number of actions 
        #   possible in the environment. Note that whenever a value denotes an 
        #   'action', it means an integer between 0 and num_actions-1.
        return self._num_actions
    
    def get_num_states(self):
        # num_states = get_num_states() : Returns the number of states it is 
        #   possible to visit in the environment. Note that whenever a value 
        #   denotes a 'state', it means an integer between 0 and num_states-1.
        return self._num_states
    
    def reset(self):
        # state = reset() : Sets up the environment to begin a new run, and 
        #   returns the starting state.

        self._cur_state = np.random.choice(self._start_states)
        return self._cur_state
    
    def step(self, action):
        # next_state, reward, done, info = step(action) - Takes the given 
        #   action, and returns the new state and the reward received, whether 
        #   the state is terminal (and so the reset method should be called), 
        #   and some info about the simulation status (currently ignored).

        # State

        cur_pos = self._state_to_pos(self._cur_state)
        actual_action = prob_choose(np.random.random(), *self._actions[action])
        next_pos = (
            cur_pos[0] + actual_action[0], 
            cur_pos[1] + actual_action[1], 
        )
        next_state = self._pos_to_state(next_pos)
        
        if (
                    next_pos[0] < 0 or 
                    next_pos[1] < 0 or 
                    next_pos[0] >= len(self._map[0]) or 
                    next_pos[1] >= len(self._map) or 
                    next_state in self._wall_states
                ):
            next_state = self._cur_state
        self._cur_state = next_state
        
        # Reward
        reward = self._rewards[self._flat_map[self._cur_state]]

        # Done
        done = self._cur_state in self._goal_states
        
        # Info
        info = None

        return next_state, reward, done, info
