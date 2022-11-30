from reinforcement.actors import TestQLearningActor
from reinforcement.drivers import EnvironmentDriver
from reinforcement.environments import MazeEnvironment
from reinforcement.learners import DiscreteQLearner
from reinforcement.setters import ConstantValueSetter
from reinforcement.stoppers import EpisodeStepStopper

from utils.strings import str_find_all


def test_all():
    assert str_find_all("abcdabcdabcd", "d") == [3, 7, 11]
    assert str_find_all("abcdabcdabcd", "d", end=11) == [3, 7]
    assert str_find_all("abcdabcdabcd", "cd") == [2, 6, 10]
    assert str_find_all("abcdabcdabcdabcd", "abcda", overlaps=False) == [0, 8]
    assert str_find_all("abcdabcdabcdabcd", "abcda", overlaps=True) == [0, 4, 8]

    env = MazeEnvironment(map="""\
            W             
    XXXXX   W   W         
        X   W   W         
        X   W   W   XXX   
        X   W   W  X   X X
 S      X       W  X G    
        X   W   X  X   WW 
        X   W   X   WWW   
        X   W   X         
    XXXXX   W   X         
            W             """.split("\n"))
    learner = DiscreteQLearner(env.get_num_states(), env.get_num_actions())
    actor = TestQLearningActor(learner)
    alpha_setter = ConstantValueSetter(0.1)
    gamma_setter = ConstantValueSetter(0.3)
    stopper = EpisodeStepStopper(200)
    driver = EnvironmentDriver(env, actor, alpha_setter, gamma_setter, learner, stopper, stopper)

    driver.run_training_episodes(300)

    policy = actor.generate_policy(env.get_num_states(), env.get_num_actions())
    env.render_policy(policy)

    print(
        driver.run_episode(-9999)
    )
    pass


if __name__ == "__main__":
    test_all()
