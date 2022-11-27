
def prob_choose(rand_0_1, *prob_and_value):
    assert 0 <= rand_0_1 < 1
    assert sum(prob for prob,_ in prob_and_value) == 1
    prob_sum = 0
    for prob, value in prob_and_value:
        prob_sum += prob
        if prob_sum > rand_0_1:
            return value
    return value
