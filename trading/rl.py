# reward function
#
#
#
# permutations
# x x x x x x
# x x x x x b
# x x x x b x
# x x x x b s
# x x x b x x
# x x x b s x
# x x x b x s
# x x b x x x
# x x b s x x
# x x b x s x
# x x b x x s
# x b x x x x
# x b s x x x
# x b x s x x
# x b x x s x
# x b x x x s
# b s x x x x
# b x s x x x
# b x x s x x
# b x x x s x
# b x x x x s
# TODO future optimization is to be able to concatenate runs.  Ex:
# b s x x
# x b s x
# x x b s
# could serve as any permutation of the horizon

class Q:

    data = None
    reward_function = None
    meta_update = None
    get_available_actions = None

    def learn(iterations=100, alpha=0.3, gamma=0.7):
        # iterate over state
        # generate metadata (metadata is run specific, basically like a session)
        # generate reward
        # get available actions,
        #   - generate tree
        #   - optimization lies in not being able to sell multiple times, or buy multiple times
        #   - basically just different lengths of holding
        # ensure we do all permutations, however, whats the point of iterations, then?
        #   - permutation generator
        #   - if we do permutations, we have to have multiple metas
        #   - we have to do like a double for loops
        pass

    def run(data):
        pass
