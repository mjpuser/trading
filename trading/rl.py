import numpy as np

def default_actions_filter(state):
    return None

def default_discretizer(state):
    return state

class Q:

    def __init__(self, r, shape, discretize=default_discretizer,
                 actions_filter=default_actions_filter, table=None):
        """
        Initializes the Q function.
        r is the reward function.
        """
        self.r = r
        self.discretize = discretize
        self.table = np.zeros(shape) if table is None else table
        self.actions_filter = actions_filter

    def update(self, s0, s1, alpha=0.3, gamma=0.9):
        ds0 = self.discretize(s0)
        ds1 = self.discretize(s1)
        s1max = self.argmax(ds1)
        # (1 - a)(Q[s0,a0]) + a(R[s0,a0] + g * Qmax[s1,a1])
        self.table[ds0] = ((1 - alpha) * self.table[ds0]) + (alpha * (self.r(s0) + (gamma * self.table[s1max])))
        return self.table

    def argmax(self, state):
        *state, _ = state
        # TODO be better at limitting allowed states for max
        allowed_states = *state, None
        best_action = np.argmax(self.table[allowed_states])
        ret = *state, best_action
        return ret

    def learn(self, episodes, iterations=10, alpha=0.3, gamma=0.9):
        for _ in range(iterations):
            for episode in episodes():
                s0 = next(episode)
                for s1 in episode:
                    self.update(s0, s1, alpha, gamma)
                    s0 = s1
                # include last state
                self.table[self.discretize(s0)] += self.r(s0)
