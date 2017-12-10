import numpy as np

def default_actions_filter(state):
    return None

class Q:

    def __init__(self, r, shape, actions_filter=default_actions_filter, table=None):
        """
        Initializes the Q function.
        r is the reward function.
        """
        self.r = r
        self.table = np.zeros(shape) if table is None else table
        self.actions_filter = actions_filter

    def update(self, s0, s1, alpha=0.3, gamma=0.7):
        s1max = self.argmax(s1)
        self.table[s0] = self.table[s0] + self.r(s0) + self.table[s1max]
        return self.table

    def argmax(self, state):
        state = list(state)[:-1]
        allowed_states = tuple( state + [self.actions_filter(state)] )
        best_action = np.argmax(self.table[allowed_states])
        return tuple(state + [best_action])

    def learn(self, states):
        s0 = next(states)
        for s1 in states:
            self.update(s0, s1)
            s0 = s1
        # include last state
        self.table[s0] += self.r(s0)
