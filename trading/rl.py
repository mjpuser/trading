import numpy as np

def default_actions_filter(state):
    return [0]

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
        value = ((1 - alpha) * self.table[ds0]) + (alpha * (self.r(s0) + (gamma * self.table[s1max])))
        self.table[ds0] = value
        return self.table

    def argmax(self, state):
        *state, _ = state
        action_indexes = self.actions_filter(state)
        allowed_states = (*state, action_indexes)
        best_action = np.argmax(self.table[allowed_states])
        ret = *state, action_indexes[best_action]
        return ret

    def learn(self, states, iterations=10, alpha=0.3, gamma=0.9):
        for i in range(iterations):
            print('starting iteration {}'.format(i))
            states_iter = states()
            s0 = next(states_iter)
            for s1 in states_iter:
                print(s0)
                self.update(s0, s1, alpha, gamma)
                s0 = s1
            # include last state
            self.table[self.discretize(s0)] += self.r(s0)
        # print('table', self.table)
