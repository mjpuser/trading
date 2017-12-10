import trading.rl
import trading.lib.tree
import unittest
import test.unit.data
import numpy as np

OWNS = 1

NOOP = 0
BUY = 1
SELL = 2


class QTestCase(unittest.TestCase):

    def setUp(self):

        def r(state):
            action = state[-1]
            if action == SELL:
                RETURN = 2
                return state[RETURN]
            return 0
        def actions_filter(state):
            if state[OWNS] == 1:
                return (NOOP, SELL,)
            else:
                return (NOOP, BUY,)
        self.q = trading.rl.Q(r=r, shape=(2, 2, 2, 3,), actions_filter=actions_filter)
        Searcher = trading.lib.tree.Searcher

    def test_learning(self):
        states = iter([
            # change | owns | return | action
            (0, 0, 0, 1,),
            (1, 1, 1, 2,),
        ])
        self.q.learn(states)
        expected = np.zeros((2,2,2,3,))
        expected[1, 1, 1, 2] = 1
        np.testing.assert_array_equal(self.q.table, expected)


# import test.unit
# import tranding.q
#
# # data has to have: pct change, actions, cummulative
# actions = ['buy', 'sell', None]
# def reward_function(state, action, meta):
#     if action == 'buy':
#         return 1
#     elif action == None and meta.get('days') > 0:
#         return meta['cummulative'] / 2.0
#     elif action == 'sell':
#         return 2.0 * meta['cummulative']
#
# # meta will have
# # - # of days
# # - cummulative
# def meta_update(state, meta, action):
#     if action == 'buy' and meta.get('days') == 0:
#         meta['days'] = 1
#     # calculate cummulative return
#     pass
#
# learner = trading.q.Learner()
#
# learner.data = test.unit.data
# learner.reward_function = reward_function
# learner.meta_update = meta_update
# # alpha is learning rate
# # gamma is discount factor
# learner.learn(iterations=100, alpha=0.2, gamma=0.8)
#
# history = learner.run(test.unit.data)
#
# def calculate_return(history):
#     # sum up cummulative return
#     pass
