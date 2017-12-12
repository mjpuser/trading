import trading.rl
import unittest
import test.unit.data
import numpy as np

class QTestCase(unittest.TestCase):

    def setUp(self):
        def reward(state):
            val, = state
            return val
        self.q = trading.rl.Q(reward, (2,))

    def test_learning(self):
        states = lambda: iter([
            # change
            (0,),
            (1,),
        ])
        self.q.learn(states)
        expected = np.zeros((2,))
        expected[1] = 1
        np.testing.assert_array_equal(self.q.table, expected)
