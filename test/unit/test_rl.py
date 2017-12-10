import trading.rl
import trading.stock_rl
import unittest
import test.unit.data
import numpy as np

BUY = trading.stock_rl.ACTION['buy']
SELL = trading.stock_rl.ACTION['sell']

class QTestCase(unittest.TestCase):

    def setUp(self):
        self.q = trading.stock_rl.Learner()

    def test_learning(self):
        states = iter([
            # change | owns | return | action
            (0, 0, 0, BUY,),
            (1, 1, 1, SELL,),
        ])
        self.q.learn(states)
        expected = np.zeros((2, 2, 2, 3,))
        expected[1, 1, 1, SELL] = 1
        np.testing.assert_array_equal(self.q.table, expected)
