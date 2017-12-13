import unittest
import trading.stock_rl
import test.unit.data

BUY = trading.stock_rl.ACTION['buy']
SELL = trading.stock_rl.ACTION['sell']
NOOP = trading.stock_rl.ACTION['noop']

class StateGeneratorTestCase(unittest.TestCase):

    def test_paths(self):
        data = [(1,), (1,), (1,),]
        paths = [
            path
            for path in trading.stock_rl.state_generator(data, horizon=3)
        ]
        expected = [
            (1, 0, 0, BUY,), (1, 1, 1, SELL,),
            (1, 0, 0, BUY,), (1, 1, 1, NOOP,), (1, 1, 3, SELL,),
            (1, 0, 0, NOOP,), (1, 0, 0, BUY,), (1, 1, 1, SELL,),
        ]
        self.assertEqual(expected, paths)

    def test_staggered_paths(self):
        data = [(1,), (1,), (2,),]
        paths = [
            path
            for path in trading.stock_rl.state_generator(data, horizon=3)
        ]
        expected = [
            (1, 0, 0, BUY,), (1, 1, 1, SELL,),
            (1, 0, 0, BUY,), (1, 1, 1, NOOP,), (2, 1, 5, SELL,),
            (1, 0, 0, NOOP,), (1, 0, 0, BUY,), (2, 1, 2, SELL,),
        ]
        self.assertEqual(expected, paths)

    def test_learning(self):
        data = test.unit.data.data

        learner = trading.stock_rl.Learner()
        learner.learn(
            lambda: trading.stock_rl.state_generator(data, horizon=10),
            iterations=1
        )

        states = [
            (1, 0, 0, 1,),
            (0, 0, 0, 1,),
            (1, 1, 0, 1,),
            (1, 1, 1, 1,),
        ]
        expecteds = [
            BUY,
            NOOP,
            NOOP,
            SELL,
        ]
        for index, state in enumerate(states):
            *_, action = learner.argmax(learner.discretize(state))
            self.assertEqual(action, expecteds[index])
