import unittest
import trading.stock_rl
import test.unit.data

BUY = trading.stock_rl.ACTION['buy']
SELL = trading.stock_rl.ACTION['sell']
WAIT = trading.stock_rl.ACTION['wait']
PROMISE = trading.stock_rl.ACTION['promise']

class StateGeneratorTestCase(unittest.TestCase):

    def test_paths(self):
        # b s - - - -
        # b s b s - -
        # b s b s b s
        # b s b - s -
        # b s b - - s
        # b s - b s -
        # b s - b - s
        # b s - - b s
        # b - s - - -
        # b - s b s -
        # b - s b - s
        # b - s - b s
        # b - - s - -
        # b - - s b s
        # b - - - s -
        # - b s - - -
        # - b s b s -
        # - b s b - s
        # - b s - b s
        # - b - s - -
        # - b - s b s
        # - b - - s -
        # - - b s - -
        # - - b s b s
        # - - b - s -
        # - - b - - s
        # - - - b s -
        # - - - b - s
        # - - - - b s
        data = [(0, 0,), (0, 1,), (0, 2), (0, 0,)]
        expected = [
            [(0, 0, 'none', 0, BUY,), (0, 1, 'own', 0, SELL),],
            [(0, 0, 'none', 0, BUY,), (0, 1, 'own', 0, SELL), (0, 2, 'none', 0, BUY), (0, 0, 'own', 0, SELL),],
            [(0, 0, 'none', 0, BUY,), (0, 1, 'own', 0, PROMISE), (0, 2, 'own', 0, SELL),],
            [(0, 0, 'none', 0, BUY,), (0, 1, 'own', 0, PROMISE), (0, 2, 'own', 0, PROMISE), (0, 0, 'own', 0, SELL),],
            [(0, 0, 'none', 0, WAIT,), (0, 1, 'none', 0, WAIT), (0, 2, 'none', 0, BUY), (0, 0, 'own', 0, SELL),],
            [(0, 0, 'none', 0, 'wait'), (0, 1, 'none', 0, 'buy'), (0, 2, 'own', 0, 'promise'), (0, 0, 'own', 0, 'sell')],
            [(0, 0, 'none', 0, 'wait'), (0, 1, 'none', 0, 'buy'), (0, 2, 'own', 0, 'sell')],
        ]
        paths = list(map(lambda x: list(x), [
            path
            for path in trading.stock_rl.state_generator(data, horizon=4)
        ]))
        print('expected', expected)
        print('paths', paths)
        self.assertEqual(expected, paths)

    def test_learning(self):
        data = test.unit.data.data

        learner = trading.stock_rl.Learner()
        learner.learn(
            lambda: trading.stock_rl.state_generator(data, horizon=1),
        )

        episodes = [
            [(1, 0, 'none', 0, 'buy',),],
            [(0, 0, 'none', 0, 'buy',),],
            [(1, 0, 'own', 0, 'sell',),],
            [(1, 0, 'own', 1, 'sell',),],
        ]
        expecteds = [
            trading.stock_rl.DACTION['promise'],
            trading.stock_rl.DACTION['promise'],
            trading.stock_rl.DACTION['promise'],
            trading.stock_rl.DACTION['promise'],
        ]
        for index, episode in enumerate(episodes):
            for state in episode:
                *_, action = learner.argmax(learner.discretize(state))
                self.assertEqual(action, expecteds[index])
