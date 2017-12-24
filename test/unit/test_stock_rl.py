import unittest
import trading.stock_rl
import test.unit.data

BUY = trading.stock_rl.ACTION['buy']
SELL = trading.stock_rl.ACTION['sell']
WAIT = trading.stock_rl.ACTION['wait']
PROMISE = trading.stock_rl.ACTION['promise']
revmap = trading.stock_rl.revmap

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
            (0, 0, 0, 0, BUY,), (0, 1, 1, 0, SELL),
            (0, 0, 0, 0, BUY,), (0, 1, 1, 0, SELL), (0, 2, 0, 0, BUY), (0, 0, 1, 0, SELL),
            (0, 0, 0, 0, BUY,), (0, 1, 1, 0, PROMISE), (0, 2, 2, 0, SELL),
            (0, 0, 0, 0, BUY,), (0, 1, 1, 0, PROMISE), (0, 2, 2, 0, PROMISE), (0, 0, 3, 0, SELL),
            (0, 0, 0, 0, WAIT,), (0, 1, 0, 0, WAIT), (0, 2, 0, 0, BUY), (0, 0, 1, 0, SELL),
            (0, 0, 0, 0, WAIT), (0, 1, 0, 0, BUY), (0, 2, 1, 0, PROMISE), (0, 0, 2, 0, SELL),
            (0, 0, 0, 0, WAIT), (0, 1, 0, 0, BUY), (0, 2, 1, 0, SELL),
        ]
        paths = [
            path
            for path in trading.stock_rl.state_generator(data, horizon=4)
        ]
        print('paths', paths)
        self.assertEqual(expected, paths)

    def test_learning(self):
        data = test.unit.data.data

        learner = trading.stock_rl.Learner()
        learner.learn(
            lambda: trading.stock_rl.state_generator(data, horizon=6),
            iterations=1,
            alpha=0.3,
            gamma=0.7
        )

        states = [
            (0, 0, 0, 0, None,),
            (-0.5, 1, 0, 0, None,),
            (1, 0, 1, 0, None,),
            (1, 0, 1, 1, None,),
        ]
        expecteds = [
            trading.stock_rl.DACTION['buy'],
            trading.stock_rl.DACTION['wait'],
            trading.stock_rl.DACTION['sell'],
            trading.stock_rl.DACTION['sell'],
        ]
        for index, state in enumerate(states):
            *_, action = learner.argmax(learner.discretize(state))
            print(state)
            self.assertEqual(revmap[str(action)], revmap[str(expecteds[index])])
