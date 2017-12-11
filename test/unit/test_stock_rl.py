import unittest
import trading.stock_rl

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
        ]
        self.assertEqual(expected, paths)
