import unittest
import trading.stock_rl

BUY = trading.stock_rl.ACTION['buy']
SELL = trading.stock_rl.ACTION['sell']
NOOP = trading.stock_rl.ACTION['noop']

class StateGeneratorTestCase(unittest.TestCase):

    def test_paths(self):
        paths = [
            path
            for path in trading.stock_rl.state_generator(horizon=5)
        ]
        expected = [
            (BUY, SELL,),
            (BUY, NOOP, SELL,),
            (BUY, NOOP, NOOP, SELL,),
            (BUY, NOOP, NOOP, NOOP, SELL,),
        ]
        self.assertEqual(expected, paths)
