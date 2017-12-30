import unittest
import trading.stock_rl
import test.unit.data

BOLLINGER = trading.stock_rl.BOLLINGER
ACTION = trading.stock_rl.ACTION

class StateGeneratorTestCase(unittest.TestCase):
    def test_discretizer(self):
        # takes change, bollinger, owns, buystate, return, action
        # returns to bollinger, owns, buystate, action
        state = tuple([
            0.01,
            'inside',
            True,
            'above',
            0.10,
            'buy',
        ])

        dstate = trading.stock_rl.discretize(state)

        expected = tuple([
            BOLLINGER['inside'],
            True,
            BOLLINGER['above'],
            ACTION['buy'],
        ])

        self.assertEqual(dstate, expected)
